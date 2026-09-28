"""MJLab adapter: reused sensor conventions and incremental DLS execution."""
from dataclasses import dataclass, field
import mujoco
import torch
from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from .config import ContactConfig
from .controller import ContactController, bounded, rotation_error
from .probe import ProbeConfig, ConicalReference
from .parking import LateralParking
from .xy_tracking import advance_reference, limit_request, unapplied_advance
from .wrench_torch import compensate_payload_gravity, transform_wrench_sensor_to_target_batch


@dataclass(kw_only=True)
class ContactActionCfg(DifferentialIKActionCfg):
    contact: ContactConfig = field(default_factory=ContactConfig)
    probe: ProbeConfig | None = None
    def build(self, env):
        return ContactAction(self, env)


class ContactAction(DifferentialIKAction):
    @property
    def action_dim(self):
        return 2

    @property
    def raw_actions(self):
        return self.xy_action

    @property
    def processed_actions(self):
        return self.xy_action

    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.controller = ContactController(cfg.contact,self.num_envs,self.device)
        self.xy_action = torch.zeros(self.num_envs,2,device=self.device)
        self.command = self._entity.data.joint_pos[:,self._joint_ids].double().clone()
        self.qvelocity = torch.zeros_like(self.command,dtype=torch.float32)
        self.wrench = torch.zeros(self.num_envs,6,device=self.device)
        self.sensor_wrench = self.wrench.clone()
        self.raw_wrench = self.wrench.clone()
        self.executed_twist = self.wrench.clone()
        self.energy_limited = torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.accel_limited = self.energy_limited.clone()
        self.parking = LateralParking(cfg.contact,self.num_envs,self.device)
        self.records=[]
        self.record=False
        self.diagnostic_hook=None
        self.diagnostic_lateral=None
        # SPIKE-012 P4 parking candidate was rejected; these opt-in fields are
        # retained only to reproduce its A/B trace. Normal control leaves them off.
        self.diagnostic_parking_speed=None
        self.diagnostic_parking_mask=None
        self.diagnostic_parking_active=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.diagnostic_parking_goal=torch.zeros(self.num_envs,3,device=self.device)
        self.probe = None if cfg.probe is None else ConicalReference(cfg.probe,self.num_envs,self.device,cfg.contact.dt)
        self.probe_records=[]
        # Nominal horizontal support frame, +Z downward; no true hole pose.
        self.control_rotation=torch.diag(torch.tensor([1.,-1.,-1.],device=self.device)).repeat(self.num_envs,1,1)
        model=env.sim.mj_model
        def idx(kind,name):
            i=mujoco.mj_name2id(model,kind,name)
            if i<0:raise ValueError(name)
            return i
        self.sensor_id=idx(mujoco.mjtObj.mjOBJ_SITE,'robot/ft_frame_SITE')
        self.geometry_id=idx(mujoco.mjtObj.mjOBJ_SITE,'robot/peg_geometry_tip_site')
        self.hole_id=idx(mujoco.mjtObj.mjOBJ_SITE,'hole/hole_mouth_site')
        self.payload_id=idx(mujoco.mjtObj.mjOBJ_BODY,'robot/hand_mount')
        self.payload_mass=float(model.body_subtreemass[self.payload_id])
        force=idx(mujoco.mjtObj.mjOBJ_SENSOR,'robot/ft_frame_SENSOR_FORCE')
        torque=idx(mujoco.mjtObj.mjOBJ_SENSOR,'robot/ft_frame_SENSOR_TORQUE')
        self.force_adr=int(model.sensor_adr[force]);self.torque_adr=int(model.sensor_adr[torque])
        self.gravity=torch.tensor(model.opt.gravity,device=self.device,dtype=torch.float32)
        actuator_ids=[idx(mujoco.mjtObj.mjOBJ_ACTUATOR,f'robot/joint_{i}') for i in range(1,7)]
        self.act_addresses=torch.tensor(model.actuator_actadr[actuator_ids],device=self.device)
        self.stiffness=torch.tensor(model.actuator_gainprm[actuator_ids,0],device=self.device,dtype=torch.float64)

    def process_actions(self,actions):
        if not torch.isfinite(actions).all():raise ValueError('non-finite action')
        self.xy_action[:]=actions.clamp(-1,1)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        self.controller.reset(ids)
        self.command[ids]=self._entity.data.joint_pos[ids][:,self._joint_ids].double()
        self.qvelocity[ids]=0
        self.xy_action[ids]=0
        self.parking.reset(ids)
        self.diagnostic_parking_active[ids]=False
        self.wrench[ids]=0
        self.sensor_wrench[ids]=0
        if self.probe is not None:self.probe.reset(ids)

    def read_wrench(self):
        d=self._env.sim.data
        raw=torch.cat((d.sensordata[:,self.force_adr:self.force_adr+3],d.sensordata[:,self.torque_adr:self.torque_adr+3]),-1)
        sr=d.site_xmat[:,self.sensor_id].view(-1,3,3)
        tr=d.site_xmat[:,self._frame_id].view(-1,3,3)
        sp=d.site_xpos[:,self.sensor_id];tp=d.site_xpos[:,self._frame_id]
        com=(sr.transpose(1,2)@(d.subtree_com[:,self.payload_id]-sp)[:,:,None]).squeeze(-1)
        sensor=compensate_payload_gravity(raw,sr,com,self.payload_mass,self.gravity)
        offset=(sr.transpose(1,2)@(tp-sp)[:,:,None]).squeeze(-1)
        force_frame=tr if self.probe is None else self.control_rotation
        tip=transform_wrench_sensor_to_target_batch(sensor,offset,force_frame.transpose(1,2)@sr)
        tip[:,2]*=-1
        self.raw_wrench[:]=raw;self.wrench[:]=tip;self.sensor_wrench[:]=sensor
        return tp,tr

    def apply_actions(self):
        c=self.cfg.contact;d=self._env.sim.data
        pos,rot=self.read_wrench()
        diagnostic=self.diagnostic_hook
        if diagnostic is not None:
            diagnostic.before(self,pos,rot)
        fresh=~self.controller.initialized
        # Preserve the XML position drives; counter residual gravity using their
        # known stiffness. The XML already compensates the six robot links.
        # Subtract that compensation to avoid applying gravity support twice.
        gravity_offset=(d.qfrc_bias-d.qfrc_gravcomp)[:,self._joint_dof_ids].double()/self.stiffness
        initial=self._entity.data.joint_pos[:,self._joint_ids].double()+gravity_offset
        d.act[:,self.act_addresses]=torch.where(fresh[:,None],initial.float(),d.act[:,self.act_addresses])
        # Timestamp refers to the sampled pre-integration state, not the next state.
        timestamp=self.controller.elapsed.clone()
        formal_parking=c.parking_enabled and self.diagnostic_parking_speed is None and self.diagnostic_lateral is None
        # Diagnostic replacements retain their original semantics. In the
        # ordinary path, parking selects the phase before tracking advances.
        tracking_limits=(c.xy_tracking_limits_enabled and
                         self.diagnostic_parking_speed is None and self.diagnostic_lateral is None)
        twist=self.controller.step(self.wrench,self.sensor_wrench,pos,rot,self.xy_action,
                                   None if self.probe is None else self.control_rotation,
                                   defer_xy_reference=formal_parking or tracking_limits)
        self._point_torch[:]=pos
        self._compute_jacobian()
        jp=self._jacp_torch[:,:,self._joint_dof_ids]
        jr=self._jacr_torch[:,:,self._joint_dof_ids]
        jac=torch.cat((jp,jr),dim=1)
        # The joint target is an integrated position command. Feeding actual pose
        # error alone into that integrator accumulates lateral/rotational pressure
        # while contact prevents motion. Close these loops on predicted commanded
        # pose, retaining force integration only on the pressing axis.
        actual=self._entity.data.joint_pos[:,self._joint_ids]
        lead=(jac@(self.command-actual.double()).float()[:,:,None]).squeeze(-1)
        axis=rot[:,:,2] if self.probe is None else self.control_rotation[:,:,2]
        healthy=(self.controller.state!=ContactController.FAULT)[:,None]
        track_mask=healthy[:,0].clone()
        if formal_parking:
            ctrl=self.controller
            permitted=ctrl.ready if ctrl.diagnostic_xy_permission is None else ctrl.diagnostic_xy_permission
            upstream=bounded(self.xy_action,1.)*c.xy_speed
            measured_linear=(jp@self._entity.data.joint_vel[:,self._joint_ids][:,:,None]).squeeze(-1)
            track_mask=self.parking.begin(pos,measured_linear,lead[:,:3],axis,upstream,
                                          permitted,~healthy[:,0],ctrl.xy_reference)
        reference_increment=torch.zeros_like(pos)
        if tracking_limits:
            request=torch.where(track_mask[:,None],self.controller.last_xy,0)
            reference,reference_increment=advance_reference(
                self.controller.xy_reference,pos,axis,request,c.dt,c.xy_reference_lead)
            self.controller.xy_reference[:]=reference
        elif formal_parking:
            self.controller.xy_reference+=torch.where(track_mask[:,None],self.controller.last_xy*c.dt,0)
        delta=c.pose_error_scale*(self.controller.xy_reference-pos)-lead[:,:3]
        lateral=delta-(delta*axis).sum(-1)[:,None]*axis
        lateral=bounded(c.lateral_hold_gain*lateral,c.lateral_hold_speed)
        angular=bounded(c.orientation_gain*(rotation_error(self.controller.anchor_rot,rot)-lead[:,3:]),c.angular_speed)
        if self.probe is not None:
            ctrl=self.controller
            present=(ctrl.state==ctrl.REGULATE)&(ctrl.filtered_force>=c.support_force)
            usable=present&(ctrl.filtered_force<c.operating_upper)&(self.wrench[:,2]<c.operating_upper)
            target,feedforward,probe_active=self.probe.step(ctrl.anchor_rot,ctrl.ready,usable,present)
            measured_omega=(jr@self._entity.data.joint_vel[:,self._joint_ids][:,:,None]).squeeze(-1)
            request=feedforward+c.orientation_gain*(rotation_error(target,rot)-lead[:,3:])-c.orientation_damping*(measured_omega-feedforward)
            angular=bounded(request,c.angular_speed)
            too_far=rotation_error(ctrl.anchor_rot,rot).norm(dim=-1)>self.cfg.probe.absolute_limit_deg*torch.pi/180
            newly=too_far&(ctrl.state!=ctrl.FAULT)
            ctrl.reason[:]=torch.where(newly,9,ctrl.reason)
            ctrl.state[:]=torch.where(newly,ctrl.FAULT,ctrl.state)
            ctrl.reanchor_event |= newly
            ctrl.velocity[:]=torch.where(newly,0,ctrl.velocity)
            if self.record:
                local=lambda r:(ctrl.anchor_rot.transpose(1,2)@rotation_error(r,ctrl.anchor_rot)[:,:,None]).squeeze(-1)
                self.probe_records.append(torch.cat((self.probe.phase[:,None],self.probe.amplitude[:,None],self.probe.started[:,None],probe_active[:,None],local(target),local(rot),rotation_error(target,rot).norm(dim=-1)[:,None],(request.norm(dim=-1)>c.angular_speed)[:,None],rotation_error(ctrl.anchor_rot,self.control_rotation)),dim=-1).detach().clone())
        normal=self.controller.velocity[:,None]*axis
        if formal_parking:
            feedforward=reference_increment/c.dt if tracking_limits else ctrl.last_xy
            normal_lateral=lateral+torch.where(track_mask[:,None],feedforward,0)
            if tracking_limits:
                normal_lateral=limit_request(normal_lateral,axis,c.xy_speed)
            transverse=self.parking.output(pos,lead[:,:3],axis,normal_lateral,
                                           self.wrench[:,2],~healthy[:,0])
        elif self.diagnostic_lateral is not None:
            # E1 replaces only the final transverse request, retaining all
            # downstream IK, limits, drives, axial and angular control.
            transverse=self.diagnostic_lateral
            transverse=transverse-(transverse*axis).sum(-1)[:,None]*axis
        else:
            # When the measured-position leash stops reference advancement,
            # stop its feedforward too; otherwise the original PPO velocity
            # would keep integrating joint targets despite the bounded reference.
            feedforward=reference_increment/c.dt if tracking_limits else self.controller.last_xy
            transverse=lateral+feedforward
        tracking_desired=transverse.clone()
        if tracking_limits and not formal_parking:
            # The bound applies to the combined request, not to the PPO term
            # alone. It is not a bound on measured robot / geometry-tip speed.
            transverse=limit_request(transverse,axis,c.xy_speed)
        reference_used=self.controller.xy_reference.clone()
        if self.diagnostic_parking_speed is not None and self.diagnostic_parking_mask is not None:
            # Experimental finite parking target: hold a world-space XY goal
            # after stop while normal/rotation control and all limits remain.
            # Initial goal includes only 25 ms of measured transverse motion.
            parking=self.diagnostic_parking_mask
            new=parking & ~self.diagnostic_parking_active
            measured_linear=(jp@self._entity.data.joint_vel[:,self._joint_ids][:,:,None]).squeeze(-1)
            initial_goal=pos+measured_linear*.025
            self.diagnostic_parking_goal[:]=torch.where(new[:,None],initial_goal,self.diagnostic_parking_goal)
            self.diagnostic_parking_active[:]=parking
            parking_error=self.diagnostic_parking_goal-pos-lead[:,:3]
            parking_error=parking_error-(parking_error*axis).sum(-1)[:,None]*axis
            parking_twist=bounded(20.*parking_error,self.diagnostic_parking_speed)
            transverse=torch.where(parking[:,None],parking_twist,transverse)
        twist=torch.cat((torch.where(healthy,normal+transverse,normal),
                         torch.where(healthy,angular,0)),dim=-1)
        twist=torch.nan_to_num(twist)
        matrix=jac.transpose(1,2)@jac
        matrix.diagonal(dim1=-2,dim2=-1).add_(self.cfg.damping**2)
        # Direct increments preserve sub-micrometre requests in float32 coordinates.
        increment=torch.linalg.solve(matrix,(jac.transpose(1,2)@(twist*c.dt)[:,:,None]).squeeze(-1))
        requested=(increment/c.dt).clamp(-c.joint_speed,c.joint_speed)
        reanchor=self.controller.reanchor_event
        actual=self._entity.data.joint_pos[:,self._joint_ids]
        measured_v=self._entity.data.joint_vel[:,self._joint_ids]
        memory=torch.where(reanchor[:,None],measured_v,self.qvelocity)
        velocity=requested.clamp(memory-c.joint_acceleration*c.dt,memory+c.joint_acceleration*c.dt)
        self.accel_limited[:]=(velocity-requested).abs().amax(-1)>1e-8
        self.qvelocity[:]=velocity
        base=torch.where(reanchor[:,None],actual.double(),self.command)
        command_before=self.command.clone()
        candidate=base+velocity.double()*c.dt
        error=candidate-actual.double()
        energy=(.5*self.stiffness*error.square()).sum(-1)
        scale=(c.command_energy/energy.clamp_min(1e-16)).sqrt().clamp(max=1)
        scale=torch.where(self.wrench[:,:3].norm(dim=-1)>=10,scale,1)
        self.energy_limited[:]=scale<1
        limited=actual.double()+error*scale[:,None]
        self.command[:]=torch.maximum(torch.minimum(limited,self._joint_upper.double()),self._joint_lower.double())
        reference_rollback=torch.zeros_like(pos)
        applied_xy=torch.zeros_like(pos)
        if formal_parking or tracking_limits:
            achieved=(jp@((self.command-command_before)/c.dt).float()[:,:,None]).squeeze(-1)
            achieved=achieved-(achieved*axis).sum(-1)[:,None]*axis
            applied_xy=achieved
        if tracking_limits:
            # Reanchoring is a separate state reset, not tracking capacity.
            eligible=track_mask[:,None] & ~reanchor[:,None]
            reference_rollback=unapplied_advance(
                reference_increment,tracking_desired,achieved,axis,c.dt)
            reference_rollback=torch.where(eligible,reference_rollback,0)
            self.controller.xy_reference-=reference_rollback
        if formal_parking:
            saturated=self.accel_limited|self.energy_limited|((self.command-limited).abs().amax(-1)>1e-10)|\
                      (((increment/c.dt)-requested).abs().amax(-1)>1e-8)
            self.parking.commit(achieved,saturated)
        self._entity.set_joint_position_target((self.command+gravity_offset).float(),joint_ids=self._joint_ids)
        self.executed_twist[:]=(jac@measured_v[:,:,None]).squeeze(-1)
        if diagnostic is not None:
            diagnostic.after(self, dict(timestamp=timestamp, pos=pos, rot=rot, axis=axis,
                reference=self.controller.xy_reference, lead=lead, lateral=lateral,
                reference_used=reference_used, reference_increment=reference_increment,
                reference_rollback=reference_rollback, tracking_desired=tracking_desired,
                applied_xy_command=applied_xy,
                normal=normal, angular=angular, final_twist=twist, increment=increment,
                requested_qvelocity=requested, memory=memory, velocity=velocity,
                candidate=candidate, energy=energy, scale=scale, gravity_offset=gravity_offset,
                command=self.command, measured_q=actual, measured_qvelocity=measured_v,
                parking_state=self.parking.state, parking_event=self.parking.event,
                parking_goal=self.parking.goal, parking_actual_velocity=self.parking.actual_velocity,
                parking_raw_correction=self.parking.raw_correction,
                parking_limited_correction=self.parking.limited_correction,
                parking_load_fraction=self.parking.load_fraction,
                parking_speed_limited=self.parking.speed_limited,
                parking_acceleration_limited=self.parking.acceleration_limited,
                parking_lead_limited=self.parking.lead_limited,
                parking_downstream_limited=self.parking.downstream_limited,
                tracking_limits_active=track_mask & tracking_limits))
        if self.record:
            ctrl=self.controller
            geom=d.site_xpos[:,self.geometry_id]
            hole=d.site_xpos[:,self.hole_id]
            delta=(d.site_xmat[:,self.hole_id].view(-1,3,3).transpose(1,2)@(geom-hole)[:,:,None]).squeeze(-1)
            angle=torch.acos((-(d.site_xmat[:,self.geometry_id].view(-1,3,3)[:,:,2]*d.site_xmat[:,self.hole_id].view(-1,3,3)[:,:,2]).sum(-1)).clamp(-1,1))
            self.records.append(torch.cat((timestamp[:,None],self.wrench,ctrl.filtered_force[:,None],ctrl.target[:,None],
                ctrl.state[:,None],ctrl.reason[:,None],ctrl.ready[:,None],ctrl.travel[:,None],ctrl.angular_drift[:,None],
                twist,self.executed_twist,pos,delta,angle[:,None],ctrl.recoveries[:,None],
                self.accel_limited[:,None],self.energy_limited[:,None],ctrl.last_xy,self.sensor_wrench,self.raw_wrench,
                ctrl.integral[:,None],lead,self.parking.state[:,None],self.parking.event[:,None],
                self.parking.goal,self.parking.actual_velocity,self.parking.raw_correction,
                self.parking.limited_correction,self.parking.load_fraction[:,None],
                self.parking.speed_limited[:,None],self.parking.acceleration_limited[:,None],
                self.parking.lead_limited[:,None],self.parking.downstream_limited[:,None],
                reference_used,reference_increment,reference_rollback,tracking_desired,applied_xy,
                (track_mask & tracking_limits)[:,None]),-1).detach().clone())


COLUMNS=(['time']+['Fx','Fy','Fz','Mx','My','Mz']+['filtered_force','target','state','reason','ready','travel','attitude_drift']+
         ['cmd_vx','cmd_vy','cmd_vz','cmd_wx','cmd_wy','cmd_wz']+
         ['actual_vx','actual_vy','actual_vz','actual_wx','actual_wy','actual_wz']+
         ['tip_x','tip_y','tip_z','hole_dx','hole_dy','hole_dz','relative_angle','recoveries','accel_limited','energy_limited']+
         ['requested_xy_x','requested_xy_y','requested_xy_z']+
         ['sensor_Fx','sensor_Fy','sensor_Fz','sensor_Mx','sensor_My','sensor_Mz']+
         ['raw_Fx','raw_Fy','raw_Fz','raw_Mx','raw_My','raw_Mz']+['integral']+['lead_x','lead_y','lead_z','lead_rx','lead_ry','lead_rz']+
         ['parking_state','parking_event','parking_goal_x','parking_goal_y','parking_goal_z',
          'parking_actual_vx','parking_actual_vy','parking_actual_vz',
          'parking_raw_vx','parking_raw_vy','parking_raw_vz',
          'parking_limited_vx','parking_limited_vy','parking_limited_vz','parking_load_fraction',
          'parking_speed_limited','parking_acceleration_limited','parking_lead_limited','parking_downstream_limited']+
         ['tracking_reference_used_x','tracking_reference_used_y','tracking_reference_used_z',
          'tracking_reference_increment_x','tracking_reference_increment_y','tracking_reference_increment_z',
          'tracking_reference_rollback_x','tracking_reference_rollback_y','tracking_reference_rollback_z',
          'tracking_desired_vx','tracking_desired_vy','tracking_desired_vz',
          'tracking_applied_vx','tracking_applied_vy','tracking_applied_vz','tracking_limits_active'])
