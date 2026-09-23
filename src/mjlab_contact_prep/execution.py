"""MJLab adapter: reused sensor conventions and incremental DLS execution."""
from dataclasses import dataclass, field
import mujoco
import torch
from mjlab.envs.mdp.actions import DifferentialIKAction, DifferentialIKActionCfg
from .config import ContactConfig
from .controller import ContactController, bounded, rotation_error
from .probe import ProbeConfig, ConicalReference
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
        self.records=[]
        self.record=False
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
        fresh=~self.controller.initialized
        # Preserve the XML position drives; counter residual gravity using their
        # known stiffness. The XML already compensates the six robot links.
        # Subtract that compensation to avoid applying gravity support twice.
        gravity_offset=(d.qfrc_bias-d.qfrc_gravcomp)[:,self._joint_dof_ids].double()/self.stiffness
        initial=self._entity.data.joint_pos[:,self._joint_ids].double()+gravity_offset
        d.act[:,self.act_addresses]=torch.where(fresh[:,None],initial.float(),d.act[:,self.act_addresses])
        # Timestamp refers to the sampled pre-integration state, not the next state.
        timestamp=self.controller.elapsed.clone()
        twist=self.controller.step(self.wrench,self.sensor_wrench,pos,rot,self.xy_action,
                                   None if self.probe is None else self.control_rotation)
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
        healthy=(self.controller.state!=ContactController.FAULT)[:,None]
        twist=torch.cat((torch.where(healthy,normal+lateral+self.controller.last_xy,normal),
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
        candidate=base+velocity.double()*c.dt
        error=candidate-actual.double()
        energy=(.5*self.stiffness*error.square()).sum(-1)
        scale=(c.command_energy/energy.clamp_min(1e-16)).sqrt().clamp(max=1)
        scale=torch.where(self.wrench[:,:3].norm(dim=-1)>=10,scale,1)
        self.energy_limited[:]=scale<1
        limited=actual.double()+error*scale[:,None]
        self.command[:]=torch.maximum(torch.minimum(limited,self._joint_upper.double()),self._joint_lower.double())
        self._entity.set_joint_position_target((self.command+gravity_offset).float(),joint_ids=self._joint_ids)
        self.executed_twist[:]=(jac@measured_v[:,:,None]).squeeze(-1)
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
                ctrl.integral[:,None],lead),-1).detach().clone())


COLUMNS=(['time']+['Fx','Fy','Fz','Mx','My','Mz']+['filtered_force','target','state','reason','ready','travel','attitude_drift']+
         ['cmd_vx','cmd_vy','cmd_vz','cmd_wx','cmd_wy','cmd_wz']+
         ['actual_vx','actual_vy','actual_vz','actual_wx','actual_wy','actual_wz']+
         ['tip_x','tip_y','tip_z','hole_dx','hole_dy','hole_dz','relative_angle','recoveries','accel_limited','energy_limited']+
         ['requested_xy_x','requested_xy_y','requested_xy_z']+
         ['sensor_Fx','sensor_Fy','sensor_Fz','sensor_Mx','sensor_My','sensor_Mz']+
         ['raw_Fx','raw_Fy','raw_Fz','raw_Mx','raw_My','raw_Mz']+['integral']+['lead_x','lead_y','lead_z','lead_rx','lead_ry','lead_rz'])
