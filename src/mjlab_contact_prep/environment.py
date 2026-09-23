"""Physical contact test environment, with no reward or search state machine."""
import math
from dataclasses import replace
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.lab_api.math import quat_from_angle_axis
from .assets import robot_cfg, detailed_hole_cfg, START_ARM_QPOS, START_PEG_GEOMETRY_TIP_WORLD, PEG_COLLISION_RADIUS, START_PEG_FACE_SLOPE_XY
from .randomization import safe_hole_mouth_height
from .execution import ContactActionCfg
from .backend import configure_backend


def plane_spec():
    spec=mujoco.MjSpec()
    body=spec.worldbody.add_body(name='support')
    body.add_geom(name='support_surface',type=mujoco.mjtGeom.mjGEOM_BOX,size=(.08,.08,.01),pos=(0,0,-.01),
                  condim=4,contype=1,conaffinity=15,friction=(.9,.2,.2),solref=(.001,2.))
    body.add_site(name='hole_mouth_site',size=(.001,.001,.001))
    return spec


def reset_cases(env,env_ids,cases,gap):
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    robot=env.scene['robot'];hole=env.scene['hole']
    ids,_=robot.find_joints('joint_[1-6]')
    q=robot.data.joint_pos[env_ids][:,ids]
    addresses=[int(env.sim.mj_model.actuator_actadr[env.sim.mj_model.actuator(f'robot/joint_{i}').id]) for i in range(1,7)]
    env.sim.data.act[env_ids[:,None],torch.tensor(addresses,device=env.device)]=q
    robot.set_joint_position_target(q,joint_ids=ids,env_ids=env_ids[:,None])
    params=torch.tensor([[x['dx_mm']*.001,x['dy_mm']*.001,math.radians(x.get('rx_deg',0)),math.radians(x.get('ry_deg',0))] for x in cases],device=env.device)
    params=params[env_ids]
    tip=torch.tensor(START_PEG_GEOMETRY_TIP_WORLD,device=env.device).expand(len(env_ids),-1)
    pose=hole.data.default_root_state[env_ids,:7].clone()
    pose[:,:2]=tip[:,:2]+params[:,:2]
    pose[:,2]=safe_hole_mouth_height(tip[:,2],params[:,:2],params[:,2:],gap,PEG_COLLISION_RADIUS,START_PEG_FACE_SLOPE_XY)
    pose[:,:3]+=env.scene.env_origins[env_ids]
    vec=torch.cat((params[:,2:],torch.zeros_like(params[:,:1])),-1)
    angle=vec.norm(dim=-1)
    quat=quat_from_angle_axis(angle,vec/angle[:,None].clamp_min(1e-9))
    quat[:,0]=torch.where(angle==0,1,quat[:,0])
    pose[:,3:]=quat
    hole.write_mocap_pose_to_sim(pose,env_ids=env_ids)


def observe(env):
    term=env.action_manager.get_term('contact')
    return torch.cat((term.wrench,term.controller.ready[:,None]),-1)


def make_env(cases,contact,*,plane=False,seconds=15.,probe=None,collision_model='partitioned',physics_backend='contact-fix'):
    configure_backend(physics_backend)
    hole=EntityCfg(spec_fn=plane_spec) if plane else detailed_hole_cfg(collision_model)
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(entities={'robot':robot_cfg(START_ARM_QPOS),'hole':hole},
            terrain=TerrainEntityCfg(terrain_type='plane',collisions=(CollisionCfg(geom_names_expr=('.*',),condim=4,contype=1,conaffinity=15,friction=(.9,.2,.2),solref=(.001,2.)),)),
            num_envs=len(cases),env_spacing=0.),
        actions={'contact':ContactActionCfg(entity_name='robot',actuator_names=('joint_[1-6]',),frame_name='peg_tip_site' if probe is None else 'peg_geometry_tip_site',frame_type='site',damping=.001,max_dq=.02,contact=contact,probe=probe)},
        observations={'actor':ObservationGroupCfg({'contact_state':ObservationTermCfg(func=observe)})},
        events={'reset_scene':EventTermCfg(func=base_mdp.reset_scene_to_default,mode='reset'),
                'cases':EventTermCfg(func=reset_cases,mode='reset',params={'cases':cases,'gap':contact.initial_gap})},
        sim=SimulationCfg(nconmax=512,njmax=2048,mujoco=MujocoCfg(timestep=contact.dt,integrator='implicitfast',solver='newton',iterations=100,cone='pyramidal',ccd_iterations=50)),
        rewards={},terminations={},decimation=20,episode_length_s=seconds+1,auto_reset=False,seed=7)
