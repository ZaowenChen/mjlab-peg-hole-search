"""Frozen-state causal diagnostic, temporary in-memory collision masks only."""
import json
from pathlib import Path
import numpy as np
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab_contact_prep.environment import make_env
from mjlab_contact_prep.config import ContactConfig
from mjlab_contact_prep.probe import ProbeConfig

root=Path(__file__).resolve().parents[1];audit=root/'evaluation/probe_cpu_gpu_hole';out=root/'evaluation/probe_contact_localization'
info=json.loads((audit/'results.json').read_text())
with np.load(audit/'states.npz') as z:states={k:z[k] for k in z.files}
env=ManagerBasedRlEnv(cfg=make_env(info['cases'],ContactConfig(),probe=ProbeConfig()),device='cuda:0')
try:
    env.reset();m=env.sim.mj_model;term=env.action_manager.get_term('contact');d=mujoco.MjData(m)
    masks=[m.geom_contype.copy(),m.geom_conaffinity.copy()];rows=[]
    for i,numbers in [(0,list(range(48,58))),(2,list(range(32,42)))]:
        ids=[g for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and masks[0][g] and
             m.mesh(int(m.geom_dataid[g])).name in [f'hole/Assem1 - Part1-{n}' for n in numbers]]
        assert len(ids)==10
        for keep in [1,2,5,10]:
            m.geom_contype[:]=masks[0];m.geom_conaffinity[:]=masks[1]
            m.geom_contype[ids[keep:]]=0;m.geom_conaffinity[ids[keep:]]=0
            for name,v in states.items():getattr(d,name)[:]=v[0,i]
            mujoco.mj_forward(m,d)
            sr=d.site_xmat[term.sensor_id].reshape(3,3)
            force_sensor=-d.sensordata[term.force_adr:term.force_adr+3]-sr.T@(m.opt.gravity*term.payload_mass)
            fz=float((sr@force_sensor)[2])
            rows.append(dict(case=info['cases'][i]['id'],snapshot=0,overlapping_group=numbers,retained=keep,cpu_contacts=d.ncon,compensated_Fz_N=fz,
                contact_dims=sorted(set(int(c.dim) for c in d.contact[:d.ncon])),
                contact_friction=[list(x) for x in sorted(set(tuple(c.friction) for c in d.contact[:d.ncon]))]))
    m.geom_contype[:]=masks[0];m.geom_conaffinity[:]=masks[1]
    (out/'overlap_ablation.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
finally:env.close()
