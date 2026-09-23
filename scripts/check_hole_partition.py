"""Independent union/overlap checks against original OBJ convex solids."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import ConvexHull
from mjlab_contact_prep.hole_collision import partition

root=Path(__file__).resolve().parents[1]
old=[]
for path in (root/'assets/hole').glob('*.obj'):
    v=np.array([[float(x) for x in line.split()[1:4]] for line in path.read_text().splitlines() if line.startswith('v ')])
    old.append(ConvexHull(v).equations)
sectors,boxes=partition();new=[ConvexHull(v).equations for v,f in sectors]
for pos,size in boxes:
    signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    new.append(ConvexHull(np.array(pos)+signs*np.array(size)).equations)

def count(points,solids,tol=1e-10):
    result=np.zeros(len(points),dtype=int)
    for eq in solids:
        inside=np.ones(len(points),dtype=bool)
        for face in eq:
            inside &= points[:,0]*face[0]+points[:,1]*face[1]+points[:,2]*face[2]+face[3]<=tol
        result+=inside
    return result

rng=np.random.default_rng(729)
points=rng.uniform([-.046,-.046,-.041],[.046,.046,.001],size=(50000,3))
# Dense radial samples check micron-scale bore/chamfer fidelity, including seams.
angles=np.linspace(0,2*np.pi,1440,endpoint=False)+.00001
radial=[]
for z,r in [( -.0005,.018),(-.02,.0175),(-.03975,.01775)]:
    for delta in [-.00002,-.000002,.000002,.00002]:
        radius=(r+delta)/np.maximum.reduce([np.cos(angles-np.deg2rad(5*i)) for i in range(72)])
        radial.extend(np.c_[radius*np.cos(angles),radius*np.sin(angles),np.full(len(angles),z)])
points=np.vstack((points,radial))
a=count(points,old);b=count(points,new)
mismatch=np.flatnonzero((a>0)!=(b>0));overlap=int((b>1).sum())
result=dict(points=len(points),union_mismatches=len(mismatch),new_overlap_points=overlap,old_max_coverage=int(a.max()),new_max_coverage=int(b.max()),
            mismatches=points[mismatch].tolist(),bore_diameter_m=.035,entrance_chamfer_m=.001,exit_chamfer_m=.0005,depth_m=.04,
            sectors=len(sectors),outer_boxes=len(boxes),note='Finite sample geometry check, not a mathematical proof; OBJ coordinates are rounded to micrometres.')
out=root/'evaluation/partition_geometry.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
assert not len(mismatch), 'union geometry differs'
assert not overlap, 'new convex solids overlap in sampled interiors'
