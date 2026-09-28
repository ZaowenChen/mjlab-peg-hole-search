"""Quantify overlap of actual OBJ top triangles at localized contact points."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1];out=root/'evaluation/probe_contact_localization'
rows=json.loads((out/'contacts.json').read_text())
meshes={}
for path in (root/'assets/hole').glob('*.obj'):
    lines=path.read_text().splitlines()
    v=np.array([[float(x) for x in l.split()[1:4]] for l in lines if l.startswith('v ')])*1000
    faces=[[int(x.split('/')[0])-1 for x in l.split()[1:]] for l in lines if l.startswith('f ')]
    top=[v[f,:2] for f in faces if np.max(np.abs(v[f,2]))<1e-6]
    if top:meshes[path.stem]=dict(triangles=top,vertices=v)

def inside(point,triangle):
    a,b,c=triangle
    coeff=np.linalg.solve(np.stack((b-a,c-a),1),point-a)
    return min(coeff)>=-1e-6 and coeff.sum()<=1+1e-6

results=[]
for row in rows:
    covered=[]
    for contact in row['cpu']:
        point=np.array(contact['hole_position_mm'][:2])
        owners=[name for name,mesh in meshes.items() if any(inside(point,t) for t in mesh['triangles'])]
        covered.append(dict(point_mm=point.tolist(),contact_mesh=contact['meshes'][1],top_surface_count=len(owners),top_surface_meshes=owners))
    results.append(dict(snapshot=row['snapshot'],case=row['case'],points=covered))
(out/'top_overlap.json').write_text(json.dumps(results,indent=2)+'\n')

fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,case in zip(axes,['5mm_xp','5mm_ym','20mm_xm']):
    row=next(r for r in rows if r['case']==case and r['snapshot']==0)
    active={c['meshes'][1].split('/')[-1] for c in row['cpu']}
    for name in active:
        if name not in meshes:continue
        points=np.unique(np.concatenate(meshes[name]['triangles']),axis=0)
        poly=points[ConvexHull(points).vertices]
        ax.fill(poly[:,0],poly[:,1],alpha=.07,color='C0')
        poly=np.vstack((poly,poly[0]));ax.plot(poly[:,0],poly[:,1],lw=.5,alpha=.35,color='C0')
    for side,marker,color,size in [('cpu','o','C1',36),('gpu','x','black',60)]:
        pos=np.array([c['hole_position_mm'] for c in row[side]])
        ax.scatter(pos[:,0],pos[:,1],s=size,marker=marker,c=color,label=side.upper(),zorder=4)
    angle=np.linspace(0,2*np.pi,200)
    ax.plot(17.5*np.cos(angle),17.5*np.sin(angle),'--',color='gray',label='nominal bore r=17.5 mm')
    ax.set_title(case+' / snapshot 0');ax.set_aspect('equal');ax.set_xlabel('hole X (mm)');ax.set_ylabel('hole Y (mm)');ax.grid(alpha=.2)
    ax.legend(fontsize=7)
fig.suptitle('Overlapping original OBJ top faces and localized contacts');fig.tight_layout()
fig.savefig(out/'top_overlap.png',dpi=170,bbox_inches='tight')
for case in ['5mm_xp','5mm_ym','20mm_xm']:
    r=next(x for x in results if x['case']==case and x['snapshot']==0)
    point=max(r['points'],key=lambda x:x['top_surface_count'])
    print(case,json.dumps(point))
