"""Matched four-frame Root/body decomposition, not a causal claim from jerk norms."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices

def decomposition(points):
    p=np.asarray(points,dtype=np.float64)
    if p.shape!=(256,22,3) or not np.isfinite(p).all():raise ValueError("Finite256x22x3 FK required")
    root=p[:,:1];body=p-root
    total=np.diff(p,n=3,axis=0)*30**3
    r=np.diff(root,n=3,axis=0)*30**3;b=np.diff(body,n=3,axis=0)*30**3
    delta=float(abs(total-r-b).max())
    if delta>1e-7:raise ValueError("Root/body vector decomposition mismatch")
    result={}
    indices=np.arange(253)
    for name,mask in (("global",np.ones(253,bool)),("core_interior",(indices>=64)&(indices+3<160)),("outside_core",((indices+3<64)|(indices>=160)))):
        result[name]={key:float(np.percentile(np.linalg.norm(x[mask],axis=-1),95)) for key,x in (("world",total),("Root_translation",r),("Root_relative_body",b))}
        result[name]["templates"]=int(mask.sum())
    return dict(regions=result,vector_closure_max_mps3=delta,norms_not_additive=True,Root_relative_includes_ancestor_rotations=True)

def rotation_groups(raw):
    r=matrices(np.asarray(raw)[:,7:].reshape(256,22,6),row=True)
    omega=Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).magnitude().reshape(255,22)*30
    groups={"Root_orientation":[0],"spine":[3,6,9],"shoulders":[13,14,16,17],"elbows":[18,19],"wrists":[20,21],"legs_feet":[1,2,4,5,7,8,10,11]}
    return {k:dict(core_rotation_geodesic_speed_p95_radps=float(np.percentile(omega[64:159,j],95)),
        core_speed_change_p95_radps2=float(np.percentile(abs(np.diff(omega[64:159,j],axis=0))*30,95)),
        interpretation="Rotation speed/its scalar variation; not terminal FK jerk attribution") for k,j in groups.items()}
