"""Finite discrete C2 boundary projection in Root / SO(3) charts, not 6D blending."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices,sixd,motion151

def root_extension(left,right):
    lv=left[-1]-left[-2];la=left[-1]-2*left[-2]+left[-3]
    rv=right[1]-right[0];ra=right[2]-2*right[1]+right[0]
    k=np.arange(1,4)[:,None]
    return left[-1]+k*lv+k*(k+1)/2*la, (right[0]-k*rv+k*(k+1)/2*ra)[::-1]

def rotation_extension(left,right):
    log=lambda x:Rotation.from_matrix(x.reshape(-1,3,3)).as_rotvec().reshape(x.shape[:-2]+(3,))
    exp=lambda x:Rotation.from_rotvec(x.reshape(-1,3)).as_matrix().reshape(x.shape[:-1]+(3,3))
    lv=log(left[-2].swapaxes(-1,-2)@left[-1]);la=lv-log(left[-3].swapaxes(-1,-2)@left[-2])
    rv=log(right[0].swapaxes(-1,-2)@right[1]);ra=log(right[1].swapaxes(-1,-2)@right[2])-rv
    L=[];R=[];l=left[-1];r=right[0]
    for k in range(1,4):
        l=l@exp(lv+k*la);r=r@exp(-(rv-k*ra));L.append(l.copy());R.append(r.copy())
    return np.asarray(L),np.asarray(R)[::-1]

def quadratic(candidate,fixed,values,weight):
    """Unique SPD equality-constrained minimum: fidelity + weight*||D3 x||^2."""
    n=len(candidate);d=np.diff(np.eye(n),n=3,axis=0);h=np.eye(n)+weight*d.T@d
    out=candidate.copy();out[fixed]=values[fixed];free=~fixed
    if free.any():
        out[free]=np.linalg.solve(h[np.ix_(free,free)],candidate[free]-h[np.ix_(free,fixed)]@out[fixed])
    return out

def contract(x,old,a,b):
    p=x[:,4:7].astype(float);r=matrices(x[:,7:].reshape(-1,24,6)).astype(float)
    targetp=root_extension(old[a-3:a,4:7].astype(float),old[b:b+3,4:7].astype(float))
    ro=matrices(old[:,7:].reshape(-1,24,6)).astype(float);targetr=rotation_extension(ro[a-3:a],ro[b:b+3])
    rows=[]
    for ids,pr,rr in ((np.arange(a,a+3),targetp[0],targetr[0]),(np.arange(b-3,b),targetp[1],targetr[1])):
        e=p[ids]-pr;angle=Rotation.from_matrix((rr.swapaxes(-1,-2)@r[ids]).reshape(-1,3,3)).magnitude()
        rows.append(dict(root_position_residual_m=float(abs(e).max()),root_d1_residual_mps=float(abs(np.diff(e,axis=0)*30).max()),
            root_d2_residual_mps2=float(abs(np.diff(e,n=2,axis=0)*900).max()),rotation_residual_rad=float(angle.max())))
    w=Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,24,3)
    for row,t in zip(rows,(a,b)):
        row["root_cross_seam_d3_max_mps3"]=float(np.linalg.norm(np.diff(p,n=3,axis=0)[t-3:t],axis=-1).max()*27000)
        row["angular_cross_seam_d2_max_radps3"]=float(np.linalg.norm(np.diff(w,n=2,axis=0)[t-3:t],axis=-1).max()*27000)
    return rows

def project(old,candidate,cfg,key_joints=None):
    old=motion151(old);candidate=motion151(candidate);a,b=cfg["edit"]
    if old.shape!=candidate.shape or a<3 or b+3>len(old) or b-a<9:raise ValueError("Real two-sided context and at least nine editable frames required")
    if not np.array_equal(old[:a],candidate[:a]) or not np.array_equal(old[b:],candidate[b:]):raise ValueError("Candidate changes protected context")
    p=candidate[a-3:b+3,4:7].astype(float);r=matrices(candidate[a-3:b+3,7:].reshape(-1,24,6)).astype(float)
    lo,hi=root_extension(old[a-3:a,4:7].astype(float),old[b:b+3,4:7].astype(float))
    L,R=rotation_extension(r[:3],r[-3:]);n=b-a;indices=np.r_[np.arange(3,6),np.arange(n,n+3)]
    desiredp=np.r_[lo,hi];desiredr=np.concatenate([L,R]);priorr=r[indices]
    required_root=float(np.linalg.norm(p[indices]-desiredp,axis=-1).max())
    required_rot=float(Rotation.from_matrix((priorr.swapaxes(-1,-2)@desiredr).reshape(-1,3,3)).magnitude().max())
    evidence=dict(required_boundary_root_m=required_root,required_boundary_rotation_rad=required_rot,root_cap_m=cfg["root_cap_m"],rotation_cap_rad=cfg["rotation_cap_rad"],
        discrete_contract="three extrapolated frames at each seam; constant Root acceleration and body angular acceleration",
        continuous_time_c2_proven=False,post_generation_boundary_projection=True,context_clipped=False)
    if required_root>cfg["root_cap_m"] or required_rot>cfg["rotation_cap_rad"]:
        return None,dict(evidence,status="NECESSARY_BOUNDARY_DISPLACEMENT_EXCEEDS_CAP",capacity_proven=False,necessary_infeasibility_within_registered_caps=True)
    base=r[2];chart=Rotation.from_matrix((base.swapaxes(-1,-2)[None]@r).reshape(-1,3,3)).as_rotvec().reshape(len(r),24,3)
    target=Rotation.from_matrix((base.swapaxes(-1,-2)[None]@desiredr).reshape(-1,3,3)).as_rotvec().reshape(6,24,3)
    if max(np.linalg.norm(chart,axis=-1).max(),np.linalg.norm(target,axis=-1).max())>=np.pi-cfg["chart_margin_rad"]:
        return None,dict(evidence,status="SO3_CHART_CUT_REJECTED",capacity_proven=False,necessary_infeasibility_within_registered_caps=False)
    fixed=np.zeros(len(p),bool);fixed[:6]=True;fixed[-6:]=True
    values=p.copy();values[indices]=desiredp;yp=quadratic(p,fixed,values,cfg["root_d3_weight"])
    yr=chart.copy();keys=key_joints or {}
    for j in range(24):
        mask=fixed.copy();v=chart[:,j].copy();v[indices]=target[:,j]
        for frame,joints in keys.items():
            if j in joints and a+3<=frame<b-3:mask[frame-a+3]=True
        yr[:,j]=quadratic(chart[:,j],mask,v,cfg["rotation_d3_weight"])
    matrices_out=base[None]@Rotation.from_rotvec(yr.reshape(-1,3)).as_matrix().reshape(len(yr),24,3,3)
    x=candidate.copy();x[a:b,4:7]=yp[3:-3].astype(np.float32);x[a:b,7:]=sixd(matrices_out[3:-3]).reshape(n,144).astype(np.float32)
    for frame,joints in keys.items():
        if a+3<=frame<b-3:
            for j in joints:x[frame,7+6*j:13+6*j]=candidate[frame,7+6*j:13+6*j]
    delta_root=float(np.linalg.norm(x[a:b,4:7]-candidate[a:b,4:7],axis=-1).max())
    actual=matrices(x[a:b,7:].reshape(n,24,6));original=matrices(candidate[a:b,7:].reshape(n,24,6))
    delta_rot=float(Rotation.from_matrix((original.swapaxes(-1,-2)@actual).reshape(-1,3,3)).magnitude().max())
    evidence.update(maximum_root_edit_m=delta_root,maximum_rotation_edit_rad=delta_rot,boundary_contract=contract(x,old,a,b))
    if delta_root>cfg["root_cap_m"] or delta_rot>cfg["rotation_cap_rad"]:
        return None,dict(evidence,status="PROPOSAL_EXCEEDS_TRUST_REGION",capacity_proven=False,necessary_infeasibility_within_registered_caps=False)
    assert np.array_equal(x[:a],old[:a]) and np.array_equal(x[b:],old[b:]) and np.array_equal(x[:,:4],candidate[:,:4])
    return x,dict(evidence,status="PROJECTED_NOT_YET_QUALIFIED",capacity_proven=False,necessary_infeasibility_within_registered_caps=False)
