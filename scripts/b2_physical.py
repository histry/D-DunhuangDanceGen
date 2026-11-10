"""One independent root-XZ correction on the saved official whole-song baseline."""
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from b2 import ROOT, read, write, pinned, sha

def main(out):
    parent=ROOT/'outputs/experiment_v1/baseline_whole'
    q=read(parent/'request.json')
    sys.path.insert(0,str(pinned(q['legacy_root'],q['legacy_commit'])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.physics import audit_motion_np,derive_contacts_np
    from motion.geometry import fk_24_np
    from motion.config import MotionGenerationConfig
    from motion_geometry import physical as native
    from motion_geometry.smpl24 import FOOT_JOINTS
    from contracts.physical_quality import evaluate_physical_audit
    out.mkdir(parents=True,exist_ok=False)
    x=np.load(parent/'native.npy')
    fk=np.load(parent/'native_fk.npy')
    feet=fk[:,list(FOOT_JOINTS)]
    contacts,_,_,_=derive_contacts_np(x,MotionGenerationConfig(),precomputed_joints=fk)
    height=native.median_filter_bool_np(feet[...,1]<=np.percentile(feet[...,1],5)+.055,native._odd_window(1/12,30))
    states=native.classify_support_states_np(fk,contacts.astype(bool),fps=30,height_support=height,support_policy='final_fail_closed')
    support=(states[1:]==native.STATIC_SUPPORT)&(states[:-1]==native.STATIC_SUPPORT)
    n=len(x)
    # Least squares for a smooth horizontal translation, measured in metres per frame.
    d=sparse.diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n))
    acc=sparse.diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
    w=sparse.diags(support.sum(1).astype(float))
    steps=np.diff(feet[:,:,[0,2]],axis=0)
    rhs=-(d.T@np.sum(steps*support[:,:,None],axis=1))
    lhs=d.T@w@d+.015*sparse.eye(n)+12*(acc.T@acc)
    correction=spsolve(lhs.tocsc(),rhs)
    correction*=min(1.,.06/max(1.e-12,np.linalg.norm(correction,axis=1).max()))
    proposal=x.copy();proposal[:,[4,6]]+=correction.astype(np.float32)
    pos=fk_24_np(proposal)
    shift=np.zeros((n,3));shift[:,[0,2]]=proposal[:,[4,6]]-x[:,[4,6]]
    actual_error=float(abs(pos-(fk+shift[:,None,:])).max())
    if actual_error>1e-5:raise ValueError('Root shift / Native FK mismatch')
    audit=audit_motion_np(proposal,precomputed_joints=pos)
    gate=evaluate_physical_audit(audit)
    old=read(parent/'physical.json')
    if old['absolute']['limits']!=gate['limits']:raise ValueError('Native thresholds changed')
    np.save(out/'native.npy',proposal);np.save(out/'native_fk.npy',pos)
    write(out/'physical.json',dict(audit=audit,absolute=gate))
    metrics=['foot_skate_mps_p95','foot_support_drift_m_p95','joint_jerk_mps3_max','joint_jerk_mps3_p95','root_horizontal_net_displacement_m']
    write(out/'comparison.json',dict(parent_sha256=sha(parent/'native.npy'),candidate_sha256=sha(out/'native.npy'),
        active_foot_steps=int(support.sum()),max_shift_m=float(np.linalg.norm(shift,axis=1).max()),fk_error=actual_error,
        before={k:old['audit'][k] for k in metrics},after={k:audit[k] for k in metrics},
        reasons=gate['reasons'],native_pass=gate['ok']))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
