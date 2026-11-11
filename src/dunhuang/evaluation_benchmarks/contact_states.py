"""Semantic contact states require verified labels, never high-speed exemption."""
import numpy as np

def diagnose(joints,floor_y,predicted_contact=None,labels=None,label_evidence=None,contact_height=.05,fps=30):
    x=np.asarray(joints)
    if x.ndim!=3 or x.shape[1:]!=(24,3) or not np.isfinite(x).all():raise ValueError('Finite SMPL24 joints required')
    feet=x[:,[7,10,8,11]];low=feet[:,:,1]-floor_y<=contact_height
    declared=np.zeros_like(low) if predicted_contact is None else np.asarray(predicted_contact)>=.5
    if declared.shape!=low.shape:raise ValueError('Four predicted contact channels required')
    candidate=low|declared;state=np.full(low.shape,-1,int)
    state[~candidate]=0 # geometric swing/flight candidate, not an intentional action label
    if labels is not None:
        if not label_evidence or label_evidence.get('validated') is not True:
            raise ValueError('Verified external contact-state evidence required')
        labels=np.asarray(labels)
        if labels.shape!=low.shape or not np.isin(labels,[-1,0,1,2]).all():raise ValueError('State label contract differs')
        state=labels.copy()
    speed=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0),axis=-1)*fps
    result=dict(schema='dunhuang_contact_extension_v1',semantic_calibration='NOT_CALIBRATED' if labels is None else 'EXTERNAL_VERIFIED_LABELS',
        states=dict(unknown_contact=-1,swing_or_flight=0,static=1,intentional_sliding=2),
        counts={str(i):int((state==i).sum()) for i in (-1,0,1,2)},
        intentional_sliding_inferred_from_speed=False,paper_skate_exemptions_applied=False,
        predicted_contact_accuracy=None,predicted_contact_f1=None,
        contact_truth_status='NO_INDEPENDENT_CONTACT_GROUND_TRUTH',
        caveat='Static/sliding semantics remain UNKNOWN without validated external evidence')
    for name,mask in [('unknown',state==-1),('static',state==1),('sliding',state==2)]:
        valid=mask[:-1]&mask[1:]
        result[name+'_horizontal_speed_p95_mps']=float(np.percentile(speed[valid],95)) if valid.any() else None
    return result
