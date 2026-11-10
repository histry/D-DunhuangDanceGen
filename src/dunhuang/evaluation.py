import numpy as np
from .contracts import matrices

def evaluate(motion, positions, anchors=(), beat_frames=()):
    speed = np.linalg.norm(np.diff(positions,axis=0),axis=-1).mean(axis=1)*30
    minima = np.where((speed[1:-1]<speed[:-2]) & (speed[1:-1]<=speed[2:]))[0]+2
    beats = np.asarray(beat_frames, dtype=int)
    result = dict(speed_mean_mps=float(speed.mean()),
        beat_alignment_mean_seconds=float(np.min(abs(beats[:,None]-minima[None,:]),axis=1).mean()/30) if len(beats) and len(minima) else None,
        beat_metric_scope='motion-speed minima proximity; not a musical-quality verdict',
        independent_dunhuang_style_quality='NOT_EVALUATED',naturalness_human_rating='NOT_EVALUATED')
    errors=[]
    for anchor in anchors:
        joints=anchor.get('joints',list(range(1,22)))
        a=matrices(motion[anchor['frame'],7:].reshape(24,6))[joints]
        b=matrices(np.asarray(anchor['pose'])[7:].reshape(24,6))[joints]
        angle=np.arccos(np.clip((np.trace(np.swapaxes(a,-1,-2)@b,axis1=-2,axis2=-1)-1)/2,-1,1))
        errors.append(dict(frame=anchor['frame'],mean_radians=float(angle.mean()),max_radians=float(angle.max())))
    result['anchor_errors']=errors
    # Report every actual 75-frame window boundary, in physical units.
    result['window_boundaries']=[dict(frame=t,
        max_joint_step_m=float(np.linalg.norm(positions[t]-positions[t-1],axis=-1).max()),
        max_joint_velocity_change_mps=float(np.linalg.norm((positions[t]-positions[t-1])-(positions[t-1]-positions[t-2]),axis=-1).max()*30))
        for t in range(75,len(motion),75)]
    return result
