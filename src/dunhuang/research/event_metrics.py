"""Physical-unit event diagnostics. No quality thresholds or semantic labels."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices, motion151


def rotation_error(a, b, joints=tuple(range(12, 20))):
    a, b = motion151(a), motion151(b)
    if a.shape != b.shape:
        raise ValueError('Reference and output clocks differ')
    ra = matrices(a[:, 7:].reshape(-1, 24, 6))[:, joints]
    rb = matrices(b[:, 7:].reshape(-1, 24, 6))[:, joints]
    angle = Rotation.from_matrix((ra.swapaxes(-1, -2) @ rb).reshape(-1, 3, 3)).magnitude()
    return float(angle.mean())


def event_margin(x1, x2, e1, e2):
    own = (rotation_error(x1, e1) + rotation_error(x2, e2)) / 2
    other = (rotation_error(x1, e2) + rotation_error(x2, e1)) / 2
    return dict(own_reference_radians=own, other_reference_radians=other,
                preference_margin_radians=other-own,
                scope='paired counterfactual pose preference, not semantic or physical qualification')


def dynamics(motion, fk):
    motion = motion151(motion)
    fk = np.asarray(fk)
    if fk.shape != (len(motion), 24, 3) or not np.isfinite(fk).all() or len(fk) < 4:
        raise ValueError('Finite matching 24-joint FK required')
    r = matrices(motion[:, 7:].reshape(-1, 24, 6))
    angle = Rotation.from_matrix((r[:-1].swapaxes(-1,-2) @ r[1:]).reshape(-1,3,3)).magnitude().reshape(-1,24)*30
    velocity = np.diff(fk, axis=0)*30
    jerk = np.linalg.norm(np.diff(fk,n=3,axis=0)*30**3,axis=-1)
    peak = np.unravel_index(np.argmax(jerk),jerk.shape)
    return dict(root_speed_mean_mps=float(np.linalg.norm(np.diff(motion[:,4:7],axis=0)*30,axis=-1).mean()),
                upper8_angular_speed_mean_radps=float(angle[:,12:20].mean()),
                leg_angular_speed_mean_radps=float(angle[:,(1,2,4,5,7,8,10,11)].mean()),
                joint_speed_mean_mps=float(np.linalg.norm(velocity,axis=-1).mean()),
                jerk_peak_mps3=float(jerk.max()),jerk_p95_mps3=float(np.percentile(jerk,95)),
                jerk_peak_four_frame_start=int(peak[0]),jerk_peak_joint=int(peak[1]),
                contact_prediction_mean=motion[:,:4].mean(0).astype(float).tolist(),
                observed_contact_quality='NOT_INFERRED_FROM_PREDICTED_CONTACT_CHANNELS')


def descriptor(tokens):
    tokens = np.asarray(tokens,dtype=np.float64)
    if tokens.ndim != 2 or not np.isfinite(tokens).all():
        raise ValueError('Finite time x feature tokens required')
    return np.concatenate([v.mean(0) for v in np.array_split(tokens,5)])


def representation_stats(features):
    x = np.asarray(features,dtype=np.float64)
    x = x - x.mean(0,keepdims=True)
    values = np.linalg.svd(x,compute_uv=False)**2
    if values.sum() < 1e-20:
        return dict(effective_rank=0.,centered_rms=0.)
    p = values/values.sum()
    return dict(effective_rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())),
                centered_rms=float(np.sqrt((x*x).mean())))


def identity_probe(features, labels, views):
    """Closed-set event identity across visibility views; NOT unseen-event accuracy."""
    x=np.asarray(features,dtype=float);labels=np.asarray(labels);views=np.asarray(views)
    train=views!=2;test=views==2;classes=np.unique(labels)
    if not train.any() or not test.any() or len(classes)<2:
        raise ValueError('Multiple identities and held visibility view required')
    center=x[train].mean(0);scale=np.sqrt(((x[train]-center)**2).mean())+1e-12
    a=(x[train]-center)/scale;b=(x[test]-center)/scale
    y=(labels[train,None]==classes[None]).astype(float)
    alpha=1e-2
    dual=np.linalg.solve(a@a.T+alpha*np.eye(len(a)),y)
    prediction=classes[np.argmax(b@a.T@dual,axis=1)]
    return dict(accuracy=float((prediction==labels[test]).mean()),chance=1/len(classes),
                training_samples=int(train.sum()),validation_samples=int(test.sum()),
                classes=len(classes),ridge_alpha=alpha,
                scope='closed-set identity, held visibility view; no source generalization claim')


def event_timing(output,reference):
    """Upper8 angular-speed phase proxy, not musical/semantic correctness."""
    from scipy.signal import find_peaks
    def profile(x):
        r=matrices(motion151(x)[:,79:127].reshape(-1,8,6))
        return Rotation.from_matrix((r[:-1].swapaxes(-1,-2)@r[1:]).reshape(-1,3,3)).magnitude().reshape(-1,8).mean(1)*30
    a,b=profile(output),profile(reference)
    if len(a)!=len(b):raise ValueError('Event timing clocks differ')
    result=dict(output_activity_radps=float(a.mean()),reference_activity_radps=float(b.mean()),
                scope='angular-speed timing proxy; not musical quality or learned warp proof')
    if a.std()<1e-6 or b.std()<1e-6:
        return dict(result,zero_lag_correlation=None,best_lag_frames=None,peak_gap_seconds=None,
                    reason='INSUFFICIENT_MOTION_VARIATION')
    corrs=[]
    for lag in range(-15,16):
        av=a[max(0,lag):len(a)+min(0,lag)]
        bv=b[max(0,-lag):len(b)+min(0,-lag)]
        corrs.append(float(np.corrcoef(av,bv)[0,1]) if av.std()>1e-6 and bv.std()>1e-6 else -1.)
    pa=find_peaks(a,distance=6,prominence=.25*a.std())[0]
    pb=find_peaks(b,distance=6,prominence=.25*b.std())[0]
    return dict(result,zero_lag_correlation=corrs[15],best_lag_frames=int(np.argmax(corrs))-15,
                best_lag_correlation=max(corrs),output_peak_count=len(pa),reference_peak_count=len(pb),
                peak_gap_seconds=float(np.min(abs(pb[:,None]-pa[None,:]),axis=1).mean()/30) if len(pa) and len(pb) else None)
