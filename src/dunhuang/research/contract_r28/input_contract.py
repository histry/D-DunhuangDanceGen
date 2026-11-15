"""Published Lodge preprocessing contract; height features are not support/contact truth."""
import numpy as np
FEET=(7,8,10,11)
THRESHOLDS=np.array([.12,.12,.05,.05],dtype=np.float32)

def alignment_height(first_toes):
    x=np.asarray(first_toes)
    if x.shape!=(2,) or not np.isfinite(x).all():raise ValueError("Two finite source-first toe heights required")
    return float(x.mean() if abs(float(x[0]-x[1]))<.02 else x.min())

def height_features(points):
    p=np.asarray(points)
    if p.ndim!=3 or p.shape[1:]!=(22,3) or not np.isfinite(p).all():raise ValueError("Finite Tx22x3 FK required")
    return (p[:,FEET,1]<THRESHOLDS).astype(np.float32)

def translate(raw,offset):
    raw=np.asarray(raw)
    if raw.ndim!=2 or raw.shape[1]!=139 or np.shape(offset)!=(3,):raise ValueError("139D and one fixed rigid offset required")
    value=raw.copy();value[:,4:7]+=offset
    return value

def foot_proxy(points,ground=0.):
    feet=np.asarray(points)[:,FEET];v=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0)*30,axis=-1)
    near=feet[:-1,:,1]<=ground+.05
    return dict(near_floor_elements=int(near.sum()),horizontal_speed_p95_mps=float(np.percentile(v[near],95)) if near.any() else None,
        minimum_height_m=float(feet[:,:,1].min()),fixed_model_floor_y_m=ground,
        contact_truth="UNKNOWN",scope="HEIGHT_GEOMETRY_PROXY_NOT_STATIC_SUPPORT",measured_contact=False)
