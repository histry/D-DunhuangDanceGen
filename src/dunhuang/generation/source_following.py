"""Source-exact upper8 SO(3) supervision and a consumable Shadow primitive contract."""
import numpy as np
import torch
from .event_geometry import _matrix
from .conditions import source_condition
from .soft_reference import resample_native
from dunhuang.io import checked
from dunhuang.planning.time_map import source_positions
from dunhuang.planning.primitive_contract import verify

def geodesic_squared(a,b):
    relative=a.transpose(-1,-2)@b
    cosine=((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    skew=torch.stack((relative[...,2,1]-relative[...,1,2],relative[...,0,2]-relative[...,2,0],relative[...,1,0]-relative[...,0,1]),-1)
    sine=(skew.square().sum(-1)/4+1e-12).sqrt()
    return torch.atan2(sine,cosine).square()

def following_loss(predicted,reference,observed,keys,scale,bias,scope):
    if scope!="same_recording_pose_exact":raise ValueError("Strong following requires same-recording pose-exact supervision")
    if predicted.shape!=reference.shape or predicted.shape[1:]!=(150,151) or observed.shape!=reference.shape or keys.shape!=(*predicted.shape[:2],8):raise ValueError("Following clocks/masks differ")
    if observed.dtype!=torch.bool or keys.dtype!=torch.bool:raise ValueError("Boolean observed key support required")
    valid=keys&observed[...,79:127].reshape(-1,150,8,6).all(-1)
    if not valid.any(-1).any(-1).all():raise ValueError("Missing observed source keys")
    # EDGE uses rows, native source uses columns; local joints need transpose.
    raw=(predicted-bias)/scale
    pr=_matrix(raw[...,79:127].reshape(-1,150,8,6)).transpose(-1,-2)
    identity=torch.tensor([1,0,0,0,1,0],device=reference.device,dtype=reference.dtype)
    source=torch.where(valid[...,None],reference[...,79:127].reshape(-1,150,8,6),identity)
    rr=_matrix(source)
    value=geodesic_squared(pr,rr)
    return value[valid].mean()

def consume(plan,contract,policy):
    verify(contract,plan)
    if plan["frames"]!=150:raise ValueError("G2 research consumes one real window only")
    value=np.zeros((150,151),np.float32);available=np.zeros((150,151),bool);phase=np.zeros((150,1),np.float32);keys=np.zeros((150,8),bool)
    for p,e in zip(plan["events"],contract["events"]):
        if p["event_uid"]!=e["event_uid"]:raise ValueError("Canonical event ordering differs")
        a,b=e["music_interval"];lo,hi=e["source_interval"]
        source=np.load(checked(e["reference"]));mask=np.load(checked(e["source_observed"]))
        positions=source_positions(p,np.arange(a,b));left=np.floor(positions).astype(int);right=np.ceil(positions).astype(int)
        value[a:b]=resample_native(source,positions);available[a:b]=mask[left]&mask[right]
        phase[a:b,0]=(positions-lo)/(hi-lo)
        for k in e["key_motion"]:
            for j in k["observed_joints"]:
                if 12<=j<20:keys[k["music_frame"],j-12]=True
    c=source_condition(torch.from_numpy(value)[None],torch.from_numpy(available)[None],torch.from_numpy(phase)[None],policy)
    keys=torch.from_numpy(keys)[None]&c.known[...,79:127].reshape(1,150,8,6).all(-1)
    if not keys.any():raise ValueError("No source keys survive inference visibility")
    return c,keys

def interpolate_observed(base,reference,known):
    """Nonlearned comparator. Reads source upper8 only at actually observed slots."""
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import matrices,sixd
    result=base.copy();ticks=np.arange(len(base))
    for j in range(8):
        ids=np.flatnonzero(known[:,79+6*j:85+6*j].all(-1))
        if not len(ids):continue
        hi=np.clip(np.searchsorted(ids,ticks),0,len(ids)-1);lo=np.maximum(hi-1,0)
        left=ids[lo];right=ids[hi];alpha=np.divide(ticks-left,np.maximum(right-left,1)).clip(0,1)
        a=Rotation.from_matrix(matrices(reference[left,79+6*j:85+6*j]))
        b=Rotation.from_matrix(matrices(reference[right,79+6*j:85+6*j]))
        delta=(a.inv()*b).as_rotvec()
        result[:,79+6*j:85+6*j]=sixd((a*Rotation.from_rotvec(delta*alpha[:,None])).as_matrix()).astype(np.float32)
    return result

def hard_keyframe_inputs(normalized,keys):
    if normalized.shape[1:]!=(150,151) or keys.shape!=(*normalized.shape[:2],8) or keys.dtype!=torch.bool:raise ValueError("Hard key clock/mask differs")
    mask=torch.zeros_like(normalized,dtype=torch.bool);mask[...,79:127]=keys.repeat_interleave(6,-1)
    if not torch.isfinite(normalized[mask]).all():raise ValueError("Nonfinite actual hard key")
    return torch.where(mask,normalized,0),mask
