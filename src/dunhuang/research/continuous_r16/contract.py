"""One global motion/event/condition clock, independent of 150-frame compute views."""
import numpy as np
from dunhuang.contracts import matrices,sixd
from dunhuang.planning.time_map import source_positions

def views(array):
    x=np.asarray(array)
    if len(x)!=225:raise ValueError("Exactly 225 global frames required")
    return np.stack((x[:150],x[75:225]))

def plan_grid(plan):
    n=plan["frames"];event_ids=np.full(n,-1,int);source=np.zeros(n,float);phase=np.zeros(n,float);cursor=0
    for i,e in enumerate(plan["events"]):
        a,b=e["music_interval"];lo,hi=e["source_interval"]
        if a!=cursor or not a<b<=n:raise ValueError("Contiguous half-open global plan required")
        t=np.arange(a,b);p=source_positions(e,t)
        event_ids[a:b]=i;source[a:b]=p;phase[a:b]=(p-lo)/(hi-lo);cursor=b
    if cursor!=n:raise ValueError("Incomplete plan")
    return dict(event_ids=event_ids,source_positions=source,phase=phase)

def proxy_plan(row,boundary):
    a=row["start_frame"]
    return dict(frames=150,kind="AIST_CONTIGUOUS_SOURCE_SEGMENTS_NOT_SEMANTIC_EVENTS",
        events=[dict(event_uid=row["recording_uid"]+":"+str(a+lo)+":"+str(a+hi),
            source_uid=row["raw_motion"]["sha256"],source_interval=[a+lo,a+hi],music_interval=[lo,hi],
            time_warp=[[a+lo,lo],[a+hi,hi]],reference=row["source_native"])
            for lo,hi in ((0,boundary),(boundary,150))])

def observations(target,boundary,joints,context=8):
    x=np.asarray(target);k=np.zeros_like(x,bool);edge=k.copy();event=k.copy()
    edge[:context,4:]=True;edge[-context:,4:]=True
    centers=[boundary//2,boundary+(150-boundary)//2]
    for t in centers:
        for j in joints:event[max(0,t-1):min(150,t+2),7+6*j:13+6*j]=True
    k=edge|event
    return np.where(k,x,0).astype(np.float32),k,edge,event,centers

def continuous_pair(left,right):
    for field in ("recording_uid","source_group","role"):
        if left[field]!=right[field]:raise ValueError("Cross-recording/role pair forbidden")
    for field in ("raw_motion","source_native","audio","clock"):
        if left[field]["sha256"]!=right[field]["sha256"]:raise ValueError("Pair source identity differs")
    if right["start_frame"]-left["start_frame"]!=75:raise ValueError("150-frame windows must overlap by 75")
    if left["role"]!="train":raise ValueError("Train-only experiment")
    return True

def patch_control(value,known,edge,event,wrong,arm):
    v=np.array(value,copy=True);k=np.array(known,copy=True)
    if arm=="C1_removed":k[event]=False
    elif arm=="C1_wrong_event":v[event]=wrong[event]
    elif arm=="C1_wrong_context":v[edge]=wrong[edge]
    elif arm not in ("C1_correct","C1_wrong_music"):raise ValueError("Unknown C1 control")
    return np.where(k,v,0).astype(np.float32),k

def align_donor(donor,recipient):
    d=donor.copy();A=matrices(recipient[0,7:13]);B=matrices(d[0,7:13]);turn=A@B.T
    d[:,4:7]=recipient[0,4:7]+(donor[:,4:7]-d[0,4:7])@turn.T
    d[:,7:13]=sixd(turn@matrices(donor[:,7:13]));return d


def condition_payload(native,mask,window_phase,normalized,known,grid):
    if len(grid["phase"]) not in (150,225,450):raise ValueError("Registered plan horizon required")
    if window_phase.shape!=native.shape[:2]+(1,) or mask.shape!=native.shape:raise ValueError("Window condition shape differs")
    return dict(native=native,mask=mask,window_phase=window_phase,normalized=normalized,known=known,**grid)
