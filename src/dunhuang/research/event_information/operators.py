"""No-learned-weights information diagnostics; labels and features stay separate."""
import numpy as np

def event_labels(events,frames,kind,interval=False,tolerance=2):
    frames=np.asarray(frames);value=np.zeros(len(frames),bool)
    for e in events:
        if e["type"]!=kind:continue
        if kind=="upper_pose_hold" and interval:
            a,b=e["evidence_span"]
            if not 0<=a<b:raise ValueError("Observed half-open hold span required")
            value|=(frames>=a)&(frames<b)
        else:value|=abs(frames-e["frame"])<=tolerance
    return value

def phase_context(events,frames,clip_seconds=3.):
    frames=np.asarray(frames);parts=[];known=[]
    for kind in ("upper_motion_onset","upper_direction_turn"):
        ticks=np.array(sorted(set(e["frame"] for e in events if e["type"]==kind)))
        if not len(ticks):
            parts.extend([np.zeros(len(frames)),np.zeros(len(frames))]);known.extend([np.zeros(len(frames),bool)]*2);continue
        p=np.searchsorted(ticks,frames,side="right")-1;q=np.searchsorted(ticks,frames,side="left")
        before=p>=0;after=q<len(ticks)
        parts.extend([np.clip((frames-ticks[np.maximum(p,0)])/30,0,clip_seconds),np.clip((ticks[np.minimum(q,len(ticks)-1)]-frames)/30,0,clip_seconds)])
        known.extend([before,after])
    # Speed-peak and hold labels are deliberately not passed as feature channels.
    return np.stack(parts,1),np.stack(known,1)

def moments(x,mask):
    x=np.asarray(x);mask=np.asarray(mask)
    if x.shape!=mask.shape or mask.dtype!=bool or not np.isfinite(x[mask]).all():raise ValueError("Finite observed feature support required")
    return dict(count=mask.sum(0).astype(np.int64),sum=np.where(mask,x,0).sum(0,dtype=np.float64),squared=np.square(np.where(mask,x,0).astype(float)).sum(0))

def finalize(parts):
    count=sum(p["count"] for p in parts);total=sum(p["sum"] for p in parts);sq=sum(p["squared"] for p in parts)
    mean=total/np.maximum(count,1);std=np.sqrt(np.maximum(0,sq/np.maximum(count,1)-mean**2)).clip(1e-6)
    return dict(count=count,mean=mean,std=std)

def normalize(x,mask,stats):
    valid=mask&(np.asarray(stats["count"])>0)
    return np.where(valid,(x-stats["mean"])/stats["std"],0).astype(np.float32),valid

def distance(x,xmask,y,ymask):
    # Pairwise observed intersections; unknown zero storage never becomes truth.
    x=np.asarray(x,float);y=np.asarray(y,float);xm=np.asarray(xmask,float);ym=np.asarray(ymask,float)
    xx=np.where(xm.astype(bool),x,0);yy=np.where(ym.astype(bool),y,0)
    count=xm@ym.T;error=(xx*xx)@ym.T+xm@(yy*yy).T-2*xx@yy.T
    return np.divide(np.maximum(error,0),count,out=np.full_like(error,np.inf),where=count>0)

def neighbours(blocks,query,bank,groups,k=5,batch=256):
    chosen=[]
    for start in range(0,len(query),batch):
        q=query[start:start+batch];total=np.zeros((len(q),len(bank)));number=np.zeros_like(total)
        for x,mask in blocks:
            d=distance(x[q],mask[q],x[bank],mask[bank]);ok=np.isfinite(d)
            total+=np.where(ok,d,0);number+=ok
        d=np.divide(total,number,out=np.full_like(total,np.inf),where=number>0)
        # Train queries leave their whole music group out; Val never shares a Train song.
        d[groups[q,None]==groups[bank][None]]=np.inf
        if np.any(np.isfinite(d).sum(1)<k):raise ValueError("Insufficient independent Train neighbour bank")
        # Stable ties use the pre-registered ordered bank, never labels.
        ids=np.argsort(d,axis=1,kind="stable")[:,:k]
        chosen.append(bank[ids])
    return np.concatenate(chosen)

def select_per_source(points,rows,role,cap):
    selected={}
    for j,(asset,frame,i,t) in enumerate(points):
        if rows[i]["role"]==role:selected.setdefault(asset,[]).append(j)
    return np.array([ids[p] for asset,ids in sorted(selected.items()) for p in np.unique(np.linspace(0,len(ids)-1,min(cap,len(ids))).round().astype(int))])

def static_witness():
    from scipy.spatial.transform import Rotation
    from dunhuang.contracts import sixd
    from dunhuang.planning.time_map import motion_features74
    a=np.zeros((30,151),np.float32);a[:,7:]=np.tile([1,0,0,0,1,0],(30,24));b=a.copy()
    b[:,79:127]=np.tile(sixd(Rotation.from_euler("z",45,degrees=True).as_matrix()),(30,8))
    av,am=motion_features74(a);bv,bm=motion_features74(b)
    return dict(dynamic_features_identical=bool(np.array_equal(av,bv) and np.array_equal(am,bm)),upper8_rotation_difference_rad=float(np.pi/4),
        both_are_holds=True,static_identity_is_not_hold_state=True,semantic_claim=False)
