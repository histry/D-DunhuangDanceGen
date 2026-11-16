"""Versioned observed-body transition proxies, not physical feasibility or cultural truth."""
import numpy as np
from scipy.spatial.transform import Rotation

def zones(event,edit_max=12):
    a,b=event["music_interval"];tick=event["primitive"]["music_frame"]
    p=[tick-12,tick+13]
    if not a<=p[0]<p[1]<=b:raise ValueError("Protected primitive outside event")
    l=min(edit_max,p[0]-a);r=min(edit_max,b-p[1])
    return dict(protected_core=p,entry_edit=[a,a+l],exit_edit=[b-r,b],full_event=[a,b])

def descriptor(rotations,points):
    R=np.asarray(rotations);P=np.asarray(points,dtype=np.float64)
    if R.shape[1:]!=(22,3,3) or P.shape!=(len(R),22,3) or len(R)<4 or not np.isfinite(R).all() or not np.isfinite(P).all():
        raise ValueError("Observed common22 time series required")
    # Use actual retimed terminal samples, never the unwarped catalog entry/exit.
    n=min(12,len(R)//2)
    vel=np.diff(P,axis=0)*30;acc=np.diff(P,n=2,axis=0)*900
    return dict(entry_R=R[0],exit_R=R[-1],entry_P=P[0],exit_P=P[-1],
        entry_v=vel[:n-1].mean(0),exit_v=vel[-n+1:].mean(0),
        entry_a=acc[:n-2].mean(0),exit_a=acc[-n+2:].mean(0))

def edges(left,right,left_events,right_events):
    def stack(side,key):return np.stack([x[key] for x in side])
    A=stack(left,"exit_R");B=stack(right,"entry_R")
    rel=A[:,None].swapaxes(-1,-2)@B[None]
    angle=Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude().reshape(len(A),len(B),22).mean(-1)
    gap=stack(left,"exit_P")[:,None]-stack(right,"entry_P")[None]
    position=np.sqrt((gap*gap).mean(axis=(-1,-2)))
    # Wrists, hands' relative arrangement, pelvis/lower body all have explicit readouts.
    wrist=np.sqrt((gap[:,:,[20,21]]**2).mean(axis=(-1,-2)))
    lower=np.sqrt((gap[:,:,[0,1,2,4,5,7,8,10,11]]**2).mean(axis=(-1,-2)))
    velocity=np.sqrt(((stack(left,"exit_v")[:,None]-stack(right,"entry_v")[None])**2).mean(axis=(-1,-2)))
    acceleration=np.sqrt(((stack(left,"exit_a")[:,None]-stack(right,"entry_a")[None])**2).mean(axis=(-1,-2)))
    budgets=np.array([[zones(a)["exit_edit"][1]-zones(a)["exit_edit"][0]+zones(b)["entry_edit"][1]-zones(b)["entry_edit"][0] for b in right_events] for a in left_events])
    duration=np.maximum(budgets,1)/30
    required_speed=position/duration
    # Fixed units registered before evaluation, no fitted threshold or safety classification.
    cost=angle+.5*position/.2+.25*wrist/.2+.15*lower/.2+.2*velocity/.5+.1*acceleration/5+.1*required_speed
    return dict(cost=cost,angle_rad=angle,body_gap_m=position,wrist_gap_m=wrist,lower_gap_m=lower,
        velocity_gap_m_s=velocity,acceleration_gap_m_s2=acceleration,editable_frames=budgets,required_mean_speed_proxy=required_speed)

def pool(options,baseline,cfg):
    threshold=baseline["scores"]["heuristic"]-cfg["music_slot_regret_max"]
    ordered=sorted((e for e in options if e["scores"]["heuristic"]>=threshold-1e-12),key=lambda e:(-e["scores"]["heuristic"],e["event_uid"]))
    counts={};keep=[]
    for e in ordered:
        key=e["source_uid"]
        if counts.get(key,0)<cfg["per_source_candidates"]:
            keep.append(e);counts[key]=counts.get(key,0)+1
    if not any(e["candidate_uid"]==baseline["candidate_uid"] for e in keep):keep.append(baseline)
    if not keep:raise ValueError("No qualified music candidate")
    return keep

def search(banks,matrices,old,cfg):
    beams=[(0.,[],0.)]
    for i,bank in enumerate(banks):
        expanded=[]
        for cost,path,regret in beams:
            for j,e in enumerate(bank):
                delta=old[i]["scores"]["heuristic"]-e["scores"]["heuristic"]
                next_regret=regret+delta
                if next_regret>cfg["music_mean_regret_max"]*(i+1)+1e-10:continue
                prev=[banks[k][v] for k,v in enumerate(path)]
                repeat=.3*sum(x["event_uid"]==e["event_uid"] for x in prev)+.04*sum(x["source_uid"]==e["source_uid"] for x in prev)
                c=cost-e["scores"]["heuristic"]+repeat+(cfg["transition_weight"]*matrices[i-1]["cost"][path[-1],j] if i else 0)
                expanded.append((float(c),path+[j],float(next_regret)))
        # Preserve the original prefix as an explicit feasible fallback, not a proven optimum.
        old_path=[next(j for j,e in enumerate(banks[k]) if e["candidate_uid"]==old[k]["candidate_uid"]) for k in range(i+1)]
        if not any(p[1]==old_path for p in expanded):
            expanded.append((path_cost(old_path,banks,matrices,cfg),old_path,0.))
        beams=sorted(expanded,key=lambda p:(p[0],p[1]))[:cfg["beam"]]
    return beams[0]

def path_cost(path,banks,matrices,cfg):
    cost=0.;seen=[]
    for i,j in enumerate(path):
        e=banks[i][j];cost-=e["scores"]["heuristic"]
        cost+=.3*sum(x["event_uid"]==e["event_uid"] for x in seen)+.04*sum(x["source_uid"]==e["source_uid"] for x in seen)
        if i:cost+=cfg["transition_weight"]*matrices[i-1]["cost"][path[i-1],j]
        seen.append(e)
    return float(cost)
