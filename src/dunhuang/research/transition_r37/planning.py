"""Fixed R36 catalog/music, full interval history, diverse beam, multi-frame FK edges."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import identity,multi_warp,relative_fk
from dunhuang.research.complete_r36.music import next_boundaries
from .memory import incremental,reuse

def boundary_descriptor(e,lib,J,parents):
    a,b=e['music_interval'];ticks=np.r_[np.arange(a,a+12),np.arange(b-12,b)]
    R=lib.sample(e,ticks);P=relative_fk(R,J,parents)
    return dict(entry_R=R[0],exit_R=R[-1],entry_P=P[0],exit_P=P[-1],
                entry_v=np.diff(P[:12],axis=0).mean(0)*30,exit_v=np.diff(P[-12:],axis=0).mean(0)*30,
                entry_a=np.diff(P[:12],n=2,axis=0).mean(0)*900,exit_a=np.diff(P[-12:],n=2,axis=0).mean(0)*900)

def edge(a,b):
    A=a['_desc'];B=b['_desc']
    angle=float(Rotation.from_matrix(A['exit_R'].swapaxes(-1,-2)@B['entry_R']).magnitude().mean())
    gap=A['exit_P']-B['entry_P'];rmse=lambda x:float(np.sqrt(np.mean(x*x)))
    pos=rmse(gap);wrist=rmse(gap[[20,21]]);lower=rmse(gap[[0,1,2,4,5,7,8,10,11]])
    velocity=rmse(A['exit_v']-B['entry_v']);acc=rmse(A['exit_a']-B['entry_a'])
    duration=48/30
    return dict(cost=angle+.5*pos/.2+.25*wrist/.2+.15*lower/.2+.2*velocity/.5+.1*acc/5+.1*pos/duration,
                angle_rad=angle,body_gap_m=pos,wrist_gap_m=wrist,lower_gap_m=lower,velocity_gap_m_s=velocity,
                acceleration_gap_m_s2=acc,editable_frames=48,scope='OBSERVED_BODY_PROXY_NOT_FEASIBILITY')

def diverse(rows,width):
    kept=[];counts={}
    for row in sorted(rows,key=lambda p:(p[0],[e['unit_uid'] for e in p[1]])):
        path=row[1];last=path[-1]
        history=tuple(sorted({(e['source_uid'],e['source_interval'][0]//180) for e in path}))
        signature=(last['source_uid'],last['family_uid'],history)
        if counts.get(signature,0)>=2:continue
        kept.append(row);counts[signature]=counts.get(signature,0)+1
        if len(kept)>=width:break
    return kept

def edge_cost_matrix(left,right):
    """Same per-pair arithmetic as edge(), one SciPy rotation batch instead of Python pair loops."""
    stack=lambda items,key:np.stack([e['_desc'][key] for e in items])
    A=stack(left,'exit_R');B=stack(right,'entry_R');rel=A[:,None].swapaxes(-1,-2)@B[None]
    angle=Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude().reshape(len(left),len(right),22).mean(-1)
    gap=stack(left,'exit_P')[:,None]-stack(right,'entry_P')[None]
    rms=lambda x:np.sqrt((x*x).reshape(len(left),len(right),-1).mean(-1))
    pos=rms(gap);wrist=rms(gap[:,:,[20,21]]);lower=rms(gap[:,:,[0,1,2,4,5,7,8,10,11]])
    velocity=rms(stack(left,'exit_v')[:,None]-stack(right,'entry_v')[None]);acc=rms(stack(left,'exit_a')[:,None]-stack(right,'entry_a')[None])
    return angle+.5*pos/.2+.25*wrist/.2+.15*lower/.2+.2*velocity/.5+.1*acc/5+.1*pos/(48/30)

def search(catalog,music,lib,cfg,J,parents):
    units=catalog['units'];N=music['frames'];energy=np.asarray(music['energy_percentile']);novel=np.asarray(music['novelty'])
    duration=np.array([u['duration_frames']-1 for u in units]);activity=np.array([u['activity_percentile'] for u in units])
    def options(a,b):
        rate=(b-a-1)/duration;valid=(rate>=.75)&(rate<=1.33)
        mismatch=abs(activity-energy[a:b].mean())+.06*abs(np.log(np.maximum(rate,1e-8)))
        ordered=sorted(np.flatnonzero(valid),key=lambda j:(float(mismatch[j]),units[j]['unit_uid']))
        # Source-balanced first round, then variants: prevent mean-energy hubs hiding all alternatives.
        bysource={}
        for j in ordered:bysource.setdefault(units[j]['source_uid'],[]).append(j)
        candidates=[v[k] for k in range(3) for _,v in sorted(bysource.items()) if len(v)>k]
        kept=[]
        for j in candidates:
            u=units[j];w=multi_warp(u['source_interval'],[a,b],u['phases'],music['beats'])
            if w is None:continue
            e={k:copy.deepcopy(v) for k,v in u.items() if k!='phases'}
            e.update(music_interval=[a,b],time_warp=w['time_warp'],time_contract=w['clock_contract'],phase_anchors=w['anchor_source_frames'],
                     scores=dict(activity_mismatch=float(mismatch[j]),time_cost=w['cost'],content_score=None,timing_score=None,sdmr=None))
            e['candidate_uid']=identity([e['unit_uid'],[a,b],e['time_warp']])
            e['_desc']=boundary_descriptor(e,lib,J,parents);kept.append((float(mismatch[j]+.05*w['cost']),e))
            if len(kept)>=cfg['candidates_per_arc']:break
        return kept
    states={0:[(0.,[])]}
    for a in range(N):
        if a not in states:continue
        current=states.pop(a) # Consumed frontier can never be visited again: all arcs advance time.
        for b in next_boundaries(a,music,cfg):
            bank=options(a,b)
            if not bank:continue
            expanded=list(states.get(b,[]))
            previous={path[-1]['candidate_uid']:path[-1] for _,path in current if path}
            left=list(previous.values());left_ids={e['candidate_uid']:i for i,e in enumerate(left)}
            transitions=edge_cost_matrix(left,[e for _,e in bank]) if left else None
            for cost,path in current:
                for j,(local,e) in enumerate(bank):
                    memory=incremental(path,e,music);trans=0.
                    if path:
                        trans=float(transitions[left_ids[path[-1]['candidate_uid']],j])
                    extra=cfg['reuse_weight']*memory['overlap_fraction']*memory['theme_discount']+cfg['backtrack_weight']*memory['backtrack']*memory['theme_discount']
                    total=cost+local*(b-a)/180+.18+cfg['transition_weight']*trans+extra-.15*novel[min(b,N-1)]
                    expanded.append((float(total),path+[e]))
            states[b]=diverse(expanded,cfg['beam'])
        if a%600==0:print('R37_SEARCH_FRAME',a,N,flush=True)
    if N not in states:raise ValueError('No legal complete plan')
    cost,path=states[N][0];events=[];boundaries=[]
    for i,e in enumerate(path):
        row={k:copy.deepcopy(v) for k,v in e.items() if not k.startswith('_')};row['history_evidence']=incremental(path[:i],e,music);events.append(row)
        if i:boundaries.append(dict(frame=e['music_interval'][0],left=path[i-1]['candidate_uid'],right=e['candidate_uid'],**edge(path[i-1],e)))
    return dict(schema='source_memory_plan_r37_v1',frames=N,fps=30,events=events,cost=cost,boundaries=boundaries,reuse=reuse(events),
                sdmr_used=False,operational_qualified=False,human_reviewed=False,Root='UNKNOWN',contact='UNKNOWN',semantics='KINEMATIC_CANDIDATES_ONLY',
                catalog_changed=False,music_candidates_changed=False)
