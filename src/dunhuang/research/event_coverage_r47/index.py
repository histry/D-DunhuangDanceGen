"""Catalog V2 research sidecar; preserve every legacy support gap and raw event."""
import argparse,subprocess,bisect
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,identity,sha,Library,components
from dunhuang.research.paired_event_r41.events import profiles,phases
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

def segment_source(p,protected,cfg,interval):
    lo,hi=interval;minimum=cfg['minimum_core_frames'];maximum=cfg['maximum_core_frames'];preferred=cfg['preferred_core_frames'];signals=[]
    for name in ('upper','lower','trunk','root_speed'):
        if name in p:
            v=gaussian_filter1d(p[name],cfg['boundary_smoothing_sigma_frames']);signals.append(v/max(float(np.quantile(v,.75)),.05))
    activity=np.mean(signals,axis=0);valleys=find_peaks(-activity,distance=cfg['boundary_grid_frames'])[0];candidates=sorted(set([lo,hi]+list(range(lo+minimum,hi-minimum+1,cfg['boundary_grid_frames']))+[int(t) for t in valleys if lo+minimum<=t<=hi-minimum]));candidates=[t for t in candidates if t in (lo,hi) or not any(a<t<b for a,b in protected)];states={lo:(0.,[])}
    for b in candidates[1:]:
        k=bisect.bisect_right(candidates,b-minimum);start=bisect.bisect_left(candidates,b-maximum);options=candidates[start:k]
        # Long jumps are only allowed across a protected span with no legal interior cut.
        if k:options+=candidates[bisect.bisect_right(candidates,candidates[k-1]-minimum):start]
        if b==hi and hi-lo<minimum:options=[lo]
        choices=[]
        for a in options:
            if a not in states or a>=b:continue
            length=b-a
            if length>maximum and bisect.bisect_left(candidates,a+minimum)<bisect.bisect_right(candidates,b-minimum):continue
            cost,path=states[a];value=cost+(0. if b==hi else float(activity[b]))+.4*((length-preferred)/preferred)**2+.08;choices.append((value,path+[[a,b]]))
        if choices:states[b]=min(choices,key=lambda v:(v[0],v[1]))
    return states[hi][1] if hi in states else [[lo,hi]]

def merge_protected(intervals,protected):
    if not intervals:return []
    cuts=[intervals[0][0]]+[b for a,b in intervals[:-1] if not any(x<b<y for x,y in protected)]+[intervals[-1][1]]
    return [[a,b] for a,b in zip(cuts[:-1],cuts[1:])]

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh R47 catalog')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json');catalog=read('src/dunhuang/research/complete_r36/config.json')['catalog'];lib=Library(catalog);roles=read('src/dunhuang/research/source_context_r42/config.json');units=[];sources=[]
    for uid in sorted(lib.rows):
        R=lib.rotations[uid];n=len(R);p=profiles(R);phase=phases(p,cfg);support=components([r['source_interval'] for r in lib.rows[uid]]);protected=[q['evidence_span'] for q in phase if q['type'] in ('pose_hold','root_turn_span')]+[[max(0,q['frame']-cfg['key_protection_frames']),min(n,q['frame']+cfg['key_protection_frames']+1)] for q in phase if q['type']=='speed_peak'];source_units=[]
        for lo,hi in support:
            proposed=segment_source(p,protected,cfg,[lo,hi]);spans=merge_protected(proposed,protected)
            assert spans[0][0]==lo and spans[-1][1]==hi and all(a[1]==b[0] for a,b in zip(spans[:-1],spans[1:]))
            for a,b in spans:
                ps=[q for q in phase if a<=q['frame']<b];cross=[q for q in ps if q['evidence_span'][0]<a or q['evidence_span'][1]>b];v=p['upper'][a:b];entry=dict(unit_uid='r47:'+identity(dict(native=lib.rows[uid][0]['reference']['sha256'],interval=[a,b],algorithm='WHOLE_REGION_DP_MERGE_PROTECTED_V1')),source_uid=uid,source_sha256=lib.rows[uid][0]['source_sha256'],native_sha256=lib.rows[uid][0]['reference']['sha256'],role='train' if uid in roles['change_train_recordings'] else 'held_diagnostic',source_interval=[a,b],duration_frames=b-a,support_component=[lo,hi],left_context=[max(lo,a-30),a],right_context=[b,min(hi,b+30)],phases=ps,stage_evidence_truncated=len(cross),mean_upper_speed_radps=float(v.mean()),p95_upper_speed_radps=float(np.quantile(v,.95)),phase_support='KINEMATIC_COMPLETE' if not cross else 'TRUNCATED_AT_LEGACY_SUPPORT_BOUNDARY',semantic_completeness='UNKNOWN',human_reviewed=False,world_root='UNKNOWN',contact='UNKNOWN',legacy_event_uids=[r['event_uid'] for r in lib.rows[uid] if max(a,r['source_interval'][0])<min(b,r['source_interval'][1])],time_variant_limits=[.75,1.33],connection_feasibility='REQUIRES_PAIR_SPECIFIC_R47_CHECK',can_route=False);units.append(entry);source_units.append(entry)
        mask=np.zeros(n,bool)
        for a,b in support:mask[a:b]=True
        unknown=[];edges=np.diff(np.r_[False,~mask,False].astype(int))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):unknown.append([int(a),int(b)])
        sources.append(dict(source_uid=uid,frames=n,legacy_events=len(lib.rows[uid]),accepted_components=support,accepted_frames=int(mask.sum()),unknown_quality_intervals=unknown,unknown_quality_frames=int((~mask).sum()),candidate_units=len(source_units),candidate_frames=sum(r['duration_frames'] for r in source_units),whole_record_phases=len(phase),protected_spans=protected,native_sha256=lib.rows[uid][0]['reference']['sha256'],role=source_units[0]['role']))
        print('R47_CATALOG',uid,len(source_units),'accepted',mask.sum(),'unknown',int((~mask).sum()),flush=True)
    write(out/'result.json',dict(units=units,sources=sources,config=cfg,catalog_sha256=sha(catalog),legacy_events_preserved=sum(r['legacy_events'] for r in sources),accepted_coverage_preserved=True,raw_deletions=0,new_semantic_labels=0,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_steps=0,can_route=False));assert all(s['candidate_frames']==s['accepted_frames'] for s in sources)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
