"""Hierarchical source proposals; old atomic candidates and raw arrays remain unchanged."""
from collections import defaultdict
import numpy as np
from dunhuang.data.change_event_phases import motion_profiles,detect
from .core import components,identity,rotation_sample

def build(lib,cfg):
    units=[];audit=[];holds=[]
    phase_cfg=dict(minimum_peak_distance=6,speed_prominence_floor_radps=.05,acceleration_prominence_floor_radps2=1.,hold_quantile=.2,hold_max_speed_radps=.2,hold_min_frames=6)
    for uid,rows in lib.rows.items():
        rows=sorted(rows,key=lambda e:e['source_interval'][0]);r=lib.rotations[uid]
        profiles=motion_profiles(lib.native[uid],list(range(22)));phases=detect(profiles,phase_cfg);speed=profiles['upper8']
        spans=components([e['source_interval'] for e in rows]);low=[p['evidence_span'] for p in phases if p['type']=='upper_pose_hold' and p['evidence_span'][1]-p['evidence_span'][0]>=cfg['hold_protection_frames']]
        holds.extend(dict(source_uid=uid,interval=p) for p in low)
        candidates=set()
        for lo,hi in spans:
            boundaries={lo,hi}
            boundaries.update(t for e in rows for t in e['source_interval'] if lo<=t<=hi)
            boundaries.update(t for p in low for t in p if lo<t<hi)
            def held(t):return any(a<t<b for a,b in low)
            protected=[t for t in sorted(boundaries) if t in (lo,hi) or not held(t)]
            for a in protected[:-1]:
                legal=[b for b in protected if cfg['minimum_unit_frames']<=b-a<=cfg['maximum_unit_frames']]
                for target in cfg['unit_lengths']:
                    if legal:candidates.add((a,min(legal,key=lambda b:(abs(b-a-target),b))))
            for a,b in low:
                if lo<=a<b<=hi and cfg['minimum_unit_frames']<=b-a<=cfg['maximum_unit_frames']:candidates.add((a,b))
        # If a continuously moving run has no long hold, all atomic boundaries remain options.
        for a,b in sorted(candidates):
            containing=next(([lo,hi] for lo,hi in spans if lo<=a<b<=hi),None)
            if containing is None:raise AssertionError('Proposed unit crosses a quality-rejected gap')
            atomics=[e['event_uid'] for e in rows if max(a,e['source_interval'][0])<min(b,e['source_interval'][1])]
            ps=[p for p in phases if a<=p['frame']<b]
            crossing=[p for p in ps if p['evidence_span'][0]<a or p['evidence_span'][1]>b]
            sample=rotation_sample(r,np.linspace(a,b-1,16));rv=sample[...,:2,:].reshape(-1)
            v=float(speed[a:b].mean());amplitude=float(np.sqrt(np.mean(np.ptp(sample,axis=0)**2)))
            boundary=float((speed[a]+speed[b-1])/2)
            units.append(dict(unit_uid='unit_'+identity([uid,a,b]),source_uid=uid,source_sha256=rows[0]['source_sha256'],reference=rows[0]['reference'],
                source_interval=[a,b],duration_frames=b-a,atomic_event_uids=atomics,phases=ps,
                phase_crossings=len(crossing),mean_upper_speed=v,amplitude=amplitude,boundary_speed=boundary,
                completeness='KINEMATIC_PROPOSAL_NOT_SEMANTIC_TRUTH',human_reviewed=False,world_Root='UNKNOWN',contact='UNKNOWN',
                support_component=containing,_sample=sample,_descriptor=rv))
        audit.append(dict(source_uid=uid,frames=len(r),covered_components=spans,accepted_frames=sum(b-a for a,b in spans),
            proposed_units=sum(e['source_uid']==uid for e in units),long_hold_spans=low))
    if not units:raise ValueError('No whole-source unit proposals')
    # Deterministic geometric family screening. Keep all variants; never delete raw motions.
    prototypes=[];descriptors=[];families=[]
    for e in units:
        family=None
        if descriptors:
            d=np.array(descriptors);dist=((d-e['_descriptor'])**2).mean(1)
            for j in np.argsort(dist)[:5]:
                p=prototypes[j];cos=np.clip(((p['_sample']*e['_sample']).sum((-2,-1))-1)/2,-1,1);angles=np.arccos(cos)
                ratio=(e['amplitude']+1e-5)/(p['amplitude']+1e-5)
                if angles.mean()<cfg['family_angle_rad'] and np.quantile(angles,.95)<cfg['family_p95_rad'] and .7<=ratio<=1/.7:
                    family=families[j];break
        if family is None:
            family='family_'+identity([e['unit_uid'],'unreviewed_geometry_v1']);prototypes.append(e);descriptors.append(e['_descriptor']);families.append(family)
        e['family_uid']=family;e['family_scope']='UNCALIBRATED_GEOMETRIC_SCREEN_NOT_CULTURAL_LABEL'
    values=np.array([e['mean_upper_speed'] for e in units])
    for e in units:
        e['activity_percentile']=float(np.mean(values<e['mean_upper_speed']))
        e.pop('_sample');e.pop('_descriptor')
    return dict(schema='hierarchical_source_units_r36',fps=30,units=units,sources=audit,long_holds=holds,
        old_events_preserved=sum(len(r) for r in lib.rows.values()),family_count=len(families),raw_deletions=0,operational_qualified=False)
