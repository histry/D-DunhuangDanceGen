"""Separate mapped source nodes from redetected actual output events."""
import numpy as np
from dunhuang.planning.anchor_reachability import multiple
from dunhuang.research.paired_event_r41.events import profiles,phases
from .contracts import CFG,read

def audit(unit,candidate,R,events):
    a,b=unit['source_interval'];A,B=candidate['target_interval'];w=np.asarray(candidate['time_warp'])
    cfg=read('src/dunhuang/research/paired_event_r41/config.json')
    # Use the same frozen detector configuration as R41, not tuned on new songs.
    detector=cfg.get('events',cfg)
    actual=phases(profiles(R),detector)
    chosen=[]
    for p in sorted(unit['phases'],key=lambda p:(p['type']!='speed_peak',p['frame'],p['body_region'])):
        if a+12<p['frame']<b-12 and all(abs(p['frame']-x['frame'])>18 for x in chosen):chosen.append(p)
        if len(chosen)==CFG['event_limit']:break
    rows=[];requests=[]
    for p in chosen:
        s=p['frame'];projected=A+(s-a)*(B-A-1)/(b-a-1);mapped=float(np.interp(s,w[:,0],w[:,1]))
        near=[e for e in events if e['kind'] in ('beat','downbeat','accent_candidate') and abs(e['frame_float']-projected)<=CFG['proposal_radius']]
        target=min(near,key=lambda e:(abs(e['frame_float']-projected),e['uid'])) if near else None
        output=[A+x['frame'] for x in actual if x['type']==p['type'] and x['body_region']==p['body_region']]
        nearest=min(output,key=lambda x:abs(x-mapped)) if output else None
        row=dict(source_frame=s,type=p['type'],body_region=p['body_region'],uniform_projection=projected,mapped_source_event_frame=mapped,
                 target=target,actual_output_same_type_frame=nearest,source_anchor_error=None if target is None else abs(mapped-target['frame_float']),
                 output_event_error=None if target is None or nearest is None else abs(nearest-target['frame_float']),
                 output_association='NEAREST_SAME_TYPE_TO_MAPPED_SOURCE_NOT_IDENTITY_OR_ARTISTIC_TRUTH',status='NULL' if target is None else 'UNVALIDATED_HYPOTHESIS')
        if target:requests.append(dict(source_frame=s,requested=target['frame_float']-A,tolerance=CFG['event_tolerance']))
        rows.append(row)
    proposal=multiple([a,b],B-A,requests)
    return dict(correspondences=rows,joint_proposal_certificate=proposal,proposal_clock_executed=False,
                candidate_clock_executed=True,semantic_quality='NOT_EVALUABLE',redetected_events=actual)
