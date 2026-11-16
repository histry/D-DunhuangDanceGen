"""Variable-duration joint source/time search. All ranking is explicitly heuristic."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .core import identity,multi_warp,rotation_sample
from .music import next_boundaries

def search(catalog,music,lib,cfg):
    units=catalog['units'];frames=music['frames'];energy=np.array(music['energy_percentile']);novel=np.array(music['novelty'])
    duration=np.array([e['duration_frames']-1 for e in units]);activity=np.array([e['activity_percentile'] for e in units]);quality=np.array([min(e['boundary_speed'],2.)+.2*e['phase_crossings'] for e in units])
    cache={}
    def options(a,b):
        key=(a,b)
        if key in cache:return cache[key]
        rate=(b-a-1)/duration;valid=(rate>=.75)&(rate<=1.33)
        mismatch=abs(activity-energy[a:b].mean())+.12*quality+.06*abs(np.log(np.maximum(rate,1e-8)))
        order=sorted(np.flatnonzero(valid),key=lambda j:(float(mismatch[j]),units[j]['unit_uid']))
        retained=[];family_count={};source_count={}
        for j in order:
            u=units[j];f=u['family_uid'];s=u['source_uid']
            if family_count.get(f,0)>=2 or source_count.get(s,0)>=3:continue
            mapped=multi_warp(u['source_interval'],[a,b],u['phases'],music['beats'])
            if mapped is None:continue
            e={k:copy.deepcopy(v) for k,v in u.items() if k not in ('phases',)}
            e.update(music_interval=[a,b],time_warp=mapped['time_warp'],time_contract=mapped['clock_contract'],phase_anchors=mapped['anchor_source_frames'],
                scores=dict(activity_mismatch=float(abs(activity[j]-energy[a:b].mean())),boundary_proxy=float(quality[j]),time_cost=mapped['cost'],sdmr=None),
                ranking_scope='ACTIVITY_AND_GEOMETRIC_HEURISTIC; NO_LEARNED_OR_CULTURAL_QUALIFICATION')
            r=lib.sample(e,[a,a+1,b-2,b-1]);e['_entry']=r[:2];e['_exit']=r[2:]
            retained.append((float(mismatch[j]+.05*mapped['cost']),e));family_count[f]=family_count.get(f,0)+1;source_count[s]=source_count.get(s,0)+1
            if len(retained)>=cfg['candidates_per_arc']:break
        cache[key]=retained;return retained
    states={0:[(0.,[])]};visited=0
    for a in range(frames):
        if a not in states:continue
        current=states[a];visited+=1
        for b in next_boundaries(a,music,cfg):
            bank=options(a,b)
            if not bank:continue
            end_bonus=.15*float(novel[min(b,frames-1)])
            expanded=list(states.get(b,[]))
            for oldcost,path in current:
                for cost,e in bank:
                    repeat=sum(p['family_uid']==e['family_uid'] for p in path[-3:])
                    transition=0.
                    if path:
                        prev=path[-1];difference=prev['_exit'][-1].swapaxes(-1,-2)@e['_entry'][0]
                        transition=float(Rotation.from_matrix(difference).magnitude().mean())
                        va=Rotation.from_matrix((prev['_exit'][0].swapaxes(-1,-2)@prev['_exit'][1])).as_rotvec()*30
                        vb=Rotation.from_matrix((e['_entry'][0].swapaxes(-1,-2)@e['_entry'][1])).as_rotvec()*30
                        transition+=.05*float(np.linalg.norm(va-vb,axis=-1).mean())
                    # Cost per interval is duration weighted, plus explicit switch cost.
                    total=oldcost+cost*(b-a)/180+.18+.35*transition+.45*repeat-end_bonus
                    expanded.append((total,path+[e]))
            states[b]=sorted(expanded,key=lambda p:(p[0],[e['unit_uid'] for e in p[1]]))[:cfg['beam']]
    if frames not in states:raise ValueError('No legal complete plan: do not over-stretch or insert holds')
    cost,selected=states[frames][0];events=[]
    for e in selected:
        e=copy.deepcopy(e);e.pop('_entry');e.pop('_exit');e['candidate_uid']=identity([e['unit_uid'],e['music_interval'],e['time_warp']]);events.append(e)
    assert events[0]['music_interval'][0]==0 and events[-1]['music_interval'][1]==frames
    assert all(x['music_interval'][1]==y['music_interval'][0] for x,y in zip(events,events[1:]))
    return dict(schema='variable_source_plan_r36',frames=frames,fps=30,events=events,cost=cost,visited_boundaries=visited,
        candidate_arcs=len(cache),sdmr_used=False,human_reviewed=False,operational_qualified=False,
        complete_motion_semantics='UNKNOWN',world_Root='UNKNOWN',contact='UNKNOWN')
