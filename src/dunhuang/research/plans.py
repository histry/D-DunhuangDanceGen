import copy,json
from pathlib import Path
from dunhuang.io import checked,write
from dunhuang.planning.planner import plan
from dunhuang.planning.event_plan import validate

def counterfactual_plans(assets,prior,out):
    original=json.loads((Path(prior)/'probe150/event_plan.json').read_text())
    timing=json.loads((Path(prior)/'probe150/music_timeline.json').read_text())
    catalog=json.loads(checked(assets['catalog']).read_text())
    used={e['source_uid'] for e in original['events']}
    alternate=plan(dict(entries=[r for r in catalog['entries'] if r['source_uid'] not in used]),timing,
        original['audio_sha256'],condition_mode='adapter')
    shifted=copy.deepcopy(original);changes=[]
    for event in shifted['events']:
        for offset in (2,-2,1,-1):
            candidate=copy.deepcopy(event)
            candidate['primitive']['music_frame']+=offset
            for row in candidate['time_warp'][1:-1]:row[1]+=offset
            probe=copy.deepcopy(shifted)
            probe['events'][shifted['events'].index(event)]=candidate
            try:validate(probe,probe['audio_sha256'],probe['frames'])
            except ValueError:continue
            event.update(candidate);changes.append(dict(event=event['event_uid'],offset_frames=offset));break
    if not changes:raise ValueError('No legal phase intervention')
    shifted['intervention']=dict(kind='registered legal time-map perturbation only',changes=changes)
    result={}
    for name,p in dict(original=original,replaced=alternate,time_warp_changed=shifted).items():
        validate(p,p['audio_sha256'],150);file=out/(name+'.json');write(file,p);result[name]=str(file)
    return result
