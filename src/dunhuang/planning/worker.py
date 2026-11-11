import sys,json
from pathlib import Path
from dunhuang.io import checked,write
from .planner import timeline,plan,propose_candidates
if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    output=Path(request['output'])
    music=timeline(checked(request['audio']),request['frames'])
    catalog=json.loads(checked(request['catalog']).read_text())
    reranker=None
    if request.get('alignment') and request.get('retrieval_policy')!='heuristic':
        from dunhuang.alignment.service import Reranker
        reranker=Reranker(request['alignment'],request['audio']['sha256'])
    if request.get('replanning'):
        from .event_plan import validate,revised_plan
        spec=request['replanning'];parent=validate(json.loads(checked(spec['parent']).read_text()),request['audio']['sha256'],request['frames'])
        slots=spec['changed_slots']
        fixed={i:e for i,e in enumerate(parent['events']) if i not in slots}
        excluded={i:[parent['events'][i]['event_uid']] for i in slots}
        proposals=propose_candidates(catalog,music,request['audio']['sha256'],max_plans=request.get('plan_count',3),
            condition_mode=request.get('generator','temporal'),fixed_events=fixed,excluded_events=excluded)
        proposals=[revised_plan(p,parent,spec['feedback'],'REPLAN_ASSOCIATED_SLOTS',slots) for p in proposals]
        write(output/'event_plans.json',proposals);result=proposals[0]
    elif request.get('input_event_plan'):
        from .event_plan import validate
        result=validate(json.loads(checked(request['input_event_plan']).read_text()),request['audio']['sha256'],request['frames'])
    elif request.get('plan_count',1)>1:
        proposals=propose_candidates(catalog,music,request['audio']['sha256'],max_plans=request['plan_count'],
            condition_mode=request.get('generator','temporal'))
        write(output/'event_plans.json',proposals);result=proposals[0]
    else:
        result=plan(catalog,music,request['audio']['sha256'],request.get('retrieval_policy','heuristic'),reranker=reranker,condition_mode=request.get('generator','temporal'))
    write(output/'music_timeline.json',music);write(output/'event_plan.json',result)
