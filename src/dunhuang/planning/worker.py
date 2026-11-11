import sys,json
from pathlib import Path
from dunhuang.io import checked,write
from .planner import timeline,plan
if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    output=Path(request['output'])
    music=timeline(checked(request['audio']),request['frames'])
    catalog=json.loads(checked(request['catalog']).read_text())
    reranker=None
    if request.get('alignment') and request.get('retrieval_policy')!='heuristic':
        from dunhuang.alignment.service import Reranker
        reranker=Reranker(request['alignment'],request['audio']['sha256'])
    result=plan(catalog,music,request['audio']['sha256'],request.get('retrieval_policy','heuristic'),reranker=reranker)
    write(output/'music_timeline.json',music);write(output/'event_plan.json',result)
