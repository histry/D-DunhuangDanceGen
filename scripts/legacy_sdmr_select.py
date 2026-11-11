"""Score feasible events and record qualified selection or fixed-source fallback."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write,sha
from dunhuang.contracts import MotionAdapter,matrices
from dunhuang.pipeline import run

def select(model,result,events,embeddings,sources,out,root):
    q=json.loads((root/'outputs/supported_upper8/5s/request.json').read_text())
    receipt=json.loads((root/'inputs/sdmr_query.json').read_text())
    if receipt['source_sha256']!=q['audio']['sha256'] or sha(root/'inputs/sdmr_query.npz')!=receipt['features_sha256']:
        raise ValueError('Query feature source binding differs')
    z=np.load(root/'inputs/sdmr_query.npz')
    scaler=np.load(root/'outputs/b2_c7be67c/5s_custom_none/scaler.npz')
    feasible=[]
    for i,event in enumerate(events):
        a,b=event['raw_interval'];center=(a+b)//2
        if center-a<12 or b-center<=12:continue
        motion=sources[event['source_uid']][2][center-12:center+13]
        edge=MotionAdapter.to_edge(motion)
        if np.any(abs(edge[:,79:127]*scaler['scale'][79:127]+scaler['bias'][79:127])>1+32*np.finfo(np.float32).eps):continue
        feasible.append(dict(index=i,event=event,clip=motion,center=center))
    norm=result['normalization']['music'];scores=[];topk=[]
    for anchor in q['anchors']:
        t=anchor['frame'];start=max(0,t-60)
        music=torch.as_tensor((z['music'][start:start+120]-norm['mean'])/norm['std']*z['mask'][start:start+120],device='cuda').float()[None]
        mask=torch.as_tensor(z['mask'][start:start+120],device='cuda')[None]
        with torch.no_grad():vector=model.music(music,mask)[0].cpu().numpy()[0]
        score=embeddings@vector;scores.append(score)
        ranking=sorted(feasible,key=lambda c:(-score[c['index']],c['event']['event_uid']))[:5]
        topk.append(dict(frame=t,candidates=[dict(event_uid=c['event']['event_uid'],source_uid=c['event']['source_uid'],
            score=float(score[c['index']]),semantic=c['event']['semantic'],support_receipt=dict(supported=True,joints=list(range(12,20)))) for c in ranking]))
    write(out/'ranking.json',dict(topk=topk,feasible_center_events=len(feasible),checkpoint_sha256=result['retrieval_checkpoint_sha256'],
        enabled=result['qualified'],expert_preference='UNKNOWN'))
    selected=[];used=set()
    for score in scores:
        available=[c for c in feasible if c['event']['event_uid'] not in used]
        if not available:break
        c=max(available,key=lambda c:float(score[c['index']]))
        used.add(c['event']['event_uid']);selected.append(c)
    if result['qualified'] and len(selected)==3:
        anchors=[]
        for target,candidate in zip(q['anchors'],selected):
            file=out/('anchor_'+str(target['frame'])+'.npy');np.save(file,candidate['clip'])
            anchors.append(dict(frame=target['frame'],pose=candidate['clip'][12].tolist(),source_frame=12,
                joints=list(range(12,20)),source=dict(path=str(file),sha256=sha(file)),catalog_id=candidate['event']['event_uid']))
        write(out/'selection.json',dict(checkpoint_sha256=result['retrieval_checkpoint_sha256'],anchors=anchors))
        q['learned_retrieval']=dict(checkpoint=dict(path=str(out/'retrieval.pt'),sha256=sha(out/'retrieval.pt')),
            qualification=dict(path=str(out/'result.json'),sha256=sha(out/'result.json')),
            selection=dict(path=str(out/'selection.json'),sha256=sha(out/'selection.json')))
    else:
        q['learned_retrieval']=dict(checkpoint=dict(path=str(out/'retrieval.pt'),sha256=sha(out/'retrieval.pt')),
            qualification=dict(path=str(out/'result.json'),sha256=sha(out/'result.json')))
    q['mode']='temporal';q['trajectory_radius']=12;q['trajectory_strength']=.55
    q['output']=str(out/'generation_5s')
    run(q)
