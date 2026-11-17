"""Relative-slot EventPlan-compatible receipts, without changing any formal Planner route."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from .scorer import scores,batch_partial


class ShadowEventReranker:
    can_route=False
    def __init__(self,cfg):self.cfg=cfg
    def rank(self,query,candidates):
        partial,coverage,cost=batch_partial(query,candidates,self.cfg);result=[]
        for j,c in enumerate(candidates):
            value,_=scores(query,c,self.cfg,False);value.update(Partial=float(partial[j]),partial_coverage=float(coverage[j]),path_cost=float(cost[j]));result.append(value)
        return result


def run(benchmark,features,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh shadow receipts')
    meta=read(Path(benchmark)/'manifest.json');cfg=meta['config'];fm=read(Path(features)/'result.json');z=np.load(Path(features)/'arrays.npz');rows=meta['rows'];service=ShadowEventReranker(cfg);receipts=[];selected={}
    for t in meta['tasks']:selected.setdefault((t['role'],t['source_group']),t)
    def token(i):return {key:z['a'+str(i)+'_'+key] for key in ('mg','dg','mt','dt','mv','dv','positions')}
    for task in selected.values():
        query=rows[task['query']];values=service.rank(token(task['query']),[token(i) for i in task['candidates']]);candidates=[]
        for i,value in zip(task['candidates'],values):
            row=rows[i];a,b=row['source_interval'];L=query['frames'];candidates.append(dict(event_uid=row['parent_event_uid'],subwindow_uid=row['uid'],source_uid=row['source_uid'],source_sha256=row['raw_motion_sha256'],source_interval=[a,b],music_interval=[0,L],query_audio_frame_interval=query['audio_frame_interval'],time_warp=[[a,0],[b-1,L-1]],clock_contract='ACTUAL_FRAME_CENTERS_WITH_HALF_OPEN_SUPPORT',scores=value,evidence='FROZEN_R41_CONTENT_PHYSICAL_CLOCK_BENCHMARK_ONLY',can_route=False))
        receipts.append(dict(query_uid=query['uid'],audio_sha256=query['audio_sha256'],candidates=candidates,selected_path_unchanged=True,formal_planner_called=False))
    write(out,dict(receipts=receipts,benchmark_sha256=sha(Path(benchmark)/'manifest.json'),model_sha256=fm['checkpoint_sha256'],normalizer=fm['normalizer'],scorer_version='R41_LOCAL_TO_EVENT_V1',can_route=False,optimizer_steps=0,operational_qualified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.benchmark,a.features,a.out)
