"""GPU interface qualification before any optimizer: visibility, length, poison and frozen weights."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import write
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.common import state
from dunhuang.research.global_clock_r17.train import digest
from .common import load,cache_data,selected_digest,frozen_except_null
from .conditions import endpoint

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();records=[];arrays={}
    for layout in ("split","joint"):
        w,d,data,p,head=load(cfg,layout);cache=cache_data(cfg,w,d,data)
        y,m,o,v,k,h,meta,t,xt=state(cfg,cache,0,519,True,cfg["seed"]+10)
        p.telemetry.enabled=True
        with torch.no_grad():
            base,_=p(xt,m,v,k,meta,t)
            poison=v.clone();poison[~k]=float("nan");other,_=p(xt,m,poison,k,meta,t);assert torch.equal(base,other)
            donor=cache["cache"][1][1];vv,kk,hh,original=endpoint(y,donor,"right_wrong")
            wrong,_=p(xt,m,vv,kk,meta,t)
            arrays[layout+"_correct"]=base[0].cpu().numpy();arrays[layout+"_right_wrong"]=wrong[0].cpu().numpy()
            nohead,_=p(xt,m,v,k,meta,t,"removed");nohead_wrong,_=p(xt,m,vv,kk,meta,t,"removed")
            arrays[layout+"_nohead"]=nohead[0].cpu().numpy();arrays[layout+"_nohead_wrong"]=nohead_wrong[0].cpu().numpy()
            far_equal=torch.equal(base[:,8:75,4:],wrong[:,8:75,4:])
            if layout=="split":assert far_equal
        records.append(dict(layout=layout,selected_digest=selected_digest(w),head_digest=digest(head),
            frozen_except_null=frozen_except_null(w),frozen_digest=w.digest(True),
            motion_parameters=sum(p.numel() for p in w.selected()),head_parameters=sum(p.numel() for p in head.parameters()),
            total_parameters=sum(p.numel() for p in w.parameters())+sum(p.numel() for p in head.parameters()),
            null_adaptation=getattr(w,"null_adaptation",None),hidden_poison_exact=True,
            right_far_response=float((base[:,8:75,4:]-wrong[:,8:75,4:]).square().mean().sqrt()),
            right_far_response_without_state=float((nohead[:,8:75,4:]-nohead_wrong[:,8:75,4:]).square().mean().sqrt()),
            far_equal=bool(far_equal),backbone_frames=225 if layout=="joint" else 150))
        p.close()
    a,b=records;assert a["selected_digest"]==b["selected_digest"] and a["head_digest"]==b["head_digest"]
    assert a["frozen_except_null"]==b["frozen_except_null"]
    assert b["total_parameters"]-a["total_parameters"]==38400
    np.savez_compressed(out/"predictions.npz",**arrays);result=dict(records=records,optimizer_steps=0,operational_qualified=False)
    if replay:
        same_arrays(out/"predictions.npz",Path(replay)/"predictions.npz");assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,task="inbetween"))
    print("R20_INTERFACE",records,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.out,a.replay_from)
