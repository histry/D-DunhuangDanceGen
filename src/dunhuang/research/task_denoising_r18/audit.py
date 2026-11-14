"""Frozen R17 attention diagnostics only: do not re-run hard/soft factor sampling."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.global_clock_r17.engine import load as old_load,tensors,observed
from dunhuang.research.global_clock_r17.contract import metadata
from .engine import population,parent_config,unknown_mse,history_contract
from .model import State,Predictor

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();pre,fit,ev=population(cfg);rows=fit[:2]+ev
    arrays={};records=[];models={}
    for mode,oldmode in (("self","self"),("full","cross"),("unique","cross")):
        reference=read(Path(cfg["parent"])/"models"/oldmode/"result.json")["checkpoint"]
        w,d,data,old,trained=old_load(parent_config(cfg),oldmode,reference);old.close()
        head=State(mode,cfg["state_width"]).cuda();head.load_state_dict(trained.state_dict());head.eval()
        before=w.digest();predict=Predictor(w,head)
        for row in rows:
            z,y,m=tensors(row)
            for task in cfg["tasks"]:
                v,k,h=observed(y,task,prefix=y[0,:75].cpu().numpy() if task=="progressive" else None)
                meta=torch.tensor(metadata()[None],device="cuda");t=torch.tensor([519],device="cuda")
                seed=cfg["seed"]+30000+row["case"]+(100 if row["role"]=="eval" else 0)
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);xt=d.q_sample(y,t)
                with torch.no_grad():
                    for control in cfg["controls"]:
                        p,_=predict(xt,m,v,k,meta,t,control)
                        key=f'{mode}_{row["role"]}_{row["case"]}_{task}_{control}'
                        arrays[key]=p[0].cpu().numpy()
                        records.append(dict(mode=mode,role=row["role"],case=row["case"],task=task,control=control,
                            unknown_mse=float(unknown_mse(p,y,k)),attention=dict(head.last)))
        assert w.digest()==before;predict.close();models[mode]=reference
    np.savez_compressed(out/"predictions.npz",**arrays)
    result=dict(records=records,models=models,optimizer_steps=0,history_contract_proposal=history_contract(),
        generated_history_training=False,new_whole_song_generations=0,operational_qualified=False)
    if replay:
        same_arrays(out/"predictions.npz",Path(replay)/"predictions.npz");assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        parent_data=ref(Path(cfg["parent"])/"data/protocol.json"),scope="Frozen same-state diagnosis; exposed eval never chooses training settings"))
    print("R18_P0",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.out,a.replay_from)
