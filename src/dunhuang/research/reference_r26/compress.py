"""Train-only fixed ridge diagnostic of frozen final reference memories, no optimizer."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import sha,write
from dunhuang.research.lodge_adapter_r25.metrics import source_metrics
from .common import *
from .geometry import fk22

def run(cfg,root,out,replay=None):
    out=fresh(out);root=Path(root);forbid_optimizers();pop,data=dataset(root)
    d,n,s,f,e,info,adapter=runtime(cfg,"C0");rows=[];arrays={}
    for arm in cfg["arms"]:
        torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
        a=Adapter(arm!="C0",width=64,part_width=16,layers=(1,4,7),hidden=512).cuda()
        restore(a,root/f"models/{arm}/adapter.pt",cfg,arm)
        memories={}
        with torch.no_grad():
            for case in pop["train_cases"]+pop["dev_cases"]:
                p=packet([data[case]["raw139"][64:160,7:]]);install(a,p);memories[case]=a.context[0][0,1:].cpu().numpy().astype(float)
            p=packet([data[pop["train_probe"][0]]["raw139"][64:160,7:]]);install(a,p);before=a.context[0].clone()
            p["features"][:,:,[16,20]]=p["features"][:,:,[20,16]];install(a,p)
            swap=float((a.context[0]-before).abs().max())
        train=np.concatenate([memories[c] for c in pop["train_cases"]]);target=np.concatenate([data[c]["raw139"][64:160,7:] for c in pop["train_cases"]])
        mean=train.mean(0);scale=train.std(0).clip(1e-6)
        def design(x):return np.c_[(x-mean)/scale,np.ones(len(x))]
        x=design(train);penalty=np.eye(x.shape[1])*1e-3;penalty[-1,-1]=0
        weights=np.linalg.solve(x.T@x+penalty,x.T@target)
        arrays[arm+"_ridge"]=weights;arrays[arm+"_mean"]=mean;arrays[arm+"_scale"]=scale
        for case in pop["train_cases"]+pop["dev_cases"]:
            prediction=(design(memories[case])@weights).astype(np.float32);motion=np.zeros((96,139),np.float32);motion[:,7:]=prediction
            with torch.no_grad():points=fk22(torch.tensor(motion[None],device="cuda"),s)[0].cpu().numpy()
            source=data[case]["raw139"][64:160];sp=data[case]["source22"][64:160]
            metrics=source_metrics(motion,source,points,sp,np.ones(96,bool));role="train" if case in pop["train_cases"] else "dev_exposed"
            rows.append(dict(arm=arm,case=case,role=role,metrics=metrics))
        arrays[arm+"_identity_swap_delta"]=np.array(swap)
        print("R26_COMPRESS",arm,swap,flush=True)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    result=dict(records=rows,ridge=1e-3,training_only_fit=True,readout_neural_optimizer_steps=0,optimizer_steps=0,
        decoder_is_a_diagnostic_not_a_generator=True,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(path)))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/"arrays.npz")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
