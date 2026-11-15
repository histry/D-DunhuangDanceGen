"""Frozen P real-source reconstruction; explicitly separate clipped and affine diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .representation import packet
from dunhuang.research.state_r27.common import *
from dunhuang.research.state_r27.trace import row_noise

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    pop=read(root/"data/result.json");data={r["case"]:dict(np.load(root/"data"/r["arrays"])) for r in pop["records"]};cases=[r["case"] for r in pop["records"]]
    d,n,s,f,e,info,a=runtime(read("src/dunhuang/research/state_r27/config.json"));restore(a,Path(cfg["r27"])/"models/P/adapter.pt")
    music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda");records=[];arrays={}
    references={mode:packet([data[c]["model_raw"][64:160,7:] for c in (cases if mode=="correct" else cases[1:]+cases[:1])]) for mode in ("correct","wrong")}
    for contract in ("legacy","model"):
        target=torch.tensor(np.stack([data[c][contract+"_target"] for c in cases]),device="cuda")
        raw=np.stack([data[c][contract+"_raw"] for c in cases]);source=np.stack([data[c][contract+"_points"] for c in cases])
        with torch.no_grad():back=n.unnormalize(target.clone());bp=f(back,s)[:,:,:22]
        for t in cfg["times"]:
            ts=torch.full((8,),t,device="cuda",dtype=torch.long)
            with row_noise([cfg["seed"]+i for i in range(8)],device="cuda"):epsilon=torch.randn_like(target)
            xt=d.q_sample(target,ts,epsilon)
            for mode in ("correct","wrong","none"):
                a.clear()
                if mode!="none":a.set_reference(**references[mode])
                with torch.no_grad():
                    pred=d.model(xt,music,None,ts,cond_drop_prob=0)
                    clipped=n.unnormalize(pred.clone());cp=f(clipped,s)[:,:,:22]
                    affine=n.scaler.inverse_transform(pred.reshape(-1,139).clone()).reshape_as(pred);ap=f(affine,s)[:,:,:22]
                arrays[f"{contract}_{t}_{mode}"]=pred.cpu().numpy()
                for i,c in enumerate(cases):
                    records.append(dict(case=c,role=pop["records"][i]["role"],contract=contract,t=t,reference=mode,
                        normalized_body_mse=float((pred[i,64:160,7:]-target[i,64:160,7:]).square().mean()),
                        predicted_saturation_fraction=float((pred[i,64:160].abs()>1).float().mean()),
                        official_clipped_vs_original=metrics(clipped[i].cpu().numpy(),cp[i].cpu().numpy(),raw[i],source[i]),
                        official_clipped_vs_roundtrip=metrics(clipped[i].cpu().numpy(),cp[i].cpu().numpy(),back[i].cpu().numpy(),bp[i].cpu().numpy()),
                        affine_unclipped_diagnostic_vs_original=metrics(affine[i].cpu().numpy(),ap[i].cpu().numpy(),raw[i],source[i])))
    result=dict(records=records,arm="FROZEN_R27_P",raw_affine_scope="DIAGNOSTIC_ONLY_NOT_SAMPLER_OR_ACCEPTANCE",physical_branch_names_explicit=True,
        source_reference_canonical=True,optimizer_steps=0,sampler_calls=0,generated_trajectories=0,operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close();print("R28_FROZEN_P_READOUT_DONE",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.replay_from)
