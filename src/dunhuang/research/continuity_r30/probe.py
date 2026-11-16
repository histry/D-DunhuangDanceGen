"""Actual visited-state readout of local/global gates, not an attention quality claim."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime,values
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,forbid_optimizers,same_arrays,restore
from dunhuang.research.state_r27.trace import sample,trace_arrays

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();d,n,s,f,e,info,a=runtime(cfg)
    data,_=values(cfg,"aist","train");music=torch.tensor(np.stack([x["music"] for x in data]),device="cuda")
    eval_arrays=np.load(root/"reading_train/arrays.npz");records=[];arrays={};calls=0
    for arm,folder in (("R1","reading_train"),("R2","temporal_evaluation")):
        if not (root/f"models/{arm}/result.json").exists():continue
        model=read(root/f"models/{arm}/result.json");a.aligned=model["aligned"];restore(a,root/f"models/{arm}/adapter.pt");a.detail=False
        seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0]
        pred,noise,trace=sample(d,a,packet(data),music,seeds,trace=True,selected=cfg["times"]);calls+=1
        expected=eval_arrays if arm=="R1" else np.load(root/folder/"arrays.npz")
        assert np.array_equal(pred.cpu().numpy(),expected[f"train_{arm}_correct_0_normalized"])
        states=trace_arrays(trace);np.savez_compressed(out/f"actual_{arm}.npz",final=pred.cpu().numpy(),**states)
        for t in cfg["times"]:
            a.detail=True;a.set_reference(**packet(data));ts=torch.full((4,),t,device="cuda",dtype=torch.long)
            xt=torch.tensor(states["xt"][states["times"].tolist().index(t)],device="cuda")
            with torch.no_grad():current=d.model(xt,music,None,ts,cond_drop_prob=0)
            assert np.array_equal(current.cpu().numpy(),states[f"p_{t}_cond"])
            for layer,z in a.details.items():
                for key in ("weighted_v","residual","gate"):
                    if z[key] is not None:arrays[f"{arm}_{t}_{layer}_{key}"]=z[key][0,64:160]
                g=z["gate"]
                records.append(dict(arm=arm,t=t,case=92,layer=int(layer),local_time_is_explicit_mapping=a.aligned,
                    global_gate_mean=None if g is None else float(g[0,64:160,0].mean()),
                    global_gate_max_step=None if g is None else float(np.abs(np.diff(g[0,64:160,0])).max()),
                    weighted_v_step_p95=float(np.percentile(np.linalg.norm(np.diff(z["weighted_v"][0,64:160],axis=0),axis=-1),95)),
                    residual_step_p95=float(np.percentile(np.linalg.norm(np.diff(z["residual"][0,64:160],axis=0),axis=-1),95)),
                    correct_prediction_exact=True,scope="ONE_FIXED_TRAIN_SOURCE_ON_ITS_ACTUAL_CORRECT_REVERSE_TRAJECTORY; DESCRIPTIVE_GATE_READOUT"))
            a.clear()
        a.detail=False
    a.close();np.savez_compressed(out/"arrays.npz",**arrays)
    result=dict(records=records,sampler_calls=calls,generated_trajectories=4*calls,optimizer_steps=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=info,files={p.name:sha(p) for p in out.glob("*.npz")}))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in out.glob("*.npz"):same_arrays(p,Path(replay)/p.name)
    print("R30_GATE_PROBE",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
