"""Matched q proxies and genuine sampler paths, with same-state reference interventions."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from .trace import sample,trace_arrays
from dunhuang.research.reference_r26.geometry import supported_losses

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();pop,records,data=sources(cfg)
    cases=[r["case"] for r in records];d,n,s,f,e,info,adapter=runtime(cfg)
    music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda")
    target=torch.tensor(np.stack([data[c]["target"] for c in cases]),device="cuda")
    references={mode:packet([data[c]["raw139"][64:160,7:] for c in (cases if mode=="correct" else cases[::-1])]) for mode in ("correct","wrong")}
    arrays={};receipts=[];readouts=[];traces={}
    # One uninstrumented execution tests that observation changes neither sampler nor RNG.
    plain,pnoise,_=sample(d,adapter,None,music,cfg["diagnostic_seeds"])
    calls=1
    for arm,mode in [("baseline","none")]+[(a,m) for a in ("C1","C2") for m in ("correct","wrong")]:
        if arm!="baseline":
            path=Path(cfg["r26"])/f"models/{arm}/adapter.pt"
            assert sha(path)==read(Path(cfg["r26"])/"artifacts_manifest.json")["files"][str(path.relative_to(Path(cfg["r26"])))]["sha256"]
            adapter.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["state"],strict=True)
        value,noise,trace=sample(d,adapter,None if mode=="none" else references[mode],music,cfg["diagnostic_seeds"],trace=True,target=target)
        calls+=1;stem=arm+"_"+mode;traces[stem]=trace_arrays(trace)
        traces[stem]["final"]=value.cpu().numpy()
        np.savez_compressed(out/(stem+".npz"),**traces[stem])
        receipts.append(dict(arm=arm,mode=mode,noise=noise,states_scope="ACTUAL_REVERSE_XT",state_steps=len(trace["times"])))
        if arm=="baseline":
            assert torch.equal(value,plain) and noise==pnoise;baseline=value.detach()
        print("R27_FROZEN_TRACE",stem,flush=True)
    baseline_trace=traces["baseline_none"]
    for arm in ("C1","C2"):
        adapter.load_state_dict(torch.load(Path(cfg["r26"])/f"models/{arm}/adapter.pt",map_location="cpu",weights_only=True)["state"],strict=True)
        own=traces[arm+"_correct"]
        for time in cfg["diagnostic_times"]:
            index=baseline_trace["times"].tolist().index(time)
            ts=torch.full((2,),time,device="cuda",dtype=torch.long)
            epsilon=torch.tensor(baseline_trace["xt"][0],device="cuda")
            anchors=dict(q_target=d.q_sample(target,ts,epsilon),q_model_terminal=d.q_sample(baseline,ts,epsilon),
                actual_baseline=torch.tensor(baseline_trace["xt"][index],device="cuda"),actual_own=torch.tensor(own["xt"][index],device="cuda"))
            for anchor,x in anchors.items():
                state={name:dict(core_mean=float(x[:,64:160,a:b].mean()),core_std=float(x[:,64:160,a:b].std())) for name,a,b in [("C4",0,4),("Root",4,7),("body",7,139)]}
                for mode in ("correct","wrong"):
                    adapter.set_reference(**references[mode])
                    with torch.no_grad():
                        unc=d.model(x,music,None,ts,cond_drop_prob=1);cond=d.model(x,music,None,ts,cond_drop_prob=0)
                        weight=float(own[f"p_{time}_weight"]);combined=unc+(cond-unc)*weight
                    for branch,pred in [("unc",unc),("cond",cond),("cfg",combined),("clipped",combined.clamp(-1,1))]:
                        decoded=n.unnormalize(pred.detach().clone())
                        with torch.no_grad():
                            official=f(decoded,s)[:,:,:22];direct=fk22(decoded,s);parity=float((official-direct).abs().max())
                        values=decoded_metrics(pred.detach(),target,n,s,f)
                        clipped_fraction={name:float((pred[:,64:160,a:b].abs()>1).float().mean()) for name,a,b in [("C4",0,4),("Root",4,7),("body",7,139)]}
                        grad_input=pred.detach().clone().requires_grad_(True)
                        g=relations(fk22(n.unnormalize(grad_input),s))[:,64:160]
                        truth=torch.tensor(np.stack([data[c]["geometry_target"][64:160] for c in cases]),device="cuda")
                        valid=torch.tensor(np.stack([data[c]["geometry_valid"][64:160] for c in cases]),device="cuda")
                        position,velocity=supported_losses(g,truth,valid)
                        loss=position+.05*velocity
                        gradient=torch.autograd.grad(loss,grad_input)[0]
                        saturated=grad_input.abs()>1
                        readouts.append(dict(arm=arm,t=time,anchor=anchor,reference=mode,branch=branch,state=state,errors=values,saturation=clipped_fraction,
                            prediction_fk_official_max_m=parity,geometry_gradient_finite=bool(torch.isfinite(gradient).all()),
                            saturated_gradient_max=float(gradient[saturated].abs().max()) if saturated.any() else None,
                            gradient_norm=float(gradient.norm())))
                    adapter.clear()
    result=dict(receipts=receipts,matched_readouts=readouts,recorded_times=cfg["diagnostic_times"],
        anchors=dict(q_target="FORWARD_Q_PROXY_PAIRED_TARGET",q_model_terminal="FORWARD_Q_PROXY_MATCHED_BASELINE_TERMINAL",actual_baseline="ACTUAL_REVERSE_XT",actual_own="ACTUAL_REVERSE_XT"),
        uninstrumented_sampler_exact=True,sampler_calls=calls,generated_trajectories=calls*2,optimizer_steps=0,full_native_status="NOT_EVALUABLE",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,files={p.name:sha(p) for p in out.glob("*.npz")},cases=cases,checkpoint=info))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in out.glob("*.npz"):same_arrays(p,Path(replay)/p.name)
    adapter.close();print("R27_FROZEN_DIAGNOSTIC_DONE",len(readouts),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
