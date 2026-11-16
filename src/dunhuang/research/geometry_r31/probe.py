"""No-update task-gradient diagnosis on q_t and each model's real reverse states."""
import argparse
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import sources
from .interface import runtime
from dunhuang.research.time_r29.common import packet
from dunhuang.research.reference_r26.geometry import fk22
from dunhuang.research.lodge_adapter_r25.model import PARTS
from dunhuang.research.contract_r28.train import loss_terms
from dunhuang.research.continuity_r30.metrics import measure
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.trace import row_noise

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pop,ledger,_=sources(cfg)
    data={r["case"]:r["data"] for r in pop if r["domain"]=="aist" and r["role"]=="train"}
    rows=[data[c] for c in cfg["train_cases"]];music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
    source=torch.tensor(np.stack([r["model_points"] for r in rows]),device="cuda");predictions={};vectors={};records=[]
    for arm in ("LegacyIO","CanonicalIO"):
        d,n,s,f,e,info,b=runtime(cfg,arm=="CanonicalIO",root/f"models/{arm}/adapter.pt")
        trace=np.load(root/f"evaluation/actual_{arm}.npz")
        with torch.no_grad():target=n.normalize(torch.tensor(np.stack([r["model_raw"] for r in rows]),device="cuda"))
        names=[name for name,p in b.named_parameters()];params=list(b.parameters())
        offsets=np.r_[0,np.cumsum([p.numel() for p in params])]
        groups={"all":list(range(len(params))),"encoder":[i for i,name in enumerate(names) if name.startswith(("reference.parts","reference.input","reference.temporal","reference.norm"))],
                "reference_output":[i for i,name in enumerate(names) if name.startswith("reference.out")],
                "action_input":[i for i,name in enumerate(names) if name.startswith("io.input_delta")],
                "action_output":[i for i,name in enumerate(names) if name.startswith("io.output_delta")]}
        for kind in ("q","actual"):
            for t in cfg["times"]:
                ts=torch.full((4,),t,device="cuda",dtype=torch.long)
                if kind=="q":
                    with row_noise(read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0],device="cuda"):epsilon=torch.randn_like(target)
                    xt=d.q_sample(target,ts,epsilon)
                else:xt=torch.tensor(trace["xt"][trace["times"].tolist().index(t)],device="cuda")
                b.clear()
                with torch.no_grad():
                    baseline=d.model(xt,music,None,ts,cond_drop_prob=0)
                    b.set_reference(**packet(rows));no_grad=d.model(xt,music,None,ts,cond_drop_prob=0)
                if kind=="actual":assert np.array_equal(no_grad.cpu().numpy(),trace[f"p_{t}_cond"])
                b.clear();b.set_reference(**packet(rows));pred=d.model(xt,music,None,ts,cond_drop_prob=0)
                forward_difference=float((pred.detach()-no_grad).abs().max());assert forward_difference<=1e-4
                reconstruct,_=loss_terms(pred,target,baseline)
                raw=n.unnormalize(pred.clone());points=fk22(raw,s)
                with torch.no_grad():official=f(raw.detach(),s)[:,:,:22]
                fk_difference=float((points.detach()-official).abs().max());assert fk_difference<=2e-5
                losses={"reconstruction":reconstruct}
                for k,beta in ((1,.5),(2,5.)):
                    diff=lambda x:torch.diff(x,n=k,dim=1)*30**k
                    losses[f"Root_D{k}"]=F.smooth_l1_loss(diff(points[:,64:160,:1]),diff(source[:,64:160,:1]),beta=beta)
                    p=diff(points[:,64:160]-points[:,64:160,:1]);q=diff(source[:,64:160]-source[:,64:160,:1])
                    losses[f"body_D{k}"]=torch.stack([F.smooth_l1_loss(p[:,:,g],q[:,:,g],beta=beta) for g in PARTS]).mean()
                gradients={}
                for j,(name,loss) in enumerate(losses.items()):
                    torch.use_deterministic_algorithms(True)
                    try:g=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=j<len(losses)-1)
                    finally:torch.use_deterministic_algorithms(False)
                    flat=torch.cat([torch.zeros_like(p).flatten() if x is None else x.flatten() for p,x in zip(params,g)]).detach().cpu().numpy()
                    assert np.isfinite(flat).all()
                    vectors[f"{arm}_{kind}_{t}_{name}"]=flat;gradients[name]=flat
                group_stats={}
                for group,indices in groups.items():
                    v={name:np.concatenate([x[offsets[i]:offsets[i+1]] for i in indices]).astype(np.float64) for name,x in gradients.items()}
                    norms={name:float(np.linalg.norm(x)) for name,x in v.items()};cos={}
                    for name,x in v.items():
                        product=norms[name]*norms["reconstruction"]
                        cos[name]=float(np.dot(x,v["reconstruction"])/product) if product else None
                    group_stats[group]=dict(norms=norms,cosine_with_reconstruction=cos,
                        norm_ratio_to_reconstruction={name:value/norms["reconstruction"] if norms["reconstruction"] else None for name,value in norms.items()})
                predictions[f"{arm}_{kind}_{t}"]=pred.detach().cpu().numpy()
                with torch.no_grad():
                    geometry=[measure(raw[i].detach().cpu().numpy(),official[i].cpu().numpy(),row["model_raw"],row["model_points"]) for i,row in enumerate(rows)]
                records.append(dict(arm=arm,kind=kind,t=t,cases=cfg["train_cases"],losses={name:float(x.detach()) for name,x in losses.items()},
                    groups=group_stats,forward_gradient_backend_max_difference=forward_difference,FK_max_difference_m=fk_difference,
                    saturated_Core_fraction={name:float((pred[:,64:160,sl].detach().abs()>1).float().mean()) for name,sl in (("Root",slice(4,7)),("body",slice(7,139)))},
                    geometry=geometry,scope="CONDITIONAL_NO_MUSIC_DROPOUT_TRAIN_TASK_GRADIENT; NOT_AN_OPTIMIZER_UPDATE"))
                b.clear()
        b.close()
    np.savez_compressed(out/"predictions.npz",**predictions);np.savez_compressed(out/"gradients.npz",**vectors)
    checks=[]
    if replay:
        old=read(Path(replay)/"result.json");assert len(old["records"])==len(records)
        for x,y in zip(records,old["records"]):
            assert {k:v for k,v in x.items() if k!="groups"}=={k:v for k,v in y.items() if k!="groups"}
            for group,a in x["groups"].items():
                expected=y["groups"][group]
                for key in ("norms","cosine_with_reconstruction","norm_ratio_to_reconstruction"):
                    for name,value in a[key].items():
                        ref=expected[key][name]
                        if ref is None:assert value is None
                        else:assert np.isclose(value,ref,rtol=1e-5,atol=1e-5 if key!="norms" else 0.)
        same_arrays(out/"predictions.npz",Path(replay)/"predictions.npz")
        saved=np.load(Path(replay)/"gradients.npz")
        for name,x in vectors.items():
            y=saved[name];relative=float(np.linalg.norm(x.astype(np.float64)-y)/max(np.linalg.norm(y.astype(np.float64)),1e-30))
            assert relative<=1e-5;checks.append(dict(name=name,relative_l2_error=relative))
    result=dict(records=old["records"] if replay else records,actual_gradient_readouts=records if replay else [],gradient_replay_checks=checks,
        gradient_vector_relative_l2_tolerance=1e-5,gradient_cosine_atol=1e-5,prediction_arrays_exact=True,
        backward_deterministic_algorithms=True,cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        optimizer_steps=0,sampler_calls=0,generated_trajectories=0,new_physical_training_admitted=False,
        operational_qualified=False,whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),files={p.name:sha(p) for p in out.glob("*.npz")}))
    print("R31_GRADIENT_PROBE",len(records),len(checks),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
