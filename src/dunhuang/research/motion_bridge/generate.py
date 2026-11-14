"""Fixed exposed DEV readout after all representation/training choices are sealed.

Only 'correct' drives each complete rollout. Wrong/removal controls are shared
x_t forward interventions and are explicitly NOT counterfactual full rollouts.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.boundary_context.common import load_model,digest
from dunhuang.research.state_bridge.common import cases
from dunhuang.research.condition_dependency.codec import restore
from dunhuang.sampling import constrained_ddim
from .model import MotionBridge,Injection,music_tokens,soft_control
from .objective import terms

def run(cfg,training,out,replay=None):
    out=fresh(out);forbid_optimizers()
    oldcfg=read("src/dunhuang/research/state_bridge/config.json");items=cases(oldcfg)
    trainroot=Path(training);source={arm:read(trainroot/arm/"result.json") for arm in cfg["arms"]}
    assert all(v["optimizer_steps"]==cfg["training_steps"] for v in source.values())
    assert source["B2"]["parameters"]==source["B3"]["parameters"]
    assert read(trainroot/"B2/protocol.json")["selected_site"]==read(trainroot/"B3/protocol.json")["selected_site"]
    q=items[0]["request"];model,diffusion,norm=load_model(q);before=digest(model)
    scale=torch.tensor(norm.scaler.scale_,device="cuda");bias=torch.tensor(norm.scaler.min_,device="cuda")
    write(out/"protocol.json",dict(code_commit=commit(),config=cfg,models={a:v["checkpoint"] for a,v in source.items()},
        previous=ref(Path(oldcfg["parent"])/"generation/protocol.json"),data_roles="4 exposed Val + 4 exposed R8 cases; fixed post-training readout",
        controls="same-state soft interventions with correct hard context and correct-driven latent; not independently generated controls"))
    bindings=[];records=[]
    for arm in ("B0","B3","B2"):
        adapter=None;hook=None
        if arm!="B0":
            payload=torch.load(checked(source[arm]["checkpoint"]),map_location="cpu",weights_only=True)
            assert payload["base_sha256"]==q["checkpoint"]["sha256"] and payload["schema"]==cfg["schema"]
            adapter=MotionBridge(payload["state"]["scale"],payload["state"]["bias"],payload["state"]["inv_rms"],arm,cfg["perturbation_rms"]).cuda().eval()
            adapter.load_state_dict(payload["state"],strict=True)
            for p in adapter.parameters():p.requires_grad_(False)
            hook=Injection(model,payload["site"])
        for i,item in enumerate(items):
            folder=out/(item["name"]+"_"+arm);folder.mkdir()
            request=item["request"];old=item["original"]
            z=np.load(checked(request["context"]));value=torch.tensor(z["values"],device="cuda")*scale+bias
            hard=torch.tensor(z["mask"],device="cuda")
            wrong=torch.tensor(np.load(checked(item["wrong_context"]))["values"],device="cuda")*scale+bias
            music=torch.tensor(np.load(checked(request["features"])),device="cuda")
            tokens=music_tokens(model,music);rows=[];predictions=[];latent=[]
            original=diffusion.model_predictions
            def predict(x,cond,t,*args,**kwargs):
                assert kwargs.get("weight",2)==2 and kwargs.get("clip_x_start",True)
                if adapter is None:base=original(x,cond,t,*args,**kwargs)
                else:
                    delta=adapter(value,hard,x,tokens,t);p=hook.predict(x,cond,t,delta)
                    # The sampler uses x0 only; preserve GaussianDiffusion's noise contract too.
                    base=(diffusion.predict_noise_from_start(x,t,p),p)
                controls=["correct"]
                if adapter is not None and i<4:
                    controls+=["removed","removed_left","removed_right","wrong_left","wrong_right"]
                allpred=[];controlscores=[]
                rng=torch.cuda.get_rng_state();xr=x.detach().clone()
                for control in controls:
                    if control=="correct":p=base[1]
                    else:
                        v,k=soft_control(value,hard,wrong,control)
                        p=hook.predict(x,cond,t,adapter(v,k,x,tokens,t))
                    scores=terms(p,value,hard,scale,bias)
                    controlscores.append(dict(control=control,**{n:float(v) for n,v in scores.items()}))
                    allpred.append(p[0,50:100].cpu().numpy().copy())
                assert torch.equal(x,xr) and torch.equal(rng,torch.cuda.get_rng_state())
                predictions.append(np.stack(allpred));latent.append(x[0,50:100].cpu().numpy().copy())
                rows.append(dict(t=int(t[0]),controls=controlscores,rng_unchanged=True))
                return base
            diffusion.model_predictions=predict
            torch.manual_seed(request["seed"]);torch.cuda.manual_seed_all(request["seed"])
            with torch.no_grad():sample,receipt=constrained_ddim(diffusion,music,value,hard)
            diffusion.model_predictions=original
            normalized=sample.cpu().numpy()
            if arm=="B0":
                # Exact normalized match validates new readout against the frozen B0 sampler.
                path=Path(oldcfg["comparison"])/"generation"/(item["name"]+"_zero")/"normalized.npy"
                assert np.array_equal(normalized,np.load(path))
            raw=((normalized[0]-bias.cpu().numpy())/scale.cpu().numpy()).astype(np.float32)
            native=restore(MotionAdapter.to_native(raw),np.asarray(request["root_origin_native"],np.float32))
            edited=old.copy();edited[60:90]=native[60:90]
            assert np.array_equal(edited[:60],old[:60]) and np.array_equal(edited[90:],old[90:])
            assert receipt["noise_sha256"]==read(item["parent_folder"]/"generation.json")["sampler"]["noise_sha256"]
            np.save(folder/"normalized.npy",normalized);np.save(folder/"edited_native.npy",edited);np.save(folder/"raw_last.npy",predictions[-1][0])
            np.savez_compressed(folder/"trace.npz",prediction=np.stack(predictions),latent=np.stack(latent))
            write(folder/"trace.json",rows);write(folder/"sampler.json",receipt)
            records.append(dict(case=item["name"],arm=arm,seed=request["seed"],noise_sha256=receipt["noise_sha256"],same_hard_context=True,protected_exact=True))
            bindings.append(dict(case=item["name"],arm=arm,native=ref(folder/"edited_native.npy"),raw_last=ref(folder/"raw_last.npy"),trace=ref(folder/"trace.npz"),trace_info=ref(folder/"trace.json")))
            if replay:
                from dunhuang.research.source_tasks.common import same_arrays
                other=Path(replay)/folder.name
                for f in ("normalized.npy","edited_native.npy","raw_last.npy"):assert np.array_equal(np.load(folder/f),np.load(other/f))
                same_arrays(folder/"trace.npz",other/"trace.npz")
                assert rows==read(other/"trace.json") and receipt==read(other/"sampler.json")
            print("R12_GENERATE",item["name"],arm,flush=True)
        if hook:hook.close()
    assert digest(model)==before
    result=dict(records=records,bindings=bindings,backbone_digest=before,backbone_unchanged=True,optimizer_steps=0,new_neural_generations=len(records),
        new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result)
    if replay:assert records==read(Path(replay)/"result.json")["records"]

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--training",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.training,a.out,a.replay_from)
