"""Fixed final checkpoint, no validation selection; masked actual Root/rotation targets."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write,sha
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from .adapter import BoundaryContextAdapter
from .common import load_model,digest,Data,masks,fixed_rows

def loss(pred,target,unknown,a,b,scaler):
    from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle
    base=(pred-target)[unknown].square().mean()
    fused=torch.where(unknown,pred,target);raw=fused/scaler[0]-scaler[1]/scaler[0];truth=target/scaler[0]-scaler[1]/scaler[0]
    inds=torch.tensor(sorted(set(range(a-3,a+3))|set(range(b-3,b+3))),device=pred.device)
    root1=torch.diff(raw[:,:,4:7],dim=1)*30;gt1=torch.diff(truth[:,:,4:7],dim=1)*30
    root2=torch.diff(root1,dim=1)*30;gt2=torch.diff(gt1,dim=1)*30
    R=rotation_6d_to_matrix(raw[:,:,7:].reshape(-1,150,24,6));G=rotation_6d_to_matrix(truth[:,:,7:].reshape(-1,150,24,6))
    w=matrix_to_axis_angle(R[:,:-1].transpose(-1,-2)@R[:,1:])*30
    v=matrix_to_axis_angle(G[:,:-1].transpose(-1,-2)@G[:,1:])*30
    accel=torch.diff(w,dim=1)*30;ga=torch.diff(v,dim=1)*30
    return base+.01*(root1[:,inds]-gt1[:,inds]).square().mean()+.0001*(root2[:,inds]-gt2[:,inds]).square().mean()+.001*(w[:,inds]-v[:,inds]).square().mean()+.00001*(accel[:,inds]-ga[:,inds]).square().mean()

def run(cfg,preflight,out,replay=None):
    out=fresh(out);pf=read(Path(preflight)/"result.json")
    if not pf["bridge_training_data_ready"]:raise ValueError("B2 real masked target/music data not ready")
    pp=read(Path(preflight)/"protocol.json");assets=read(checked(pp["assets"]));rows=read(Path(preflight)/"rows.json")
    fit=[r for r in rows if r["role"]=="train"];val=fixed_rows(rows,"val",cfg["validation_windows"],cfg["seed"])
    assert set(r["source_group"] for r in fit).isdisjoint(r["source_group"] for r in val)
    model,diffusion,norm=load_model(assets);before=digest(model);export=np.load(checked(assets["scaler"]));data=Data(export)
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"]);adapter=BoundaryContextAdapter().cuda()
    state={}
    def hook(module,args,value):return value+adapter(state["value"],state["known"],value) if state["enabled"] else value
    handle=model.cond_encoder.register_forward_hook(hook)
    scaler=(torch.tensor(export["scale"],device="cuda"),torch.tensor(export["bias"],device="cuda"))
    protocol=dict(code_commit=commit(),config=cfg,preflight=ref(Path(preflight)/"protocol.json"),rows=ref(Path(preflight)/"rows.json"),
        training_rows=len(fit),validation_rows=val,final_checkpoint_only=True,contact_truth=False,known_context_and_unknown_target_support=True,full_window_scaler_support_claimed=False)
    write(out/"protocol.json",protocol)
    def evaluate(control):
        values=[];adapter.eval()
        with torch.no_grad(),torch.random.fork_rng(devices=[0]):
            torch.manual_seed(cfg["seed"]+71);torch.cuda.manual_seed_all(cfg["seed"]+71)
            for start in range(0,len(val),2):
                target,music=data.batch(val[start:start+2]);known,unknown=masks(target,60,90);value=target.clone()
                if control=="removed":known=torch.zeros_like(known)
                elif control=="wrong":value=torch.roll(value,1,0) if len(value)>1 else value.flip(1)
                state.update(value=value,known=known,enabled=control!="zero")
                for t in cfg["validation_timesteps"]:
                    ts=torch.full((len(target),),t,device="cuda",dtype=torch.long)
                    pred=model(diffusion.q_sample(target,ts),music,ts,cond_drop_prob=0.)
                    values.append(float((pred-target)[unknown].square().mean()))
        return float(np.mean(values))
    if replay:
        saved=torch.load(checked(read(Path(replay)/"result.json")["checkpoint"]),map_location="cpu",weights_only=True)
        adapter.load_state_dict(saved["state"],strict=True);updates=0;history=[]
    else:
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg["learning_rate"]);rng=np.random.default_rng(cfg["seed"]);history=[]
        for update in range(cfg["steps"]):
            subset=[fit[i] for i in rng.integers(0,len(fit),cfg["batch"])]
            target,music=data.batch(subset);a=int(rng.integers(60,67));b=int(rng.integers(84,91))
            observed=None if rng.random()<.25 else rng.random(24)<.8;known,unknown=masks(target,a,b,observed)
            state.update(value=target,known=known,enabled=True);adapter.train()
            t=torch.randint(50,950,(len(target),),device="cuda")
            pred=model(diffusion.q_sample(target,t),music,t,cond_drop_prob=0.)
            objective=loss(pred,target,unknown,a,b,scaler)
            if not torch.isfinite(objective):raise ValueError("Nonfinite B2 masked boundary objective")
            optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);optimizer.step()
            history.append(float(objective));assert all(p.grad is None for p in model.parameters())
            if (update+1)%25==0:print("R9_B2_TRAIN",update+1,history[-1],flush=True)
        updates=cfg["steps"]
    adapter.eval();scores={control:evaluate(control) for control in ("zero","observed","removed","wrong")}
    assert digest(model)==before
    if replay:
        old=read(Path(replay)/"result.json");assert scores==old["validation"]
        checkpoint=old["checkpoint"]
    else:
        torch.save(dict(schema="boundary_context_adapter_v1",state=adapter.state_dict(),base_sha256=assets["checkpoint"]["sha256"],code_commit=commit(),config=cfg,
            trained_steps=updates,condition_schema="real_actual_root_rotation_masked_context",operational_qualified=False),out/"adapter.pt")
        checkpoint=ref(out/"adapter.pt")
    result=dict(validation=scores,history=history,optimizer_steps=updates,checkpoint=checkpoint,backbone_before=before,backbone_after=digest(model),
        parameters=sum(p.numel() for p in adapter.parameters()),new_whole_song_generations=0,operational_qualified=False,contact_supervision=False)
    write(out/"result.json",result);handle.remove();print("R9_B2_VALIDATION",scores,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--preflight",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.preflight,a.out,a.replay_from)
