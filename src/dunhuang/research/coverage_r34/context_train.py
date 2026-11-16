"""Independent context training, only after frozen coverage admission; no gap truth in conditions."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
from .common import *
from .data import packet
from .context_contract import context
from .objective import region_loss

def checkpoint(rt,path,cfg,step):
    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in rt.c.context_branch.state_dict().items()},config=cfg,step=step,frozen=rt.frozen),path)
def restore_context(rt,path):
    x=torch.load(path,map_location="cpu",weights_only=True);assert x["frozen"]==rt.frozen;rt.c.context_branch.load_state_dict(x["state"],strict=True)
def cg_values(a,conditions):
    y=torch.full((len(conditions),256,139),float("nan"),device="cuda")
    y[:,:,7:]=torch.tensor(np.stack([a["body"][max(e,0)] for m,e in conditions]),device="cuda")
    cp=context(y,[64,160],partial_body=True)
    for i,(m,e) in enumerate(conditions):
        if e<0:cp.valid[i]=False;cp.values[i]=float("nan")
    return cp

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out)
    if replay:forbid_optimizers()
    gate=read(root/"evaluation/result.json")["gate"]
    if cfg.get("defer_context_training_for_visual_review") or not gate["context_training_admitted"]:
        write(out/"result.json",dict(admitted=False,reason="VISUAL_FIRST_CONTEXT_NOT_STARTED" if cfg.get("defer_context_training_for_visual_review") else "CONTINUOUS_COVERAGE_GATE_FAILED",coverage_gate=gate,optimizer_steps=0,sampler_calls=0,generated_trajectories=0,
            operational_qualified=False,whole_song_generations=0));print("R34_CONTEXT_BLOCKED",flush=True);return
    rt=Runtime(cfg,root/"models/Coverage/adapter.pt",context_training=True)
    _,_,data,z=population_data();a=dict(np.load(root/"data/arrays.npz"));cal=read(Path(cfg["r33"])/"calibration/result.json")
    initial=digest(rt.c.context_branch);source_fixed=digest(rt.c.change)
    conditions=[(m,e) for m in range(2) for e in (-1,0,1)];select=[1,2,4,5]
    arrays={};noise=[];calls=0
    def refresh(path):
        music=torch.tensor(np.stack([z["music"][m] for m,e in conditions]),device="cuda")
        pred,nr,tr=sample(rt,music,source_packet=packet(a,conditions,True),context_packet=cg_values(a,conditions),common_seed=24041,trace=True)
        baseline=np.load(root/"evaluation/arrays.npz")["Coverage_correct_normalized"]
        assert np.array_equal(pred[[0,3]].cpu().numpy(),baseline[[0,3]])
        values=trace_arrays(tr);np.savez_compressed(path,**values)
        return dict(xt=values["xt"][:,select],times=values["times"]),nr
    states,nr=refresh(out/"initial_states.npz");calls+=1;noise.append(nr)
    body=torch.tensor(a["body"][[0,1,0,1]],device="cuda");points=torch.tensor(a["local_points"][[0,1,0,1]],device="cuda")
    cmusic=torch.tensor(z["music"][[0,0,1,1]],device="cuda");ref=packet(a,[(0,0),(0,1),(1,0),(1,1)],True);cp=cg_values(a,[(0,0),(0,1),(1,0),(1,1)])
    schedule=[{k:v for k,v in r.items() if k not in ("loss","parts")} for r in read(root/"models/Coverage/result.json")["history"]]
    def loss(row):
        seed=cfg["context_seed"]+row["step"];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rt.c.clear_all()
        if row["task"]=="paired":
            rows=[data[c] for c in row["cases"]];y=rt.n.normalize(torch.tensor(np.stack([r["model_raw"] for r in rows]),device="cuda"));music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
            ts=torch.full((4,),row["t"],device="cuda",dtype=torch.long);xt=rt.d.q_sample(y,ts,torch.randn_like(y))
            gaps=cfg["context_gap_intervals"][(row["step"]//2)%3];mode=("both","no_left","no_right","partial")[(row["step"]//2)%4]
            ctx=context(y,gaps,mode,seed)
            if (row["step"]//2)%4==0:rt.c.set_source(source([core_body(r) for r in rows],"aist",uids=[f"aist_{c}" for c in row["cases"]]))
            rt.c.set_context(ctx);pred=rt.d.model(xt,music,None,ts,cond_drop_prob=.2);rt.c.clear_all()
            parts=dict(body=(pred[:,:,7:]-y[:,:,7:]).square().mean(),root=(pred[:,:,4:7]-y[:,:,4:7]).square().mean(),
                velocity=(torch.diff(pred[:,:,7:]-y[:,:,7:],dim=1)).square().mean())
            return parts["body"]+.1*parts["root"]+.1*parts["velocity"],{k:float(v.detach()) for k,v in parts.items()}
        xt=torch.tensor(states["xt"][states["times"].tolist().index(row["t"])],device="cuda")
        ts=torch.full((4,),row["t"],device="cuda",dtype=torch.long);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        with torch.no_grad():teacher=rt.d.model(xt,cmusic,None,ts,cond_drop_prob=.2)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);rt.c.set_source(ref);rt.c.set_context(cp)
        pred=rt.d.model(xt,cmusic,None,ts,cond_drop_prob=.2);rt.c.clear_all()
        weights=read("src/dunhuang/research/content_r32/config.json")["partial_weights"]
        base,parts,dyn,dp,_,_=region_loss(pred,body,points,teacher,rt.n,rt.s,weights,cal["scales"])
        return base+cal["weight"]*dyn,dict({k:float(v.detach()) for k,v in parts.items()},dynamic=float(dyn.detach()))
    observations=[];actual=[];checks=[];history=[]
    if not replay:
        (out/"snapshots").mkdir();checkpoint(rt,out/"initial.pt",cfg,0);opt=torch.optim.AdamW(rt.c.context_branch.parameters(),lr=cfg["context_learning_rate"])
        for row in schedule:
            i=row["step"]
            if i==300:checkpoint(rt,out/"weights_300.pt",cfg,i);states,nr=refresh(out/"refresh_300.npz");calls+=1;noise.append(nr)
            watch=i in cfg["snapshot_steps"]
            if watch:checkpoint(rt,out/f"snapshots/{i}_pre.pt",cfg,i);before=vector(rt.c.context_branch).clone()
            opt.zero_grad(set_to_none=True);value,parts=loss(row);assert torch.isfinite(value);torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            grad=float(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None).sqrt())
            torch.nn.utils.clip_grad_norm_(rt.c.context_branch.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(row,loss=float(value.detach()),parts=parts))
            if watch:
                checkpoint(rt,out/f"snapshots/{i}_post.pt",cfg,i+1);observations.append(dict(step=i,loss=float(value.detach()),parts=parts,gradients={"all":grad},update_norm=float((vector(rt.c.context_branch)-before).norm())))
            if (i+1)%100==0:print("R34_CONTEXT_TRAIN",i+1,float(value.detach()),flush=True)
        checkpoint(rt,out/"context.pt",cfg,600);steps=600
    else:
        old=read(Path(replay)/"result.json");assert old["initial_digest"]==initial
        for fn,h in old["files"].items():assert sha(Path(replay)/fn)==h
        same_arrays(out/"initial_states.npz",Path(replay)/"initial_states.npz")
        for item in old["observations"]:
            i=item["step"]
            if i>=300:
                v=np.load(Path(replay)/"refresh_300.npz");states=dict(xt=v["xt"][:,select],times=v["times"])
            restore_context(rt,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(rt.c.context_branch).clone();rt.c.context_branch.zero_grad(set_to_none=True)
            value,parts=loss(schedule[i]);torch.use_deterministic_algorithms(True)
            try:value.backward()
            finally:torch.use_deterministic_algorithms(False)
            grad=float(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None).sqrt())
            restore_context(rt,Path(replay)/f"snapshots/{i}_post.pt")
            current=dict(step=i,loss=float(value.detach()),parts=parts,gradients={"all":grad},update_norm=float((vector(rt.c.context_branch)-before).norm()))
            actual.append(current);checks.append(readout_check(current,item))
        restore_context(rt,Path(replay)/"weights_300.pt");states,nr=refresh(out/"refresh_300.npz");calls+=1;noise.append(nr);same_arrays(out/"refresh_300.npz",Path(replay)/"refresh_300.npz")
        restore_context(rt,Path(replay)/"context.pt");steps=0;history=old["history"];observations=old["observations"]
    rt.check();final=digest(rt.c.context_branch);rt.close()
    result=dict(admitted=True,optimizer_steps=steps,initial_digest=initial,final_digest=final,source_fixed_digest=source_fixed,frozen=rt.frozen,
        history=history,observations=observations,actual_readouts=actual,replay_checks=checks,noise=noise,
        sampler_calls=calls,generated_trajectories=6*calls,context_parameters=268160,source_training=False,partial_Root_C4_truth=False,
        no_target_gap_in_context=True,hard_observation_override=False,partial_state_augmentation="NOMINAL_CG_BOTH_BODY_ONLY; RANDOM_GAPS/MASKS_ON_COMPLETE_PAIRED_AIST",
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE_22_OF24")
    if replay:assert final==old["final_digest"]
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,coverage_gate_sha256=sha(root/"evaluation/result.json")))
    print("R34_CONTEXT_DONE",steps,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
