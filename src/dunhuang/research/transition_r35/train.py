"""Train only Context on continuous paired tasks; partial ChangE protects the old Core."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
from collections import Counter
from dunhuang.research.coverage_r34.common import *
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.content_r32.partial import partial_loss
from dunhuang.research.control_r33.objective import body_dynamics
from dunhuang.research.reference_r26.geometry import fk22
from dunhuang.research.coverage_r34.context_train import checkpoint,restore_context
from dunhuang.research.plan_view_r34.common import ref

def schedule(cfg,cases):
    if len(cases)!=16 or len(set(cases))!=16:raise ValueError("Frozen16 source population")
    rng=np.random.default_rng(cfg["seed"]);order=np.concatenate([rng.permutation(cases) for _ in range(100)])
    rows=[];k=0
    for i in range(cfg["train_steps"]):
        if i%3==2:rows.append(dict(step=i,task="change",t=cfg["times"][(i//3)%4],seed=cfg["seed"]+i))
        else:
            ids=[int(c) for c in order[4*k:4*k+4]]
            rows.append(dict(step=i,task="paired",cases=ids,t=cfg["times"][k%4],seed=cfg["seed"]+i,
                state="actual" if k%2==0 else "q_sample",mask="both" if k%2==0 else ("no_left","no_right","partial")[(k//2)%3],
                gap=[64,160] if k%2==0 else cfg["gap_intervals"][(k//2)%len(cfg["gap_intervals"])],source="none" if k%2==0 else "general"))
            k+=1
    assert k==400
    return rows

def targets(rt,data,cases):
    rows=[data[c] for c in cases]
    y=rt.n.normalize(torch.tensor(np.stack([r["model_raw"] for r in rows]),device="cuda"))
    music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
    return rows,y,music

def cg(rt,z,a):
    bodies=z["source139"][[0,1,0,1],64:160,7:]
    points=z["source22"][[0,1,0,1],64:160];points=points-points[:,:,:1]
    music=torch.tensor(z["music"][[0,0,1,1]],device="cuda")
    packet=source(list(bodies),"change",uids=["female_FeiTian","male_pipa_1"]*2)
    full=torch.full((4,256,139),float("nan"),device="cuda");full[:,:,7:]=torch.tensor(a["body"][[0,1,0,1]],device="cuda")
    cp=context(full,[64,160],partial_body=True)
    return torch.tensor(bodies,device="cuda"),torch.tensor(points,device="cuda"),music,packet,cp

def run(cfg,root,out,replay=None):
    out=fresh(out)
    if replay:forbid_optimizers()
    rt=Runtime(cfg,context_training=True);initial=digest(rt.c.context_branch)
    assert sum(p.numel() for p in rt.c.context_branch.parameters() if p.requires_grad)==268160
    assert all(not p.requires_grad for m in (rt.d.model,rt.io,rt.c.general,rt.c.change) for p in m.parameters())
    _,_,data,z=population_data();a=dict(np.load(Path(cfg["r34"])/"data/arrays.npz"))
    parent=read(Path(cfg["r33"])/"models/Dyn/result.json")
    cases=sorted({c for r in parent["history"] if r["task"]=="paired" for c in r["cases"]})
    rows=schedule(cfg,cases);counts=Counter(c for r in rows if r["task"]=="paired" for c in r["cases"])
    assert set(counts.values())=={100}
    body,points,cmusic,cpkt,cctx=cg(rt,z,a);cal=read(Path(cfg["r33"])/"calibration/result.json")
    noise=[];calls=0;trajectories=0
    def refresh(path):
        nonlocal calls,trajectories
        stored={};times=None
        for n,ids in enumerate([cases[i:i+4] for i in range(0,16,4)]):
            _,y,m=targets(rt,data,ids)
            pred,nr,tr=sample(rt,m,context_packet=context(y,[64,160]),seeds=[cfg["seed"]+c for c in ids],trace=True)
            idx=[tr["times"].index(t) for t in cfg["times"]];xt=np.stack([tr["states"][i] for i in idx])
            stored[f"paired{n}"]=xt;stored[f"paired{n}_output"]=pred.cpu().numpy()
            noise.append(nr);calls+=1;trajectories+=4
        pred,nr,tr=sample(rt,cmusic,source_packet=cpkt,context_packet=cctx,common_seed=24041,trace=True)
        idx=[tr["times"].index(t) for t in cfg["times"]];stored["change"]=np.stack([tr["states"][i] for i in idx])
        stored["change_output"]=pred.cpu().numpy();stored["times"]=np.array(cfg["times"]);stored["cases"]=np.array(cases)
        noise.append(nr);calls+=1;trajectories+=4
        np.savez_compressed(path,**stored);return stored
    states=refresh(out/"initial_states.npz")
    # Initial zero Context must exactly preserve pinned R33 main output.
    prior=np.load(Path(cfg["r33"])/"evaluation/arrays.npz")["train_Dyn_correct_normalized"][[1,2,4,5]]
    assert np.array_equal(states["change_output"],prior)
    def loss(row):
        torch.manual_seed(row["seed"]);torch.cuda.manual_seed_all(row["seed"]);rt.c.clear_all()
        if row["task"]=="paired":
            selected,y,music=targets(rt,data,row["cases"]);ts=torch.full((4,),row["t"],device="cuda",dtype=torch.long)
            if row["state"]=="actual":
                ti=cfg["times"].index(row["t"])
                table={c:states[f"paired{j//4}"][ti,j%4] for j,c in enumerate(cases)}
                xt=torch.tensor(np.stack([table[c] for c in row["cases"]]),device="cuda")
            else:xt=rt.d.q_sample(y,ts,torch.randn_like(y))
            if row["source"]=="general":rt.c.set_source(source([core_body(r) for r in selected],"aist",uids=[f"aist_{c}" for c in row["cases"]]))
            rt.c.set_context(context(y,row["gap"],row["mask"],row["seed"]))
            pred=rt.d.model(xt,music,None,ts,cond_drop_prob=.2);rt.c.clear_all()
            parts=dict(body=(pred[:,:,7:]-y[:,:,7:]).square().mean(),root=(pred[:,:,4:7]-y[:,:,4:7]).square().mean(),
                velocity=torch.diff(pred[:,:,7:]-y[:,:,7:],dim=1).square().mean())
            return parts["body"]+.1*parts["root"]+.1*parts["velocity"],{k:float(v.detach()) for k,v in parts.items()}
        xt=torch.tensor(states["change"][cfg["times"].index(row["t"])],device="cuda");ts=torch.full((4,),row["t"],device="cuda",dtype=torch.long)
        rt.c.set_source(cpkt);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state()
        with torch.no_grad():teacher=rt.d.model(xt,cmusic,None,ts,cond_drop_prob=.2)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state(gpu);rt.c.set_context(cctx)
        pred=rt.d.model(xt,cmusic,None,ts,cond_drop_prob=.2);rt.c.clear_all()
        w=read("src/dunhuang/research/content_r32/config.json")["partial_weights"]
        value,parts=partial_loss(pred,body,points,teacher,rt.n,rt.s,w)
        fk=fk22(rt.n.unnormalize(pred.clone()),rt.s)[:,64:160]
        dyn,_,_=body_dynamics(fk,points,torch.ones_like(points[...,0],dtype=torch.bool),cal["scales"])
        return value+cal["weight"]*dyn,dict({k:float(v.detach()) for k,v in parts.items()},dynamic=float(dyn.detach()))
    def readout(row):
        value,parts=loss(row);assert torch.isfinite(value)
        torch.use_deterministic_algorithms(True)
        try:value.backward()
        finally:torch.use_deterministic_algorithms(False)
        grad=float(sum(p.grad.square().sum() for p in rt.c.context_branch.parameters() if p.grad is not None).sqrt())
        assert np.isfinite(grad) and grad>0
        return value,parts,grad
    obs=[];actual=[];checks=[];history=[]
    if not replay:
        (out/"snapshots").mkdir();checkpoint(rt,out/"initial.pt",cfg,0);opt=torch.optim.AdamW(rt.c.context_branch.parameters(),lr=cfg["lr"])
        for row in rows:
            i=row["step"]
            if i==300:checkpoint(rt,out/"weights_300.pt",cfg,i);states=refresh(out/"refresh_300.npz")
            watch=i in cfg["snapshots"]
            if watch:checkpoint(rt,out/f"snapshots/{i}_pre.pt",cfg,i);before=vector(rt.c.context_branch).clone()
            opt.zero_grad(set_to_none=True);value,parts,g=readout(row)
            torch.nn.utils.clip_grad_norm_(rt.c.context_branch.parameters(),1.,error_if_nonfinite=True);opt.step()
            history.append(dict(row,loss=float(value.detach()),parts=parts))
            if watch:
                checkpoint(rt,out/f"snapshots/{i}_post.pt",cfg,i+1);obs.append(dict(step=i,loss=float(value.detach()),parts=parts,gradients={"all":g},update_norm=float((vector(rt.c.context_branch)-before).norm())))
            if (i+1)%100==0:print("R35_CONTEXT",i+1,float(value.detach()),flush=True)
        checkpoint(rt,out/"context.pt",cfg,600);steps=600
    else:
        old=read(Path(replay)/"result.json");assert initial==old["initial_digest"]
        for name,h in old["files"].items():assert sha(Path(replay)/name)==h
        same_arrays(out/"initial_states.npz",Path(replay)/"initial_states.npz")
        for item in old["observations"]:
            i=item["step"]
            states=dict(np.load(Path(replay)/("refresh_300.npz" if i>=300 else "initial_states.npz")))
            restore_context(rt,Path(replay)/f"snapshots/{i}_pre.pt");before=vector(rt.c.context_branch).clone();rt.c.context_branch.zero_grad(set_to_none=True)
            value,parts,g=readout(rows[i]);restore_context(rt,Path(replay)/f"snapshots/{i}_post.pt")
            now=dict(step=i,loss=float(value.detach()),parts=parts,gradients={"all":g},update_norm=float((vector(rt.c.context_branch)-before).norm()))
            checks.append(readout_check(now,item));actual.append(now)
        restore_context(rt,Path(replay)/"weights_300.pt");states=refresh(out/"refresh_300.npz");same_arrays(out/"refresh_300.npz",Path(replay)/"refresh_300.npz")
        restore_context(rt,Path(replay)/"context.pt");steps=0;history=old["history"];obs=old["observations"]
    final=digest(rt.c.context_branch);rt.check();rt.close()
    if replay:assert final==old["final_digest"] and noise==old["noise"]
    result=dict(code_commit=commit(),optimizer_steps=steps,initial_digest=initial,final_digest=final,frozen=rt.frozen,
        observations=obs,actual_readouts=actual,replay_checks=checks,history=history,paired_presentations=dict(counts),
        context_trainable_parameters=268160,source_updated=False,io_updated=False,backbone_updated=False,
        sampler_calls=calls,generated_trajectories=trajectories,noise=noise,cache="ACTUAL_NOMINAL_BOTH_NO_SOURCE_PAIRED_AND_BODY_ONLY_CG; REFRESH300; NOT_EVERY_STEP",
        hard_observation_override=False,unknown_Root_C4_formed_x0=False,additional_ablation_calls=0,
        files=old["files"] if replay else {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.suffix in (".pt",".npz")},
        whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result)
    print("R35_CONTEXT_DONE",steps,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read("src/dunhuang/research/transition_r35/config.json"),a.root,a.out,a.replay_from)
