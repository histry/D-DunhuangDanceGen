"""Published preflight: source lineage, R33 equality, actual visited states and finite-update admission."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
from .common import *
from .data import observed,packet
from .objective import forward

def run(cfg,out,replay=None):
    out=fresh(out)
    if replay:forbid_optimizers()
    rt=Runtime(cfg);arrays,records=observed(cfg,rt);_,_,aist,z=population_data()
    conditions=[(m,e) for m in range(2) for e in (-1,0,1)];music=torch.tensor(np.stack([z["music"][m] for m,e in conditions]),device="cuda")
    noises=[];calls=0
    prior=np.load(Path(cfg["r33"])/"evaluation/arrays.npz")
    for coverage in (False,True):
        arm="Coverage" if coverage else "Legacy"
        pred,nr,tr=sample(rt,music,source_packet=packet(arrays,conditions,coverage),common_seed=24041,trace=True)
        if not coverage:assert np.array_equal(pred.cpu().numpy(),prior["train_Dyn_correct_normalized"])
        assert np.array_equal(pred[[0,3]].cpu().numpy(),prior["train_Dyn_correct_normalized"][[0,3]])
        arrays[arm+"_normalized"]=pred.cpu().numpy();values=trace_arrays(tr);np.savez_compressed(out/(arm+"_states.npz"),**values);calls+=1;noises.append(nr)
    cfg32=read("src/dunhuang/research/content_r32/config.json");cases=cfg32["holdout_cases"];rows=[aist[c] for c in cases]
    m=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda");seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0]
    for mode in ("correct","none"):
        ref=None if mode=="none" else source([core_body(r) for r in rows],"aist",uids=[f"aist_{c}" for c in cases])
        pred,nr,_=sample(rt,m,source_packet=ref,seeds=seeds);assert np.array_equal(pred.cpu().numpy(),np.load("outputs/geometry_r31_02/evaluation/arrays.npz")[f"holdout_CanonicalIO_{mode}_0_normalized"])
        arrays["general_"+mode]=pred.cpu().numpy();calls+=1;noises.append(nr)
    # Two discarded finite AdamW updates, using only the registered training observations.
    states=np.load(out/"Coverage_states.npz");sel=[1,2,4,5];source_packet=packet(arrays,[(0,0),(0,1),(1,0),(1,1)],True)
    body=torch.tensor(arrays["body"][[0,1,0,1]],device="cuda");points=torch.tensor(arrays["local_points"][[0,1,0,1]],device="cuda");m=music[sel]
    calibration=read(Path(cfg["r33"])/"calibration/result.json");scales=calibration["scales"];weight=calibration["weight"]
    initial={k:v.detach().clone() for k,v in rt.c.change.state_dict().items()};updates=[];actual=[];checks=[]
    if not replay:(out/"snapshots").mkdir()
    for time in (999,19):
        rt.c.change.load_state_dict(initial);xt=torch.tensor(states["xt"][states["times"].tolist().index(time),sel],device="cuda")
        seed=cfg["seed"]+time
        if not replay:save(rt,out/f"snapshots/{time}_pre.pt",cfg,"preflight",0)
        rt.c.change.zero_grad(set_to_none=True);before=vector(rt.c.change).clone()
        base,parts,dyn,dp,pred,fk,count=forward(rt,xt,m,source_packet,time,seed,body,points,scales);loss=base+weight*dyn
        torch.use_deterministic_algorithms(True)
        try:loss.backward()
        finally:torch.use_deterministic_algorithms(False)
        grad=float(sum(p.grad.square().sum() for p in rt.c.change.parameters() if p.grad is not None).sqrt())
        torch.nn.utils.clip_grad_norm_(rt.c.change.parameters(),1.,error_if_nonfinite=True)
        if not replay:
            opt=torch.optim.AdamW(rt.c.change.parameters(),lr=cfg["learning_rate"]);opt.step();save(rt,out/f"snapshots/{time}_post.pt",cfg,"preflight",1)
        else:restore(rt,Path(replay)/f"snapshots/{time}_post.pt")
        item=dict(time=time,loss=float(loss.detach()),gradients={"all":grad},update_norm=float((vector(rt.c.change)-before).norm()))
        with torch.no_grad():post,pp,pd,_,prediction,_,_=forward(rt,xt,m,source_packet,time,seed,body,points,scales)
        item.update(body_before=float(parts["body"].detach()),body_after=float(pp["body"]),geometry_before=float(parts["local_geometry"].detach()),geometry_after=float(pp["local_geometry"]))
        arrays[f"finite_{time}"]=prediction.cpu().numpy()
        if replay:
            old=read(Path(replay)/"result.json");expected=next(r for r in old["updates"] if r["time"]==time);checks.append(readout_check(item,expected));actual.append(item)
        updates.append(item)
    admitted=all(u["gradients"]["all"]>0 and u["update_norm"]>0 and u["body_after"]<=1.1*u["body_before"] and u["geometry_after"]<=1.1*u["geometry_before"] for u in updates)
    rt.c.change.load_state_dict(initial);rt.check();frozen=rt.frozen;rt.close()
    np.savez_compressed(out/"arrays.npz",**arrays)
    result=dict(records=records,coverage_frames=256,legacy_frames=96,coverage_fraction=1.,general_and_NoRef_exact=True,legacy_R33_exact=True,
        reference_parameters=265328,context_training=False,updates=updates,actual_readouts=actual,replay_checks=checks,training_admitted=admitted,
        frozen=frozen,optimizer_steps=0 if replay else 2,sampler_calls=calls,generated_trajectories=20,noise=noises,
        unknown_Root_C4_formed_x0=False,full_native_status="NOT_EVALUABLE_22_OF24",operational_qualified=False,whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),files={p.name:sha(p) for p in out.glob("*.npz")}))
    if replay:
        old=read(Path(replay)/"result.json");assert updates==old["updates"] and admitted==old["training_admitted"]
        for p in out.glob("*.npz"):same_arrays(p,Path(replay)/p.name)
    print("R34_PREFLIGHT",admitted,calls,20,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
