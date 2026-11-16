"""Actual teacher-visited states, not zero-filled incomplete diffusion targets."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from dunhuang.research.state_r27.common import write,fresh,commit,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.trace import sample,record_sampler,trace_arrays
from dunhuang.research.lodge_adapter_r25.sample import sample as change_sample
from dunhuang.research.lodge_adapter_r25.replay import change_conditions

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();pop,ledger,data,z=population(cfg)
    d,n,s,f,e,info,b=runtime(cfg);frozen=fingerprint(d,b)
    previous=np.load(Path(cfg["r31"])/"evaluation/arrays.npz");arrays={};noise=[];calls=0
    for role in ("train","holdout"):
        cases=cfg["train_cases"] if role=="train" else cfg["holdout_cases"]
        music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda")
        for k,seeds in enumerate(read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"]):
            p,nr,_=sample(d,b,None,music,seeds)
            assert np.array_equal(p.cpu().numpy(),previous[f"{role}_CanonicalIO_none_{k}_normalized"])
            arrays[f"{role}_none_{k}"]=p.cpu().numpy();calls+=1;noise.append(nr)
    old=np.load(Path(cfg["r31"])/"change/arrays.npz");conditions=change_conditions()
    music=torch.tensor(np.stack([z["music"][m] for m,event in conditions]),device="cuda")
    for mode in ("correct","wrong_clock","swapped"):
        events=[1-max(event,0) if mode=="swapped" else max(event,0) for m,event in conditions]
        ref=reference([z["source139"][ev,64:160,7:] for ev in events],drops=[ev<0 for m,ev in conditions])
        ref["mapped_source"]=mapping(6,"cuda",wrong=mode=="wrong_clock")
        with record_sampler(d,selected=cfg["times"]) as trace:
            p,nr=change_sample(d,b,ref,music,24041,"CanonicalIO")
        assert np.array_equal(p.cpu().numpy(),old[f"CanonicalIO_{mode}_normalized"])
        arrays[f"change_{mode}"]=p.cpu().numpy();calls+=1;noise.append(nr)
        if mode=="correct":
            np.savez_compressed(out/"change_states.npz",**trace_arrays(trace))
    teacher=Path(cfg["r31"])/"evaluation/actual_CanonicalIO.npz"
    assert sha(teacher)==read(Path(cfg["r31"])/"artifacts_manifest.json")["files"]["evaluation/actual_CanonicalIO.npz"]["sha256"]
    a=np.load(teacher);np.savez_compressed(out/"aist_states.npz",xt=a["xt"],times=a["times"])
    target=np.stack([z["source139"][i,64:160,7:] for i in (0,1,0,1)])
    assert target.shape==(4,96,132) and np.isfinite(target).all()
    np.savez_compressed(out/"partial_observations.npz",body=target,local_points=z["source22"][[0,1,0,1],64:160]-z["source22"][[0,1,0,1],64:160,:1],music=z["music"][[0,0,1,1]])
    np.savez_compressed(out/"arrays.npz",**arrays)
    assert_frozen(d,b,frozen);b.close()
    result=dict(platform_NoRef_exact=True,initial_ChangE_exact=True,Root_C4_targets_created=False,partial_body_shape=list(target.shape),
        aist_recordings=16,aist_groups=8,change_sources=2,change_music_pairing="USER_SPECIFIED_EXPOSED_UNPAIRED; NOT_GROUND_TRUTH",
        sampler_calls=calls,generated_trajectories=34,optimizer_steps=0,whole_song_generations=0,operational_qualified=False,frozen=frozen)
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in out.glob("*.npz"):same_arrays(p,Path(replay)/p.name)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,files={p.name:sha(p) for p in out.glob("*.npz")},parent_manifest=cfg["r31_manifest_sha256"]))
    print("R32_PREPARE",calls,34,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
