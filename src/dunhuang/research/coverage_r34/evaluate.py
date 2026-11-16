"""Frozen Dyn, untrained full input, and matched continuation/coverage on actual DDIM."""
import argparse
from .common import *
from .data import packet
from .metrics import measure,coverage_gate

def run(cfg,root,out,replay=None,preview=False):
    root=Path(root);out=fresh(out);forbid_optimizers();rt=Runtime(cfg);initial={k:v.detach().clone() for k,v in rt.c.change.state_dict().items()}
    a=dict(np.load(root/"data/arrays.npz"));_,_,data,z=population_data();conditions=[(m,e) for m in range(2) for e in (-1,0,1)]
    music=torch.tensor(np.stack([z["music"][m] for m,e in conditions]),device="cuda");arrays={};records=[];noise=[];retention=[];calls=0
    old=np.load(Path(cfg["r33"])/"evaluation/arrays.npz")
    cases=read("src/dunhuang/research/content_r32/config.json")["holdout_cases"];rows=[data[c] for c in cases]
    general_music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda");seeds=read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0]
    baseline=np.load("outputs/geometry_r31_02/evaluation/arrays.npz")
    for arm in ("Frozen","FrozenFull","Extra","Coverage"):
        if arm.startswith("Frozen"):rt.c.change.load_state_dict(initial)
        else:restore(rt,root/f"models/{arm}/adapter.pt")
        for mode in (() if preview and arm!="Frozen" else ("correct","none")):
            ref=None if mode=="none" else source([core_body(r) for r in rows],"aist",uids=[f"aist_{c}" for c in cases])
            p,nr,_=sample(rt,general_music,source_packet=ref,seeds=seeds);calls+=1;noise.append(nr)
            assert np.array_equal(p.cpu().numpy(),baseline[f"holdout_CanonicalIO_{mode}_0_normalized"])
            arrays[f"general_{arm}_{mode}"]=p.cpu().numpy();retention.append(dict(arm=arm,mode=mode,cases=cases,exact_R31=True))
        full=arm in ("FrozenFull","Coverage")
        for mode in (("correct",) if preview else ("correct","wrong_clock","swapped")):
            pred,nr,_=sample(rt,music,source_packet=packet(a,conditions,full,mode),common_seed=24041);calls+=1;noise.append(nr)
            assert np.array_equal(pred[[0,3]].cpu().numpy(),old["train_Dyn_correct_normalized"][[0,3]])
            if arm=="Frozen":assert np.array_equal(pred.cpu().numpy(),old[f"train_Dyn_{mode}_normalized"])
            if arm=="FrozenFull" and mode=="correct":assert np.array_equal(pred.cpu().numpy(),a["Coverage_normalized"])
            raw=rt.n.unnormalize(pred.clone())
            with torch.no_grad():points=rt.f(raw,rt.s)[:,:,:22]
            raw=raw.cpu().numpy();points=points.cpu().numpy();stem=f"{arm}_{mode}"
            arrays[stem+"_normalized"]=pred.cpu().numpy();arrays[stem+"_139"]=raw;arrays[stem+"_22"]=points
            for i,(m,e) in enumerate(conditions):
                records.append(dict(arm=arm,mode=mode,music=m,event=e,reference_frames=256 if full else 96,
                    errors=[measure(raw[i],points[i],body,p) for body,p in zip(a["body"],a["local_points"])]))
            print("R34_SAMPLE",arm,mode,flush=True)
        rt.check()
    rt.close()
    gate=coverage_gate(records,cfg)
    result=dict(records=records,gate=gate,retention=retention,noise=noise,NoRef_exact=True,general_exact=True,optimizer_steps=0,sampler_calls=calls,generated_trajectories=32 if preview else 104,preview_only=preview,
        extra_recordings_used_for_selection=False,full_native_status="NOT_EVALUABLE_22_OF24",operational_qualified=False,whole_song_generations=0)
    assert calls==(6 if preview else 20)
    np.savez_compressed(out/"arrays.npz",**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(out/"arrays.npz"),source_data_sha256=sha(root/"data/arrays.npz")))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
    print("R34_EVAL_GATE",gate,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");p.add_argument("--preview-only",action="store_true");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from,a.preview_only)
