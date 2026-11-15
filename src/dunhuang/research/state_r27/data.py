"""Coupled model-generated139D targets; no real contact labels and no channel mixing."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.data.sources import ref
from .common import *
from .trace import sample,trace_arrays

def coverage(cfg,pop):
    root=Path(cfg["r26"]);manifest=read(root/"artifacts_manifest.json")["files"];rows=[]
    for row in pop["records"]:
        name="data/"+row["arrays"];assert sha(root/name)==manifest[name]["sha256"]
        with np.load(root/name) as z:v=z["geometry_valid"]
        core=v[64:160];velocity=core[1:]&core[:-1]
        rows.append(dict(case=row["case"],role=row["role"],source_group=row["source_group"],
            full_position=v.mean(0).tolist(),core_position=core.mean(0).tolist(),core_velocity=velocity.mean(0).tolist()))
    return dict(relations=["left_wrist_shoulder","right_wrist_shoulder","wrist_pair","left_elbow_shoulder","right_elbow_shoulder"],core_frames=96,core_velocity_templates=95,records=rows)

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();pop,records,data=sources(cfg)
    d,n,s,f,e,info,adapter=runtime(cfg)
    cases=[r["case"] for r in records for draw in range(2)]
    music=torch.tensor(np.stack([data[c]["music"] for c in cases]),device="cuda")
    value,noise,trace=sample(d,adapter,None,music,cfg["teacher_seeds"],trace=True)
    assert bool((value.abs()<=1).all())
    raw=n.unnormalize(value.clone());points=f(raw,s)[:,:,:22]
    pairs=[]
    for i in (0,2):
        m=metrics(raw[i].cpu().numpy(),points[i].cpu().numpy(),raw[i+1].cpu().numpy(),points[i+1].cpu().numpy())
        delta=float((value[i,64:160,7:]-value[i+1,64:160,7:]).square().mean())
        assert delta>=cfg["teacher_diversity_mse"] and m["body22_rad"]>=cfg["teacher_diversity_angle"]
        pairs.append(dict(music=i//2,normalized_body_mse=delta,physical=m))
    target=out/"teacher.npz";np.savez_compressed(target,target=value.cpu().numpy(),raw139=raw.cpu().numpy(),points22=points.cpu().numpy(),music=music.cpu().numpy())
    traced=out/"teacher_trace.npz";np.savez_compressed(traced,final=value.cpu().numpy(),**trace_arrays(trace))
    sources_meta=[dict(case=r["case"],recording_uid=r["recording_uid"],source_group=r["source_group"],audio=ref(Path(cfg["r26"])/"data"/r["crop_audio"])) for r in records]
    result=dict(teachers=[dict(id=i,music=i//2,teacher_seed=cfg["teacher_seeds"][i],target_scope="MODEL_GENERATED_FULL139",C4_scope="MODEL_PREDICTION_NOT_CONTACT_TRUTH",channels_coupled=True) for i in range(4)],
        sources=sources_meta,noise=noise,diversity=pairs,coverage=coverage(cfg,pop),sampler_calls=1,generated_trajectories=4,optimizer_steps=0,
        teacher_eval_seed_disjoint=not set(cfg["teacher_seeds"])&set(sum(cfg["eval_noise_seeds"],[])),full_native_status="NOT_EVALUABLE",operational_qualified=False)
    assert result["teacher_eval_seed_disjoint"]
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,files={p.name:sha(p) for p in (target,traced)}))
    if replay:
        assert result==read(Path(replay)/"result.json");same_arrays(target,Path(replay)/target.name);same_arrays(traced,Path(replay)/traced.name)
    adapter.close();print("R27_TEACHERS_READY",pairs,flush=True)

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
