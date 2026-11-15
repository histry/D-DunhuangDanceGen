"""12 frozen-model ChangE × Dunhuang-music outputs; identical native soft sampler."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.branch_r22.lodge import load
from dunhuang.research.branch_r22.local_contract import common22,require_native24
from dunhuang.research.context_r21.data import same_audio_pcm
from .compiler import compile_events,angular_errors,frequency_readout
from .noise import common_noise

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers()
    prior=Path(cfg["r8_root"]);seal=read("reports/r8_receipt.json");pub=read("reports/r8_summary.json")
    assert sha("reports/r8_summary.json")==seal["summary_sha256"]
    for key in ("prepared/protocol.json","prepared/pairs.json","prepared/music.json"):
        assert sha(prior/key)==pub["evidence"][key]
    pair=read(prior/"prepared/pairs.json")[cfg["pair"]];T=pair["frames"]
    plans=[read(checked(pair["plans"][label])) for label in ("A","B")];events=[p["events"][0] for p in plans]
    sources=[np.load(checked(e["reference"])) for e in events]
    values,mask,phases=compile_events(events,sources,T,cfg)
    distinction=angular_errors(values[0],values[1],list(range(12,20)))[cfg["core_start"]:cfg["core_start"]+T].mean()
    assert distinction>=cfg["min_upper_source_distance_rad"],"Fixed pair not distinguishable; no automatic replacement"
    pc=read("src/dunhuang/research/branch_r22/config.json")
    diffusion,norm,skeleton,fk,extract,info,manifest=load(pc)
    music=[];music_bindings=[];old_music=read(prior/"prepared/music.json")
    for k in cfg["music_cases"]:
        row=next(x for x in old_music if x["case"]==k);audio=checked(row["full_original_audio"])
        wave,sr=sf.read(audio,dtype="float32",always_2d=True)
        start=round(cfg["audio_start_seconds"]*sr);end=start+round(256/30*sr);assert end<=len(wave)
        path=out/("music"+str(k)+".wav");sf.write(path,wave[start:end],sr,subtype="FLOAT")
        features,_=extract(str(path),skip_completed=False);features=features[:256].astype(np.float32)
        assert features.shape==(256,35) and np.isfinite(features).all();music.append(features)
        music_bindings.append(dict(index=k,full_original_audio=row["full_original_audio"],start_sample=start,end_sample=end,sample_rate=sr,
            feature_dim=35,file=path.name,exposure="EXPOSED_DUNHUANG_TRANSFER_DEVELOPMENT; not independent test"))
        if replay:assert same_audio_pcm(path,Path(replay)/path.name)
    source22=[]
    for i in range(2):
        x=torch.tensor(values[i:i+1],device="cuda")
        with torch.no_grad():
            points=fk(x,skeleton)[0,:,:22].cpu().numpy()
            values[i,:,5]+=-1.2-float(np.quantile(points[:,[7,8,10,11],1],.02))
            source22.append(fk(torch.tensor(values[i:i+1],device="cuda"),skeleton)[0,:,:22].cpu().numpy())
    raw=torch.tensor(values,device="cuda");normalized=norm.normalize(raw.clone());roundtrip=norm.unnormalize(normalized.clone()).cpu().numpy()
    a=cfg["core_start"];b=a+T;core=np.zeros(256,bool);core[a:b]=True;cue=mask[:,7:].all(-1).astype(bool);unprompted=core&~cue
    assert unprompted.sum()>T//2
    normalization=[]
    for i in range(2):
        angles=angular_errors(roundtrip[i],values[i])
        affine=raw[i]*norm.scaler.scale_.to(raw.device)+norm.scaler.min_.to(raw.device)
        normalization.append(dict(event=events[i]["event_uid"],cue_mean_pose_rad=float(angles[cue].mean()),core_mean_pose_rad=float(angles[core].mean()),
            cue_rot6_clipped_fraction=float((abs(affine[torch.tensor(cue,device=raw.device),7:])>1).float().mean()),source_root_observed=False))
    distinct=angular_errors(values[0],values[1]);records=[];receipts=[];arrays=dict(source139=values,source22=np.stack(source22),
        reference_normalized=normalized.cpu().numpy(),reference_roundtrip139=roundtrip,cue_mask=mask,music=np.stack(music),unprompted_core=unprompted)
    conditions=[(m,event) for m in range(2) for event in (-1,0,1)]
    batch_music=torch.tensor(np.stack([music[m] for m,e in conditions]),device="cuda")
    masks=torch.tensor(np.stack([np.zeros_like(mask) if e<0 else mask for m,e in conditions]),device="cuda")
    prompts=torch.stack([torch.zeros_like(normalized[0]) if e<0 else normalized[e] for m,e in conditions])
    prompts=prompts*masks
    for seed in cfg["seeds"]:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        with common_noise(seed,batch=6,device="cuda") as receipt:
            sample=diffusion.inpaint_soft_ddim((6,256,139),batch_music,None,constraint=dict(value=prompts,mask=masks))
        assert receipt["draws"]==50
        motion=norm.unnormalize(sample.detach().clone());points=fk(motion,skeleton)[:,:,:22].detach().cpu().numpy();motion=motion.cpu().numpy()
        arrays[str(seed)+"_139"]=motion;arrays[str(seed)+"_22"]=points;arrays[str(seed)+"_normalized"]=sample.cpu().numpy()
        receipts.append(dict(seed=seed,**receipt))
        for i,(m,e) in enumerate(conditions):
            errors=[angular_errors(motion[i],values[j]) for j in range(2)]
            upper=[angular_errors(motion[i],values[j],list(range(12,20))) for j in range(2)]
            row=dict(seed=seed,index=i,music=m,event=e,event_uid=None if e<0 else events[e]["event_uid"],
                source_pose_unprompted_rad=[float(v[unprompted].mean()) for v in errors],
                source_pose_cue_rad=[float(v[cue].mean()) for v in errors],source_upper_unprompted_rad=[float(v[unprompted].mean()) for v in upper],
                **common22(points[i]),frequency=frequency_readout(points[i]),root_observed=False,C4_observed=False)
            records.append(row)
        print("R24_LODGE_SEED",seed,"six conditions complete",flush=True)
    controls=[]
    for seed in cfg["seeds"]:
        for m in range(2):
            baseline=next(r for r in records if (r["seed"],r["music"],r["event"])==(seed,m,-1));tests=[]
            for e in range(2):
                r=next(r for r in records if (r["seed"],r["music"],r["event"])==(seed,m,e))
                source=r["source_pose_unprompted_rad"];upper=r["source_upper_unprompted_rad"]
                tests.append(dict(event=e,closer_to_assigned=source[e]<source[1-e],upper_closer_to_assigned=upper[e]<upper[1-e],
                    improves_over_no_reference=source[e]<=(1-cfg["relative_retention_gain"])*baseline["source_pose_unprompted_rad"][e],
                    common22_jerk_retained=r["jerk_p95_mps3"]<=cfg["max_jerk_ratio"]*baseline["jerk_p95_mps3"]))
            passed=all(t["closer_to_assigned"] and t["upper_closer_to_assigned"] and t["improves_over_no_reference"] and t["common22_jerk_retained"] for t in tests)
            controls.append(dict(seed=seed,music=m,tests=tests,bidirectional_retention_pass=passed))
    music_response=[]
    for seed in cfg["seeds"]:
        x=arrays[str(seed)+"_139"]
        for e in (-1,0,1):
            i=conditions.index((0,e));j=conditions.index((1,e))
            music_response.append(dict(seed=seed,event=e,pose_change_rad=float(angular_errors(x[i],x[j])[core].mean()),root_change_rmse_m=float(np.sqrt(np.mean((x[i,:,4:7]-x[j,:,4:7])**2))),
                perceptual_music_adaptation_proven=False))
    path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    result=dict(records=records,controls=controls,music_response=music_response,normalizer_roundtrip=normalization,
        source_distinction_mean_rad=float(distinct[core].mean()),source_upper_distinction_mean_rad=float(distinct[core][:,12:20].mean()),
        common_cue_frames=int(cue.sum()),unprompted_core_frames=int(unprompted.sum()),core_interval=[a,b],
        selected_events=[dict(event_uid=e["event_uid"],source_uid=e["source_uid"],source_interval=e["source_interval"],reference=e["reference"],time_warp=e["time_warp"],phases=p,
            source_rotation_joints=list(range(22)),source_root_translation_observed=False,semantic_event_truth="UNKNOWN") for e,p in zip(events,phases)],
        music=music_bindings,checkpoint=info,noise_receipts=receipts,generated_trajectories=12,batched_sampler_calls=2,
        all_conditions_same_soft_sampler=True,all_conditions_same_guidance_ramp=True,common_random_numbers_all_conditions=True,
        training_updates=0,full_native_status="NOT_EVALUABLE",operational_qualified=False,whole_song_generations=0,
        qualification=dict(bidirectional_retention_groups_passed=sum(r["bidirectional_retention_pass"] for r in controls),groups=4,
            adapter_training_started=False,reason="Zero-training mechanism first; no unpaired warped-source music targets or post-result budget selection"))
    if replay:
        old=read(Path(replay)/"result.json");assert result==old;same_arrays(path,Path(replay)/"arrays.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),r8_receipt_sha256=sha("reports/r8_receipt.json"),
        r8_pairs=ref(prior/"prepared/pairs.json"),plans=pair["plans"],checkpoint=info,arrays=ref(path),mask_root_contacts=False,
        selection="Fixed historical pair0 female_FeiTian / male_pipa_1, source-only; no output-driven replacement"))
    print("R24_COMPLETE",result["qualification"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
