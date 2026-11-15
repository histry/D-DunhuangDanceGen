"""One real256 crop per distinct recording; frozen DataV3 roles, no Native filtering."""
import argparse,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import soundfile as sf
import torch
from dunhuang.io import checked,sha,write
from dunhuang.data.conversion import decode
from dunhuang.contracts import matrices
from dunhuang.research.branch_r22.lodge import load
from dunhuang.research.source_capacity.common import read,fresh,commit,same_arrays,forbid_optimizers
from dunhuang.research.context_r21.data import same_audio_pcm
from .geometry import fk22,relations
from .reference import events

def choose(rows,dev,seed):
    distinct={}
    for row in rows:
        if row["role"]=="train":distinct.setdefault(row["recording_uid"],row)
    eligible={};ledger=[]
    for row in distinct.values():
        frames=len(np.load(checked(row["source_native"]),mmap_mode="r"))
        ok=frames>=256
        ledger.append(dict(recording_uid=row["recording_uid"],source_group=row["source_group"],frames=frames,eligible=ok))
        if ok:eligible.setdefault(row["source_group"],[]).append(row)
    key=lambda text:hashlib.sha256((str(seed)+text).encode()).hexdigest()
    groups=sorted(set(eligible)-set(dev),key=key)
    if len(groups)!=24:raise ValueError("Registered24 Train music groups required")
    chosen=[]
    for role,gs,n in (("train",groups,4),("dev_exposed",[g for g in dev if g in eligible],2)):
        for group in gs:
            candidates=sorted(eligible[group],key=lambda r:key(r["recording_uid"]))
            if len(candidates)<n:raise ValueError("Not enough independent recordings")
            chosen.extend((role,r) for r in candidates[:n])
    return chosen,ledger

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers()
    source=Path(cfg["preflight"]);assert sha(source/"rows.json")==cfg["rows_sha256"]
    assert sha(source/"protocol.json")==cfg["preflight_sha256"]
    dev=read("outputs/branch_r22_03/data/result.json")["dev_groups"]
    selected,ledger=choose(read(source/"rows.json"),dev,cfg["seed"])
    d,n,s,f,extract,info,_=load(read("src/dunhuang/research/branch_r22/config.json"))
    records=[];errors=[]
    for case,(role,row) in enumerate(selected):
        full=np.load(checked(row["source_native"]));original,_=decode(checked(row["raw_motion"]),"aistpp")
        assert np.array_equal(full,original)
        a=min(row["start_frame"],len(full)-256);native=full[a:a+256];assert native.shape==(256,151)
        clock=read(checked(row["clock"]));cal=clock["calibration"]
        assert clock["status"]=="VERIFIED_PAIR" and cal["verified"]
        assert clock["raw_motion"]["sha256"]==row["raw_motion"]["sha256"] and clock["raw_audio"]["sha256"]==row["audio"]["sha256"]
        wave,sr=sf.read(checked(row["audio"]),dtype="float32",always_2d=True)
        begin=round((cal["audio_offset_seconds"]+a/30)*sr);end=begin+round(256/30*sr)
        assert 0<=begin<end<=len(wave)
        wav=out/f"{case}.wav";sf.write(wav,wave[begin:end],sr,subtype="FLOAT")
        music,_=extract(str(wav),skip_completed=False);music=music[:256].astype(np.float32);assert music.shape==(256,35)
        R=matrices(native[:,7:].reshape(256,24,6))[:,:22]
        raw=np.concatenate((np.zeros((256,4),np.float32),native[:,4:7],R[...,:2,:].reshape(256,132)),1).astype(np.float32)
        raw[:,[4,6]]-=raw[:1,[4,6]]
        with torch.no_grad():
            source_points=f(torch.tensor(raw[None],device="cuda"),s)[0,:,:22]
            shift=-1.2-float(torch.quantile(source_points[:,[7,8,10,11],1].flatten(),.02));raw[:,5]+=shift
            tensor=torch.tensor(raw[None],device="cuda");target=n.normalize(tensor.clone());back=n.unnormalize(target.clone())
            points=f(tensor,s)[:,:,:22];round_points=f(back,s)[:,:,:22]
            direct=fk22(back,s);fk_error=float((direct-round_points).abs().max());assert fk_error<2e-5
            gt=relations(points);reachable=relations(round_points)
            distortion=(reachable-gt).norm(dim=-1);valid=distortion<=cfg["geometry_support_m"]
        arrays=dict(raw139=raw,target=target[0].cpu().numpy(),music=music,source22=points[0].cpu().numpy(),native=native,
            geometry_target=reachable[0].cpu().numpy(),geometry_valid=valid[0].cpu().numpy())
        path=out/f"{case}.npz";np.savez_compressed(path,**arrays)
        records.append(dict(case=case,role=role,source_group=row["source_group"],recording_uid=row["recording_uid"],
            raw_motion=row["raw_motion"],source_native=row["source_native"],audio=row["audio"],clock=row["clock"],
            source_start=a,prior_start=row["start_frame"],crop_audio=wav.name,arrays=path.name,root_y_shift=shift,
            root_roundtrip_rmse_m=float((back[:,:,4:7]-tensor[:,:,4:7]).square().mean().sqrt()),
            body_roundtrip_rmse=float((back[:,:,7:]-tensor[:,:,7:]).square().mean().sqrt()),
            geometry_support_fraction=float(valid.float().mean()),geometry_distortion_max_m=float(distortion.max()),
            matrix_fk_official_max_m=fk_error,phase_events=events(raw[64:160,7:]),
            geometry_target="Official normalizer roundtrip FK, masked where raw-source distortion>0.05m",
            C4_observed=False,Root_supervised_as_truth=False))
        if replay:same_arrays(path,Path(replay)/path.name);assert same_audio_pcm(wav,Path(replay)/wav.name)
        if case%10==0:print("R26_DATA",case,flush=True)
    train=[r for r in records if r["role"]=="train"];devrows=[r for r in records if r["role"]=="dev_exposed"]
    for field in ["recording_uid","source_group"]:
        assert not {r[field] for r in train}&{r[field] for r in devrows}
    for field in ["raw_motion","source_native","audio"]:
        assert not {r[field]["sha256"] for r in train}&{r[field]["sha256"] for r in devrows}
    assert len(train)==96 and len(devrows)==6
    train_probe=[next(r["case"] for r in train if r["source_group"]==g) for g in sorted({r["source_group"] for r in train})[:4]]
    dev_probe=[next(r["case"] for r in devrows if r["source_group"]==g) for g in sorted({r["source_group"] for r in devrows})]
    # Unknown C4 placeholders remain in x_t; describe their distribution, never label contacts.
    targets=np.stack([np.load(out/r["arrays"])["target"] for r in train])
    free=np.load(Path(cfg["r25"])/"samples/arrays.npz")["paired_baseline_none_normalized"]
    audit=[]
    rng=np.random.default_rng(cfg["seed"]);noise=rng.standard_normal(targets.shape).astype(np.float32)
    for step in [19,519,999]:
        alpha=float(d.alphas_cumprod[step].cpu())
        def stats(x,epsilon):
            qt=np.sqrt(alpha)*x+np.sqrt(1-alpha)*epsilon
            return {k:dict(mean=float(qt[:,:,a:b].mean()),std=float(qt[:,:,a:b].std())) for k,a,b in [("C4",0,4),("Root",4,7)]}
        audit.append(dict(t=step,paired_placeholder=stats(targets,noise),free_sample_proxy=stats(free,noise[:len(free)])))
    official=Path("inputs/lodge_r22/official/dld/data/render_joints/smplfk.py")
    model=official.parents[2]/"models/architectures/model.py"
    ground=dict(official_fk_sha256=sha(official),official_model_sha256=sha(model),
        source_lines=[l.strip() for l in official.read_text().splitlines() if "floor_height=-1.2" in l or "root_pos[:, :]" in l],
        fixed_transform="Native matrix-columns to Lodge rows; Y-up; XZ origin; source-only rigid height shift as R25",
        generated_motion_translated=False,ground_truth_calibrated=False,near_ground_empty_is_not_zero_slip=True,
        missing_native24_joints=[22,23],native24_is_not_detailed_fingers=True)
    result=dict(records=records,eligibility_ledger=ledger,train_cases=[r["case"] for r in train],dev_cases=[r["case"] for r in devrows],
        train_probe=train_probe,dev_probe=dev_probe,train_groups=sorted({r["source_group"] for r in train}),dev_groups=sorted({r["source_group"] for r in devrows}),
        state_input_audit=audit,ground_contract=ground,FineDance_admitted=False,split="Global DataV3 Train only; preserve R22 adaptation DEV groups; historical exposure/performer/choreography/pretraining independence not claimed",
        optimizer_steps=0,operational_qualified=False,whole_song_generations=0)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,files={p.name:sha(p) for p in out.iterdir() if p.suffix in (".npz",".wav")}))
    print("R26_DATA_READY",len(train),len(devrows),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
