"""Actual audio clocks and body22 Lodge targets, with disjoint adaptation music groups."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from dunhuang.io import checked,write,sha
from dunhuang.data.sources import ref
from dunhuang.data.conversion import decode
from dunhuang.contracts import matrices
from dunhuang.research.branch_r22.lodge import load
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.context_r21.data import same_audio_pcm

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();parent=Path(cfg["paired_parent"])
    receipt=read("reports/r22_release/r22_receipt.json")
    for name in ("data/result.json","data/protocol.json"):assert sha(parent/name)==receipt["evidence"][name]
    pop=read(parent/"data/result.json");pc=read("src/dunhuang/research/branch_r22/config.json")
    eligibility=[];available=[]
    for row in pop["records"]:
        source=np.load(checked(row["left"]["source_native"]),mmap_mode="r")
        valid=len(source)>=256
        eligibility.append(dict(case=row["case"],role=row["role"],source_group=row["left"]["source_group"],native_frames=len(source),status="ELIGIBLE" if valid else "EXCLUDED_SHORT_NATIVE_RECORDING"))
        if valid:available.append(row)
    diffusion,norm,skeleton,fk,extract,info,manifest=load(pc);records=[]
    for row in available:
        l=row["left"];full=np.load(checked(l["source_native"]));original,_=decode(checked(l["raw_motion"]),"aistpp");assert np.array_equal(full,original)
        a=min(l["start_frame"],len(full)-256);assert a>=0
        native=full[a:a+256];assert native.shape==(256,151)
        clock=read(checked(l["clock"]));assert clock["status"]=="VERIFIED_PAIR" and clock["calibration"]["verified"]
        assert clock["raw_motion"]["sha256"]==l["raw_motion"]["sha256"] and clock["raw_audio"]["sha256"]==l["audio"]["sha256"]
        wave,sr=sf.read(checked(l["audio"]),dtype="float32",always_2d=True)
        begin=round((clock["calibration"]["audio_offset_seconds"]+a/30)*sr);end=begin+round(256/30*sr)
        assert 0<=begin<end<=len(wave)
        wav=out/f'{row["case"]}.wav';sf.write(wav,wave[begin:end],sr,subtype="FLOAT")
        music,_=extract(str(wav),skip_completed=False);music=music[:256].astype(np.float32);assert music.shape==(256,35)
        R=matrices(native[:,7:].reshape(256,24,6))[:,:22]
        raw=np.concatenate((np.zeros((256,4),np.float32),native[:,4:7],R[...,:2,:].reshape(256,132)),1).astype(np.float32)
        raw[:,[4,6]]-=raw[:1,[4,6]]
        with torch.no_grad():
            points=fk(torch.tensor(raw[None],device="cuda"),skeleton)[0,:,:22].cpu().numpy()
            raw[:,5]+=-1.2-float(np.quantile(points[:,[7,8,10,11],1],.02))
            tensor=torch.tensor(raw[None],device="cuda");target=norm.normalize(tensor.clone());back=norm.unnormalize(target.clone())
            source_points=fk(tensor,skeleton)[0,:,:22].cpu().numpy()
        path=out/f'{row["case"]}.npz';np.savez_compressed(path,raw139=raw,target=target[0].cpu().numpy(),music=music,source22=source_points,native=native)
        records.append(dict(case=row["case"],role=row["role"],source_group=l["source_group"],recording_uid=l["recording_uid"],
            raw_motion=l["raw_motion"],source_native=l["source_native"],audio=l["audio"],clock=l["clock"],source_start=a,prior_start=l["start_frame"],
            audio_start_sample=begin,audio_end_sample=end,sample_rate=sr,arrays=path.name,crop_audio=wav.name,
            root_roundtrip_rmse_m=float(((back-tensor)[:,:,4:7].square().mean()).sqrt()),
            body_roundtrip_rmse=float(((back-tensor)[:,:,7:].square().mean()).sqrt()),
            target_is_official_normalizer_clipped=True,Root_supervised_as_truth=False,C4_supervised=False))
        if replay:same_arrays(path,Path(replay)/path.name);assert same_audio_pcm(wav,Path(replay)/wav.name)
        print("R25_PAIRED_DATA",row["case"],flush=True)
    train=[r for r in records if r["role"]=="train"];dev=[r for r in records if r["role"]=="dev_exposed"]
    for key in ("source_group","recording_uid"):
        assert not {r[key] for r in train}&{r[key] for r in dev}
    for key in ("raw_motion","audio"):
        assert not {r[key]["sha256"] for r in train}&{r[key]["sha256"] for r in dev}
    assert len({r["source_group"] for r in train})==cfg["required_train_groups"] and len({r["source_group"] for r in dev})==cfg["required_dev_groups"]
    paths=[Path("assets/datasets/finedance"),Path("/home/disk/lsm/storage/danceRAG/assets/datasets/finedance")];fd=[]
    for base in paths:
        waves=list(base.rglob("*.wav"));pointers=sum(p.stat().st_size<1024 and p.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1") for p in waves)
        fd.append(dict(root=str(base),wav_files=len(waves),LFS_pointers=pointers,real_audio=len(waves)-pointers))
    result=dict(records=records,eligibility_ledger=eligibility,train_cases=[r["case"] for r in train],dev_cases=[r["case"] for r in dev],train_groups=sorted({r["source_group"] for r in train}),dev_groups=sorted({r["source_group"] for r in dev}),
        target_frames=256,music_dim=35,FineDance_audio_audit=fd,FineDance_admitted=False,paired_clock_verified=True,
        split="Fixed R22 adaptation music groups; recording/motion/audio disjoint; historical DEV and pretraining exposure not independent",
        source_clock_adjustment="Shift backward within same recording only if original225 crop lacks31 additional true frames; no padding",
        optimizer_steps=0,human_reviewed=False)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,r22_receipt_sha256=sha("reports/r22_release/r22_receipt.json"),files={p.name:sha(p) for p in out.iterdir() if p.suffix in (".npz",".wav",".npy")}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
