"""Rebuild the historical 28 and prospectively inventory physical mechanism data."""
import argparse,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from dunhuang.io import checked,write,sha
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.context_r21.data import run as rebuild
from dunhuang.research.context_r21.contract import numpy_regions
from dunhuang.research.masked_denoiser.common import assets_and_population
from dunhuang.research.global_clock_r17.contract import pair_candidates
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.masked_denoiser.metrics import decode
from dunhuang.research.task_denoising_r18.evaluate import dynamics
from .engine import parent

def domains(points,left,right):
    hard=np.zeros((225,151),bool);hard[:left,4:]=True;hard[225-right:,4:]=True
    jerk=np.linalg.norm(np.diff(points,n=3,axis=0)*30**3,axis=-1);out={}
    for name,mask in numpy_regions(hard,3).items():
        a=jerk[mask];out[name]=dict(templates=int(mask.sum()),max=float(a.max()),p95=float(np.percentile(a,95)))
    assert sum(x["templates"] for x in out.values())==222
    return out

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pc=parent(cfg)
    rebuild(pc,out/"rebuilt",Path(cfg["parent"])/"data")
    data=read(out/"rebuilt/result.json");pre=read(out/"rebuilt/protocol.json")
    # Freshly rebuilt references, same registered scientific population.
    write(out/"result.json",data);write(out/"protocol.json",dict(pre,config=cfg,source_config=pc))
    from motion.geometry import fk_24_np
    scaler=np.load(checked(pre["assets"]["scaler"]));history=[]
    for row in data["records"]:
        z=np.load(checked(row["arrays"]));x=z["native"];points=fk_24_np(x);physical=read(checked(row["source_physical"]))["absolute"]
        roundtrip=decode(z["encoded"],scaler,x);rp=fk_24_np(roundtrip)
        ds={str(i):domains(points,*pair) for i,pair in enumerate(cfg["contexts"])}
        history.append(dict(case=row["case"],role=row["role"],source_group=row["left"]["source_group"],
            native_ok=physical["ok"],native_reasons=physical["reasons"],native_audit=physical["audit"],
            domains=ds,roundtrip_fk_max_m=float(abs(points-rp).max()),
            roundtrip_jerk_max_delta=float(abs(np.diff(points,n=3,axis=0)*30**3-np.diff(rp,n=3,axis=0)*30**3).max()),
            proxy_floor=row["full_source_floor"],native_floor=physical["audit"]["floor_y_m"],
            immutable_jerk_exclusion={i:dict(lower_bound=d["fixed"]["max"],excluded=d["fixed"]["max"]>cfg["fixed_jerk_limit"]) for i,d in ds.items()}))
    assets,rows,_,_,_=assets_and_population(read("src/dunhuang/research/masked_denoiser/config.json"))
    lookup={(l["recording_uid"],l["start_frame"]):(l,r) for l,r in pair_candidates(rows)}
    eligible=[r for r in data["candidate_ledger"] if r["status"]=="ELIGIBLE"]
    full={};candidates=[]
    (out/"candidate_physical").mkdir()
    for i,item in enumerate(eligible):
        l,r=lookup[item["recording_uid"],item["start_frame"]];key=l["source_native"]["sha256"]
        if key not in full:full[key]=np.load(checked(l["source_native"]))
        x=full[key][l["start_frame"]:l["start_frame"]+225]
        points,physical=audit_native(x);dyn=dynamics(x,points)
        activity=float(np.linalg.norm(np.diff(points,axis=0)*30,axis=-1).mean())
        movement=float(np.mean(dyn["joint_range_p95_rad"][1:22]))
        group=l["source_group"];role="train" if group in data["train_groups"] else "dev_exposed" if group in data["dev_groups"] else "outside_registered_roles"
        status="ELIGIBLE" if physical["absolute"]["ok"] and activity>=cfg["mechanism_activity_min_mps"] and movement>=cfg["mechanism_joint_range_min_rad"] and role!="outside_registered_roles" else "SOURCE_NATIVE_FAIL" if not physical["absolute"]["ok"] else "LOW_ACTIVITY_OR_RANGE" if activity<cfg["mechanism_activity_min_mps"] or movement<cfg["mechanism_joint_range_min_rad"] else "OUTSIDE_REGISTERED_ROLES"
        candidates.append(dict(source_group=group,recording_uid=l["recording_uid"],start_frame=l["start_frame"],
            raw_motion_sha256=l["raw_motion"]["sha256"],source_native_sha256=key,audio_sha256=l["audio"]["sha256"],
            role=role,status=status,native_ok=physical["absolute"]["ok"],native_reasons=physical["absolute"]["reasons"],
            activity_mps=activity,mean_joint_range_rad=movement,source=l))
        write(out/f"candidate_physical/{i}.json",physical)
        if i%50==0:print("R22_SOURCE_AUDIT",i,len(eligible),flush=True)
    selected=[];counts=Counter()
    for item in sorted(candidates,key=lambda r:hashlib.sha256((str(cfg["seed"])+r["raw_motion_sha256"]+str(r["start_frame"])).encode()).hexdigest()):
        key=item["role"],item["source_group"]
        if item["status"]=="ELIGIBLE" and counts[key]<(2 if key[0]=="train" else 1):
            if any(r["recording_uid"]==item["recording_uid"] for r in selected):continue
            selected.append(item);counts[key]+=1
    tr={r["source_group"] for r in selected if r["role"]=="train"};dv={r["source_group"] for r in selected if r["role"]=="dev_exposed"}
    for field in ("raw_motion_sha256","source_native_sha256","audio_sha256"):
        assert not {r[field] for r in selected if r["role"]=="train"}&{r[field] for r in selected if r["role"]=="dev_exposed"},field
    audit=dict(historical=history,candidates=candidates,mechanism_selected=selected,mechanism_train_groups=len(tr),mechanism_dev_groups=len(dv),
        mechanism_coverage_ready=len(tr)>=6 and len(dv)>=3,
        mechanism_used_for_R22_training=False,mechanism_status="PROSPECTIVE_SOURCE_INVENTORY_ONLY",
        native_failure_counts=dict(Counter(n for x in history for n in x["native_reasons"])),
        source_reconstruction="Original observed poses and Root are exact paired targets; no claim that their full Native status is ground-truth physics",
        allowed_generation_deviation=cfg["qualification"],
        hard_observations="Immutable first/last context; internal sparse observations are soft references; unknown C4 not supervised",
        strict_lower_bounds="Only fully fixed four-frame joint Jerk maxima; floor/support/global statistics are diagnostics, not certified lower bounds",
        human_reviewed=False,contact_truth="UNKNOWN",operational_qualified=False,whole_song_generations=0)
    if replay:assert audit==read(Path(replay)/"audit.json")
    write(out/"audit.json",audit);write(out/"audit_protocol.json",dict(config=cfg,code_commit=commit(),parent_data=ref(Path(cfg["parent"])/"data/result.json"),
        source_protocol=ref(Path(cfg["parent"])/"data/protocol.json"),new_data=ref(out/"result.json"),before_model_outputs=True))
    print("R22_DATA_READY",len(history),dict(Counter(x["status"] for x in candidates)),"mechanism",len(tr),len(dv),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
