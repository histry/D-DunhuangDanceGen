"""Fresh Data V3 full Root/rotation masked-target support and real cached music join."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read,commit,fresh

def run(out):
    out=fresh(out);parent=Path("outputs/transfer_audit_01/a2");seal=read("reports/transfer_audit_summary.json")
    assert sha(parent/"manifest.json")==seal["evidence"]["a2/manifest.json"]
    protocol=read(parent/"protocol.json");data=read(parent/"manifest.json")["rows"]
    music=read(checked(protocol["paired_manifest"]))["rows"]
    index={(r["raw_motion"]["sha256"],r["audio"]["sha256"],r["start_frame"],r["audio_offset_seconds"]):r for r in music}
    assets=read(checked(protocol["assets"]));scaler=np.load(checked(assets["scaler"]));counts={r:dict(total=0,context_supported=0,target_supported=0,real_feature_join=0,ready=0) for r in ("train","val")}
    native_cache={};verified_music=set();rows=[];records=[];groups={r:set() for r in counts}
    support_frames=np.r_[52:60,60:90,90:98]
    for row in data:
        role=row["role"]
        if role not in counts:raise ValueError("No Test reads allowed")
        counts[role]["total"]+=1;key=row["source_native"]["sha256"]
        if key not in native_cache:native_cache[key]=np.load(checked(row["source_native"]))
        start=row["start_frame"];x=native_cache[key][start:start+150]
        if len(x)!=150:raise ValueError("Incomplete real target")
        origin=x[0,4:7].copy();origin[1]=0;edge=MotionAdapter.to_edge(x,origin)
        normalized=edge*scaler["scale"]+scaler["bias"];tol=1+32*np.finfo(np.float32).eps
        context=bool((abs(normalized[np.r_[52:60,90:98],4:])<=tol).all());target=bool((abs(normalized[60:90,4:])<=tol).all())
        counts[role]["context_supported"]+=context;counts[role]["target_supported"]+=target
        k=(row["raw_motion"]["sha256"],row["audio"]["sha256"],start,row["audio_offset_seconds"])
        m=index.get(k);counts[role]["real_feature_join"]+=m is not None
        if context and target and m:
            checked(row["source_clock_evidence"]);checked(m["alignment_attestation"])
            f=m["jukebox"]
            if f["sha256"] not in verified_music:
                feature=np.load(checked(f),mmap_mode="r")
                if feature.shape!=(150,4800) or not np.isfinite(feature).all():raise ValueError("Real Jukebox shape differs")
                verified_music.add(f["sha256"])
            counts[role]["ready"]+=1;groups[role].add(row["source_group"])
            rows.append(dict(role=role,source_group=row["source_group"],recording_uid=row["recording_uid"],source_native=row["source_native"],
                start_frame=start,jukebox=f,clock=row["source_clock_evidence"],raw_motion=row["raw_motion"],audio=row["audio"],contact_target_observed=False))
    assert groups["train"].isdisjoint(groups["val"])
    ready=counts["train"]["ready"]>=32 and counts["val"]["ready"]>=8 and len(groups["train"])>=3 and len(groups["val"])>=2
    result=dict(counts=counts,groups={r:len(v) for r,v in groups.items()},bridge_training_data_ready=ready,contacts_excluded_from_truth=True,
        target_coordinates="Root and actual 24 rotations, only masked frames; observed context support independently checked",
        test_reads=0,neural_weight_updates=0,new_neural_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"rows.json",rows);write(out/"protocol.json",dict(code_commit=commit(),manifest=dict(path=str((parent/"manifest.json").resolve()),sha256=sha(parent/"manifest.json")),
        paired_manifest=protocol["paired_manifest"],assets=protocol["assets"],preflight_thresholds=dict(train=32,val=8,train_music_groups=3,val_music_groups=2),
        root_origin="actual first observed frame XZ; height preserved",context=[52,60,90,98],edit=[60,90],contact_supervision=False))
    print("R9_B2_DATA_PREFLIGHT",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();run(a.out)
