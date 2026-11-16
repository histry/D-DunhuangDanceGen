"""Recording-level adaptation holdouts, fixed from catalog identity before output inspection."""
from pathlib import Path
import numpy as np
import torch
from .common import read,sha
from dunhuang.io import checked
from dunhuang.contracts import matrices
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.feasible_time_map import solve
from dunhuang.planning.time_map import source_positions

def compile_holdouts(cfg,rt,music):
    assets=read("outputs/event_plan_01/prepare/assets.json");catalog=read(checked(assets["catalog"]));source=read(checked(assets["catalog_source"]))
    entries=catalog["entries"];records=[];bodies=[];local=[]
    excluded={r["source_sha256"] for r in source["source_records"] if r["source_uid"] in ("female_FeiTian","male_pipa_1")}
    for recording in cfg["held_recordings"]:
        candidates=sorted([e for e in entries if e["recording_uid"]==recording and e["duration_frames"]>=96],key=lambda e:e["event_uid"])
        if not candidates:raise ValueError("Registered held recording lacks complete duration; no replacement")
        event=candidates[0];assert event["source_sha256"] not in excluded
        origin=next(r for r in source["source_records"] if r["source_uid"]==event["source_uid"]);checked(origin["raw_receipt"])
        beats=np.flatnonzero(music[0,:96,-1]>.5).tolist()
        solution=solve(dict(source_interval=event["source_interval"],center=event["center"],normalization_support=True),dict(start=0,end=96),beats)
        if solution["status"]!="PASS":raise ValueError("Registered holdout map failed; no best-output replacement")
        mapped=dict(event,music_interval=[0,96],time_warp=solution["time_warp"])
        native=resample_native(np.load(checked(event["reference"])),source_positions(mapped,np.arange(96)))
        R=matrices(native[:,7:].reshape(96,24,6))[:,:22];body=R[...,:2,:].reshape(96,132).astype(np.float32)
        # Translation zero is ONLY a FK gauge; no complete source x0 or world trajectory is made.
        tmp=torch.zeros(1,96,139,device="cuda");tmp[0,:,7:]=torch.tensor(body,device="cuda")
        with torch.no_grad():p=rt.f(tmp,rt.s)[0,:,:22].cpu().numpy()
        bodies.append(body);local.append(p-p[:,:1])
        records.append(dict(event_uid=event["event_uid"],source_uid=event["source_uid"],recording_uid=recording,source_sha256=event["source_sha256"],
            raw_receipt=origin["raw_receipt"],reference=event["reference"],source_interval=event["source_interval"],time_warp=solution["time_warp"],
            map_certificate=solution,historical_usage_count=origin.get("historical_usage_count"),historical_split=origin.get("historical_recording_split"),
            adaptation_role="NOT_IN_R33_TRAIN; HISTORICALLY_EXPOSED_CATALOG_NOT_NEW_BLIND_TEST",Root_truth="UNKNOWN",C4_truth="UNKNOWN"))
    assert len({r["recording_uid"] for r in records})==2 and len({r["source_sha256"] for r in records})==2
    return dict(body=np.stack(bodies),local_points=np.stack(local)),records,dict(catalog=assets["catalog"],source=assets["catalog_source"])
