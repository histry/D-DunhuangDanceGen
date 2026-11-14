"""Real contiguous AIST pairs and frozen ChangE E1-W contract, before GPU outcomes."""
import argparse,hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter
from dunhuang.data.conversion import decode as decode_raw
from dunhuang.generation.soft_reference import build_soft_reference,build_native_event_reference
from dunhuang.research.source_capacity.common import read,commit,fresh,same_arrays
from dunhuang.research.masked_denoiser.common import assets_and_population
from .contract import views,continuous_pair,proxy_plan,plan_grid,observations

def run(cfg,out,replay=None):
    out=fresh(out)
    assert sha("reports/r15_summary.json")==read("reports/r15_receipt.json")["summary_sha256"]
    r15=Path(cfg["r15_root"]);seal=read("reports/r15_summary.json")
    for name in ("preflight/result.json","native/result.json"):
        assert sha(r15/name)==seal["evidence"][name]
    old=read(cfg["r13_config"]);assets,rows,_,monitors,_=assets_and_population(old)
    assert sha(Path(cfg["r13_root"])/"main/protocol.json")==read("reports/r13_summary.json")["evidence"]["main/protocol.json"]
    assert rows==read(Path(cfg["r13_root"])/"main/protocol.json")["fit_population"]
    by={(r["recording_uid"],r["start_frame"]):r for r in rows};pairs=[]
    for l in rows:
        r=by.get((l["recording_uid"],l["start_frame"]+75))
        # Real outside 8-frame evaluation halos are required, not padded.
        if r is not None and l["start_frame"]>=8:pairs.append((l,r))
    groups=sorted({l["source_group"] for l,r in pairs},key=lambda x:hashlib.sha256((str(cfg["seed"])+x).encode()).hexdigest())
    if len(groups)<cfg["cases"]:raise ValueError("Insufficient registered continuous groups")
    chosen=[]
    for g in groups[:cfg["cases"]]:
        chosen.append(min((p for p in pairs if p[0]["source_group"]==g),key=lambda p:hashlib.sha256((str(cfg["seed"])+p[0]["raw_motion"]["sha256"]+str(p[0]["start_frame"])).encode()).hexdigest()))
    scaler=np.load(checked(assets["scaler"]));bindings=[];records=[]
    for i,(left,right) in enumerate(chosen):
        continuous_pair(left,right);full=np.load(checked(left["source_native"]));a=left["start_frame"]
        decoded,observed=decode_raw(checked(left["raw_motion"]),"aistpp")
        if decoded.shape!=full.shape or not observed[4:].all():raise ValueError("Raw source support/shape differs")
        raw_error=float(np.max(abs(decoded-full)))
        if raw_error>1e-6:raise ValueError("Fresh raw SMPL decode differs from frozen full recording")
        if a+225>len(full) or a+158>len(full):raise ValueError("No real continuous source/halo")
        x=full[a:a+225].copy();halo=full[a-8:a+158].copy()
        clock=read(checked(left["clock"]));cal=clock["calibration"]
        if clock["status"]!="VERIFIED_PAIR" or not cal["verified"]:raise ValueError("Unverified source clock")
        if clock["raw_motion"]["sha256"]!=left["raw_motion"]["sha256"] or clock["raw_audio"]["sha256"]!=left["audio"]["sha256"]:raise ValueError("Clock identity differs")
        features=[];producers=[];crops=[]
        for row in (left,right):
            f=np.load(checked(row["jukebox"]));producer=Path(checked(row["jukebox"])).parent/"producer.json";p=read(producer)
            if f.shape!=(150,4800) or f.dtype!=np.float32 or not np.isfinite(f).all():raise ValueError("Real Jukebox shape")
            expected=round((cal["audio_offset_seconds"]+row["start_frame"]/30)*p["rate"])
            if p["audio"]!=row["audio"]["sha256"] or p["sample_start"]!=expected or p["sample_end"]-expected!=5*p["rate"] or p["features_sha256"]!=row["jukebox"]["sha256"] or p["layer"]!=66 or p["fps"]!=30:raise ValueError("Jukebox provenance/time differs")
            audio,sr=sf.read(checked(p["crop"]),dtype="float32",always_2d=True)
            if sr!=p["rate"]:raise ValueError("Crop sample clock differs")
            features.append(f);producers.append(ref(producer));crops.append(audio)
        rate=read(checked(producers[0]))["rate"];half=round(2.5*rate)
        if not np.array_equal(crops[0][half:],crops[1][:half]):raise ValueError("Audio overlap is not the same physical samples")
        juke=np.concatenate((features[0],features[1][75:]),axis=0)
        origin=x[0,4:7].copy();origin[1]=0
        encoded=MotionAdapter.to_edge(x,origin)*scaler["scale"]+scaler["bias"]
        v,k,edge,event,keys=observations(encoded[:150],cfg["event_boundaries"][i],cfg["source_joints"],cfg["context_frames"])
        # Explicitly save the canonical timeline plus original crop-dependent views.
        file=out/("case_"+str(i)+".npz")
        np.savez_compressed(file,native=x,halo=halo,encoded=encoded,music=juke,original_music=np.stack(features),
            window_motion=views(encoded),window_music=views(juke),values=v,known=k,edge=edge,event=event,keyframes=np.array(keys))
        plan=proxy_plan(left,cfg["event_boundaries"][i]);grid=plan_grid(plan)
        assert np.array_equal(full[grid["source_positions"].astype(int)],x[:150])
        if (abs(v[k])>1+32*np.finfo(np.float32).eps).any():raise ValueError("Selected observations outside frozen Scaler support")
        write(out/("case_"+str(i)+"_plan.json"),plan)
        bindings.append(dict(case=i,left=left,right=right,arrays=ref(file),proxy_plan=ref(out/("case_"+str(i)+"_plan.json")),producer=producers))
        records.append(dict(case=i,recording=left["recording_uid"],source_group=left["source_group"],start_frame=a,
            fresh_raw_motion_max_abs=raw_error,motion_overlap_exact=True,audio_overlap_exact=True,feature_overlap_max_abs=float(abs(features[0][75:]-features[1][:75]).max()),
            global_jukebox_policy="Earlier crop owns duplicated time; second crop contributes final75. Not a full-context 225-frame extraction.",
            global_encoded_out_of_support_coordinates=int((abs(encoded)>1+32*np.finfo(np.float32).eps).sum()),native_frames=225,first_window_halo=166,keyframes=keys,semantic_event_truth="UNKNOWN"))
    planner=Path(cfg["planner_root"]);pub=read("reports/protected_source_summary.json")
    for f in ("protocol.json","result.json"):assert sha(planner/f)==pub["evidence"]["planner/"+f]
    plan_bindings=[];contracts=[]
    for b in read(planner/"protocol.json")["bindings"]:
        if b["arm"]!="E1_W":continue
        plan=read(checked(b["plan"]));grid=plan_grid(plan)
        native,mask,phase,_=build_native_event_reference(plan,plan["frames"])
        normalized,known,_,_=build_soft_reference(plan,plan["frames"],scaler["scale"],scaler["bias"])
        if mask[:,:,:79].any() or mask[:,:,127:].any():raise ValueError("Invented ChangE Root/contact/wrist support")
        if len(native)>1:
            assert np.array_equal(native[:-1,75:],native[1:,:75]) and np.array_equal(mask[:-1,75:],mask[1:,:75])
        file=out/("change_"+str(b["case"])+".npz");np.savez_compressed(file,native=native,mask=mask,phase=phase,
            normalized=normalized,known=known,**grid)
        plan_bindings.append(dict(case=b["case"],plan=b["plan"],arrays=ref(file)))
        contracts.append(dict(case=b["case"],frames=plan["frames"],windows=len(native),events=len(plan["events"]),
            global_event_clock_consistent=True,root_contact_observed=False,full_body_source_truth=False))
    result=dict(records=records,change_contracts=contracts,pairs=len(bindings),music_groups=len(chosen),
        paired_continuity_ready=True,c2_neural_training_admitted=False,optimizer_steps=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,assets=assets,
        pairs=bindings,change_plans=plan_bindings,training_population_scope="Exposed R13 fit music groups; no Val/Test selection"))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for file in out.glob("*.npz"):same_arrays(file,Path(replay)/file.name)
        for file in out.glob("*_plan.json"):assert read(file)==read(Path(replay)/file.name)
    print("R16_CONTINUOUS_DATA",len(bindings),len(contracts),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
