"""Independent aggregate readout; never overwrites frozen scientific artifacts."""
import argparse,collections,json,re,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write,checked
from dunhuang.alignment.conditional_candidates import nuisance_statistics
from dunhuang.alignment.dense_coverage import fast_candidates
from dunhuang.planning.time_map import motion_features74
from dunhuang.planning.primitive_contract import verify

def read(p):return json.loads(Path(p).read_text())
def seal(out,receipt):
    out=Path(out);receipt=Path(receipt);dest=out/"public_receipt.json"
    if dest.exists():raise ValueError("Receipt destination already exists")
    if (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("tmux launcher did not succeed")
    summary=read(out/"public_summary.json")
    for name,digest in summary["evidence"].items():
        if sha(out/name)!=digest:raise ValueError("Frozen public evidence seal differs")
    g=read(out/"g1/result.json");a=read(out/"a1/result.json")
    for directory,name,status in (("g1_replay","replay.json","FRESH_G1_REPLAY_VERIFIED"),("a1_replay","replay.json","FRESH_FULL_SOURCE_DENSE_REBUILD_VERIFIED")):
        replay=read(out/directory/name)
        if replay["status"]!=status or replay["optimizer_calls"]!=0:raise ValueError("Fresh replay qualification absent")
    predictions=np.load(out/"g1/denoising.npz")
    invariance=0;bypass=0;legacy_sensitive=0;native=collections.Counter();limits=None
    for case in range(2):
        for t in (100,500,900):
            key=str(case)+"_"+str(t)+"_"
            same=np.array_equal(predictions[key+"observed_a"],predictions[key+"observed_corrupted"])
            zero=np.array_equal(predictions[key+"observed_removed"],predictions[key+"zero_adapter"])
            if not same or not zero:raise ValueError("Real EDGE visibility invariant failed")
            invariance+=int(same);bypass+=int(zero)
            legacy_sensitive+=not np.array_equal(predictions[key+"legacy_a"],predictions[key+"legacy_corrupted"])
        for ref in ("reference_a","reference_b"):
            folder=out/"g1"/("case_"+str(case))/ref;verify(read(folder/"condition_contract.json"),read(folder/"plan.json"))
        for control in ("legacy_a","legacy_b","legacy_removed","observed_a","observed_b","observed_removed","zero_adapter"):
            original=out/"g1"/("case_"+str(case))/control;replayed=out/"g1_replay"/("case_"+str(case))/control
            p=read(original/"physical.json");r=read(replayed/"physical.json")
            if p!=r:raise ValueError("Native full protocol replay differs")
            if limits is None:limits=p["limits"]
            if p["limits"]!=limits:raise ValueError("Native limits differ across interventions")
            native.update(p["absolute"]["reasons"])
    rows=read(out/"a1/data/manifest.json")["rows"];bank=dict(np.load(out/"a1/data/nodes.npz"));c=dict(np.load(out/"a1/data/conditional.npz"))
    for i in np.flatnonzero(c["eligible"]):
        for j in c["candidates"][i][c["positive"][i]]:
            if not bank["mv"][j].any() or not bank["dv"][j].any():raise ValueError("Frozen candidate has unobserved positive")
    manifest=read(out/"a1/data/manifest.json")
    event_source=Path(manifest["source_population"]["path"]).parent.parent
    cache=Path(manifest["source_clock_manifest"]["path"]).parent/"audio"
    energy_m=[];energy_d=[];source_values={};audio_values={};events={}
    for row in rows:
        rk=row["raw_motion"]["sha256"];ak=row["audio"]["sha256"]
        if rk not in source_values:source_values[rk]=motion_features74(np.load(checked(row["source_native"])))[0][:,66]
        if ak not in audio_values:
            audio_values[ak]=np.load(cache/(ak+".npz"))["values"][:,0]
            events[ak]=read(event_source/"data/sidecars"/(ak+".json"))["events"]
        start=row["start_frame"];music_start=start+15+round(30*row["audio_offset_seconds"])
        energy_m.append(audio_values[ak][music_start:music_start+120])
        energy_d.append(source_values[rk][start:start+150])
    strict=read("configs/local_identifiability.json")["strict"]
    stats=nuisance_statistics(rows,bank,dict(energy_music=np.stack(energy_m),energy_motion=np.stack(energy_d)),events)
    current,evidence=fast_candidates(rows,bank,stats,strict)
    if any(not np.array_equal(current[k],c[k]) for k in current):raise ValueError("Hardened candidate code changed frozen cohort")
    if read(out/"a1/protocol.json")["config"]["strict"]!=strict:raise ValueError("Historical strict gates changed")
    fd=[e for r in g["denoising"] for v in r.get("sensitivity",[]) for e in v["finite_differences"]]
    if len(fd)!=6:raise ValueError("Registered finite-difference population differs")
    fd_pass=all(e["absolute_error"]<=max(1e-10,.05*abs(e["autograd"])) for e in fd)
    payload=dict(schema="dense_condition_completed_receipt_v1",
        execution_commit=(receipt/"execution_commit.txt").read_text().strip(),
        readout_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        exit_status=0,started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        g1_generation_count=g["generations_executed"],g1_fresh_generation_replays=14,
        real_edge_unknown_storage_invariance_passes=invariance,real_edge_removed_equals_bypass_passes=bypass,
        legacy_unknown_storage_sensitive_cases=int(legacy_sensitive),native_passes=g["native_passes"],native_failure_histogram=dict(native),
        native_limits_constant=True,native_limits_sha256=__import__("hashlib").sha256(json.dumps(limits,sort_keys=True).encode()).hexdigest(),
        derivative_fd_confirmed=fd_pass,derivative_status="CONFIRMED_REGISTERED_STEPS" if fd_pass else "NOT_QUALIFIED_FINITE_DIFFERENCE_SIGNAL_TOO_WEAK",
        a1_windows=a["windows"],a1_recordings=a["recordings"],a1_population=a["strict_population"],a1_optimizer_updates=a["optimizer_updates"],
        thresholds_relaxed=False,original_scientific_files_unchanged=True,current_candidate_code_exact_replay=True,
        logs={name:sha(receipt/name) for name in ("pytest.log","edge_check.log","exit_status.txt","started_utc.txt","finished_utc.txt","execution_commit.txt")},
        final_validation={name:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1] for name in ("validation_pytest.log","validation_edge.log")},
        evidence_sha256={name:sha(out/name) for name in ("public_summary.json","g1/result.json","g1_replay/replay.json","a1/result.json","a1_replay/replay.json")},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,new_blind_source=False,operational_qualified=False)
    if any(s in json.dumps(payload) for s in ("/home/","window_uid","aist_capture:","known_component:")):raise ValueError("Private readout rejected")
    write(dest,payload);print("PUBLIC_RECEIPT_SEALED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True)
    a=p.parse_args();seal(a.out,a.receipt)
