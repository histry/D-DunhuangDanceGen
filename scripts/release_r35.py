"""Exact replay verification and receipts; never converts graph cost to naturalness."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write
from dunhuang.data.sources import ref
from dunhuang.research.state_r27.common import read,fresh,commit,forbid_optimizers,same_arrays,readout_check

def aggregate(result):
    rows=[]
    for role in ("train","holdout"):
        for arm in ("Frozen","Context"):
            values=[v for v in result["paired"] if v["role"]==role and v["arm"]==arm]
            rows.append(dict(role=role,arm=arm,n=len(values),body_rad=float(np.mean([v["geometry"]["body22_rad"] for v in values])),
                root_rmse_m=float(np.mean([v["root_rmse_m"] for v in values])),
                cross_D2=float(np.mean([v["cross"][k] for v in values for k in ("64_D2","160_D2")])),
                gap_jerk=float(np.mean([v["gap_body_jerk"] for v in values]))))
    protection=[]
    for event in (0,1):
        values={arm:[v["geometry"] for v in result["source_protection"] if v["arm"]==arm and v["event"]==event] for arm in ("Frozen","Context")}
        ratios={k:float(np.mean([v[k] for v in values["Context"]])/np.mean([v[k] for v in values["Frozen"]])) for k in ("body22_rad","world_wrist_rmse_m","body_activity_ratio","arm_range_ratio")}
        protection.append(dict(event=event,ratios=ratios,passed=max(ratios["body22_rad"],ratios["world_wrist_rmse_m"])<=1.1 and all(.8<=ratios[k]<=1.25 for k in ("body_activity_ratio","arm_range_ratio"))))
    h={v["arm"]:v for v in rows if v["role"]=="holdout"}
    primary=all(h["Context"][k]<h["Frozen"][k] for k in ("body_rad","root_rmse_m","cross_D2")) and h["Context"]["gap_jerk"]<=h["Frozen"]["gap_jerk"]*1.1
    return dict(paired=rows,source_protection=protection,finite_primary_gate=bool(primary and all(p["passed"] for p in protection)),
        scope="HISTORICALLY_EXPOSED_SMALL_MAIN_EFFECT; SIDE_NECESSITY_ABLATIONS_DEFERRED; NOT_NATIVE_OR_POPULATION_GENERALIZATION")

def verify(root):
    root=Path(root);forbid_optimizers();out=root/"independent_checks.json";assert not out.exists()
    cfg=read("src/dunhuang/research/transition_r35/config.json")
    same_arrays(root/"planning/edges.npz",root/"planning_replay/edges.npz")
    for n in ("songs.json","source_audit.json","result.json"):
        x=json.dumps(read(root/"planning"/n),sort_keys=True).replace(str((root/"planning").resolve()),"<RUN>")
        y=json.dumps(read(root/"planning_replay"/n),sort_keys=True).replace(str((root/"planning_replay").resolve()),"<RUN>")
        assert x==y
    a=read(root/"context/result.json");b=read(root/"context_replay/result.json")
    assert a["optimizer_steps"]==600 and b["optimizer_steps"]==0
    assert a["final_digest"]==b["final_digest"] and a["frozen"]==b["frozen"]
    for n,h in a["files"].items():assert sha(root/"context"/n)==h
    for n in ("initial_states.npz","refresh_300.npz"):same_arrays(root/"context"/n,root/"context_replay"/n)
    checks=[readout_check(x,y) for x,y in zip(b["actual_readouts"],a["observations"])]
    assert len(checks)==4
    same_arrays(root/"generation/arrays.npz",root/"generation_replay/arrays.npz")
    x=read(root/"generation/result.json");y=read(root/"generation_replay/result.json")
    for row in x["records"]:row.pop("audio_window")
    for row in y["records"]:row.pop("audio_window")
    assert x==y
    for parent,h in ((cfg["r33"],cfg["r33_manifest_sha256"]),(cfg["r34"],cfg["r34_manifest_sha256"])):
        p=Path(parent);assert sha(p/"artifacts_manifest.json")==h
        for n,v in read(p/"artifacts_manifest.json")["files"].items():assert sha(p/n)==v["sha256"]
    changed=subprocess.check_output(["git","diff",cfg["baseline"],"--name-only","--","src/dunhuang"],text=True).splitlines()
    assert all(n.startswith("src/dunhuang/research/transition_r35/") for n in changed)
    fail=read(root/"context_failed01/failure_accounting.json")
    result=dict(code_commit=commit(),scientific_array_bundles_exact=4,actual_update_pairs=4,update_checks=checks,
        successful_optimizer_steps=600,replay_optimizer_steps=0,
        successful_sampler_calls=a["sampler_calls"]+x["sampler_calls"],successful_trajectories=a["generated_trajectories"]+x["generated_trajectories"],
        replay_sampler_calls=b["sampler_calls"]+y["sampler_calls"],replay_trajectories=b["generated_trajectories"]+y["generated_trajectories"],
        failed_original_sampler_calls=fail["sampler_calls"],failed_original_trajectories=fail["generated_trajectories"],failed_optimizer_steps=0,
        frozen_R33_R34_seals_unchanged=True,source_backbone_IO_frozen=True,context_only=True,additional_ablation_calls=0,whole_song_generations=0,
        comparison=aggregate(x),operational_qualified=False,full_native_status="NOT_EVALUABLE_22_OF24")
    write(out,result);print("R35_INDEPENDENT",result["successful_trajectories"],flush=True)

def collect(root,out):
    root=Path(root);out=fresh(out);manifest=root/"artifacts_manifest.json";assert not manifest.exists()
    qa=read(root/"browser_qa.json");assert qa["verified"] and qa["localhost_closed"]
    for n,h in qa["ui_hashes"].items():assert sha(root/"viewer"/n)==h
    required=("plan","train02","preview")
    for phase in required:assert Path(str(root)+"_"+phase+"_receipts/exit_status.txt").read_text().strip()=="0"
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.rglob("*") if p.is_file()}
    write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    summary=dict(config=read("src/dunhuang/research/transition_r35/config.json"),planning=read(root/"planning/result.json"),
        training={k:v for k,v in read(root/"context/result.json").items() if k not in ("history","observations","actual_readouts")},
        checks=read(root/"independent_checks.json"),viewer=read(root/"viewer/protocol.json"),browser_qa=qa,scientific_manifest=ref(manifest),
        human_reviewed=False,operational_qualified=False,whole_song_generations=0,complete_three_event_sequences=0,full_native_status="NOT_EVALUABLE_22_OF24")
    write(out/"r35_summary.json",summary);write(out/"r35_checks.json",summary["checks"])
    write(out/"r35_receipt.json",dict(code_commit=commit(),scientific_manifest=ref(manifest),summary_sha256=sha(out/"r35_summary.json"),
        receipts={phase:{p.name:sha(p) for p in Path(str(root)+"_"+phase+"_receipts").iterdir() if p.is_file()} for phase in (*required,"train")},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0))
    print("R35_COLLECTED",len(files),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["verify","collect"],required=True);p.add_argument("--root",required=True);p.add_argument("--out");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    else:collect(a.root,a.out)
