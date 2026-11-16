"""R34 exact scientific replay and public metadata; two-stage visual-first delivery."""
import argparse,json,hashlib,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
from dunhuang.io import sha,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,fresh,commit,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.common import readout_check
from dunhuang.research.plan_view_r34.verify import verify_generation

def verify(root):
    root=Path(root);forbid_optimizers();out=root/"independent_checks.json";assert not out.exists()
    cfg=read("src/dunhuang/research/coverage_r34/config.json");arrays=0;updates=[]
    a=read(root/"data/result.json");b=read(root/"data_replay/result.json")
    assert a["training_admitted"] and a["optimizer_steps"]==2 and b["optimizer_steps"]==0
    for x,y in zip(b["actual_readouts"],a["updates"]):updates.append(readout_check(x,y))
    for p in (root/"data").glob("*.npz"):same_arrays(p,root/"data_replay"/p.name);arrays+=1
    models=[]
    for arm in ("Extra","Coverage"):
        p=root/f"models/{arm}";a=read(p/"result.json");b=read(root/f"replay/{arm}/result.json")
        assert a["optimizer_steps"]==600 and b["optimizer_steps"]==0
        for k in ("initial_digest","final_digest","frozen","curves","history","dynamic_weight","paired_source_counts"):assert a[k]==b[k]
        for name,h in a["files"].items():assert sha(p/name)==h
        for x,y in zip(b["actual_readouts"],a["observations"]):updates.append(readout_check(x,y))
        for file in p.glob("*.npz"):same_arrays(file,root/f"replay/{arm}"/file.name);arrays+=1
        assert a["source_target_frames"]==(96 if arm=="Extra" else 256) and not a["context_trained"]
        models.append(a)
    assert models[0]["initial_digest"]==models[1]["initial_digest"]
    ledger=lambda m:[{k:v for k,v in r.items() if k not in ("loss","parts")} for r in m["history"]]
    assert ledger(models[0])==ledger(models[1])
    counts=Counter(c for row in models[0]["history"] if row["task"]=="paired" for c in row["cases"]);assert len(counts)==16 and set(counts.values())=={75}
    preview=read(root/"preview/result.json");assert preview==read(root/"preview_replay/result.json") and preview["preview_only"]
    same_arrays(root/"preview/arrays.npz",root/"preview_replay/arrays.npz");arrays+=1
    for name in ("candidate_banks.json","songs.json","result.json"):
        a=json.dumps(read(root/"planning"/name),sort_keys=True).replace(str((root/"planning").resolve()),"<RUN>")
        b=json.dumps(read(root/"planning_replay"/name),sort_keys=True).replace(str((root/"planning_replay").resolve()),"<RUN>");assert a==b
    plan_check=verify_generation(root/"plan_generation",root/"plan_generation_replay");arrays+=1
    p=Path(cfg["r33"]);assert sha(p/"artifacts_manifest.json")==cfg["r33_manifest_sha256"]
    for name,v in read(p/"artifacts_manifest.json")["files"].items():assert sha(p/name)==v["sha256"]
    changes=subprocess.check_output(["git","diff",cfg["baseline"],"--name-only","--","src/dunhuang"],text=True).splitlines()
    assert all(x.startswith(("src/dunhuang/research/coverage_r34/","src/dunhuang/research/plan_view_r34/")) for x in changes)
    assert arrays==9 and len(updates)==10
    original_calls=sum(x["sampler_calls"] for x in [read(root/"data/result.json"),*models,preview,read(root/"plan_generation/result.json")])
    original_trajectories=sum(x["generated_trajectories"] for x in [read(root/"data/result.json"),*models,preview,read(root/"plan_generation/result.json")])
    assert (original_calls,original_trajectories)==(18,76)
    result=dict(code_commit=commit(),array_bundles_exact=arrays,actual_update_pairs=10,update_checks=updates,
        optimizer_steps=1202,replay_optimizer_steps=0,sampler_calls=18,generated_trajectories=76,replay_sampler_calls=18,replay_generated_trajectories=76,
        components=dict(preflight=[4,20],training=[2,12],primary_preview=[6,32],plan_windows=[6,12]),paired_presentations=dict(counts),
        PCM_generation_replay=plan_check,legacy_96_exact_R33=True,general_and_NoRef_exact=True,old_scientific_seal_unchanged=True,
        context_optimizer_steps=0,additional_ablation_calls=0,full_song_generations=0,operational_qualified=False,full_native_status="NOT_EVALUABLE_22_OF24")
    write(out,result)
    write(root/"context_deferred.json",dict(reason="VISUAL_FIRST_AND_COVERAGE_STABILITY_GATE_FAILED",coverage_gate=preview["gate"],context_parameters_updated=False,
        context_training_code_present=True,actual_context_optimizer_steps=0,new_ablation_calls=0,semantic_phase_channels_trained=False,whole_song_generations=0))
    print("R34_INDEPENDENT",arrays,len(updates),18,76,flush=True)

def comparison(result):
    rows=[]
    for arm in ("Frozen","FrozenFull","Extra","Coverage"):
        for e in (0,1):
            records=[r["errors"][e] for r in result["records"] if r["arm"]==arm and r["event"]==e and r["mode"]=="correct"]
            for region in ("before","core","after","full"):
                record=dict(arm=arm,event=e,region=region,music_conditions=len(records))
                for key in ("body22_rad","world_wrist_rmse_m","arm_range_ratio","body_activity_ratio"):record[key]=float(np.mean([r["regions"][region]["geometry"][key] for r in records]))
                for k in ("1","2","3"):record["D"+k]=float(np.mean([r["regions"][region]["paired_body"][k]["rmse"] for r in records]))
                for key in ("world","Root_translation","Root_relative_body"):record["Jerk_"+key]=float(np.mean([r["regions"][region]["jerk"][key] for r in records]))
                rows.append(record)
    return rows

def collect(root,out):
    root=Path(root);out=fresh(out);manifest=root/"artifacts_manifest.json";assert not manifest.exists()
    required=("preflight","train02","preview","viewbundle")
    for phase in required:
        p=Path(str(root)+"_"+phase+"_receipts");assert (p/"exit_status.txt").read_text().strip()=="0" and (p/"phase.txt").read_text().strip()=="completed"
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.rglob("*") if p.is_file()}
    write(manifest,dict(code_commit=commit(),files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values())))
    summary=dict(config=read("src/dunhuang/research/coverage_r34/config.json"),preflight=read(root/"data/result.json"),
        models={arm:{k:v for k,v in read(root/f"models/{arm}/result.json").items() if k!="history"} for arm in ("Extra","Coverage")},
        comparison=comparison(read(root/"preview/result.json")),coverage_gate=read(root/"preview/result.json")["gate"],
        context=read(root/"context_deferred.json"),planning=read(root/"planning/result.json"),plan_generation=read(root/"plan_generation/result.json"),
        checks=read(root/"independent_checks.json"),viewer=read(root/"viewer/protocol.json"),scientific_manifest=ref(manifest),
        human_reviewed=False,operational_qualified=False,full_native_status="NOT_EVALUABLE_22_OF24",whole_song_generations=0)
    write(out/"r34_summary.json",summary);write(out/"r34_checks.json",summary["checks"])
    write(out/"r34_receipt.json",dict(collection_commit=commit(),summary_sha256=sha(out/"r34_summary.json"),scientific_manifest=ref(manifest),
        receipts={phase:{p.name:sha(p) for p in Path(str(root)+"_"+phase+"_receipts").iterdir() if p.is_file()} for phase in required},
        failed_receipts={phase:{p.name:sha(p) for p in Path(str(root)+"_"+phase+"_receipts").iterdir() if p.is_file()} for phase in ("train","planview")},
        evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json")},launcher_logs={str(p):sha(p) for p in Path("outputs").glob("coverage_r34_*_launcher.log")},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0))
    print("R34_PUBLIC_STATS",sha(out/"r34_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--task",choices=["verify","collect"],required=True);p.add_argument("--root",required=True);p.add_argument("--out");a=p.parse_args()
    if a.task=="verify":verify(a.root)
    else:collect(a.root,a.out)
