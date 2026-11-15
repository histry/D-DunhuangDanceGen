"""Fresh public destination; retain every frozen and failed scientific output."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha,checked
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit

def q_summary(scores):
    result=[]
    for case,variant in sorted({(x["case"],x["variant"]) for x in scores}):
        rows=[r for r in scores if r["case"]==case and r["variant"]==variant]
        old=[r for r in rows if r["t"] in (999,519,119,19)]
        result.append(dict(case=case,variant=variant,grid_points=len(rows),grid_mean=float(np.mean([r["unknown"] for r in rows])),
            old4_mean=float(np.mean([r["unknown"] for r in old])),
            t19=next(r["unknown"] for r in old if r["t"]==19),
            branch_old4={branch:float(np.mean([r["branches"][branch]["unknown"] for r in old])) for branch in ("unconditional_music","conditional_music")},
            controls={c:float(np.mean([r["controls"][c] for r in old])) for c in ("left_wrong","right_wrong","both_absent")} if all("controls" in r for r in old) else {}))
    return result

def contrasts(cfg,records,q_scores):
    lookup={(r["arm"],r["case"],r["variant"]):r for r in records if r["control"]=="correct" and r["role"]=="dev_exposed"}
    qs={a:{(r["case"],r["variant"]):r for r in q_summary(rows)} for a,rows in q_scores.items()}
    pairs=[("B1","B0","layout"),("B3","B2","layout"),("B2","B0","boundary"),("B3","B1","boundary"),("M","Q","cache")]
    if "F" in q_scores:pairs.append(("F","Q","support"))
    limits=cfg["qualification"];out=[]
    for new,base,kind in pairs:
        cases=[]
        for case in sorted({k[1] for k in lookup if k[0]==new}):
            variants=[]
            for variant in cfg["eval_variants"]:
                a=lookup[new,case,variant];b=lookup[base,case,variant]
                qa=qs[new][case,variant];qb=qs[base][case,variant]
                rom=lambda x:float(np.mean(x["dynamics"]["joint_range_p95_rad"][1:22]))
                common=a["pose_error_rad"]<=b["pose_error_rad"]*limits["max_pose_ratio"] and a["root_rmse_m"]<=b["root_rmse_m"]*limits["max_root_ratio"]
                common=common and a["activity_mps"]>=b["activity_mps"]*limits["min_activity_ratio"] and rom(a)>=rom(b)*limits["min_activity_ratio"]
                common=common and a["hard_domains"]["editable"]["p95"]<=b["hard_domains"]["editable"]["p95"]*limits["max_internal_jerk_ratio"]
                common=common and qa["grid_mean"]<=qb["grid_mean"]*limits["max_q_ratio"] and qa["old4_mean"]<=qb["old4_mean"]*limits["max_q_ratio"]
                if kind in ("layout","boundary"):target=a["hard_domains"]["crossing"]["max"]<=b["hard_domains"]["crossing"]["max"]*(1-limits["min_gain"])
                elif kind=="cache":target=a["actual_t19_unknown"]<=b["actual_t19_unknown"]*(1-limits["min_gain"])
                else:
                    pa=a["source_proxy_readouts"];pb=b["source_proxy_readouts"]
                    valid=all(x is not None for x in (pa["velocity_rms_mps"],pb["velocity_rms_mps"],pa["anchor_rmse_m"],pb["anchor_rmse_m"]))
                    target=valid and pa["velocity_rms_mps"]<=pb["velocity_rms_mps"]*(1-limits["min_gain"]) and pa["anchor_rmse_m"]<=pb["anchor_rmse_m"]*limits["max_pose_ratio"]
                    target=target and a["native_audit"]["foot_skate_mps_p95"]<=b["native_audit"]["foot_skate_mps_p95"]*(1-limits["min_gain"]) and a["native_audit"]["foot_support_drift_m_p95"]<=b["native_audit"]["foot_support_drift_m_p95"]*(1-limits["min_gain"])
                variants.append(dict(variant=variant,retained=bool(common),target_evaluable=valid if kind=="support" else True,
                    status="NOT_EVALUABLE" if kind=="support" and not valid else "PASS" if common and target else "FAIL",target_improved=bool(target),passed=bool(common and target)))
            cases.append(dict(case=case,variants=variants,passed=all(r["passed"] for r in variants)))
        out.append(dict(new=new,base=base,kind=kind,cases=cases,passed_groups=sum(x["passed"] for x in cases),
            research_contrast_passed=sum(x["passed"] for x in cases)>=limits["required_improved_dev_groups"],used_for_model_selection=False))
    return out

def run(root,destination,receipt_root=None):
    root=Path(root).resolve();destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for n in ("r21_summary.json","r21_checks.json","r21_receipt.json"):assert not (destination/n).exists(),"Use a fresh publish directory; preserve prior files"
    completion_path=Path(receipt_root) if receipt_root else Path(str(root)+"_receipts")
    assert (completion_path/"exit_status.txt").read_text().strip()=="0"
    original_path=Path(str(root)+"_receipts");cfg=read(root/"data/protocol.json")["config"];data=read(root/"data/result.json");checks=read(root/"independent_checks.json")
    arms=[a for a in cfg["arms"] if a!="F" or data["support_admitted"]]
    models={}
    for arm in arms:
        model=read(root/"models"/arm/"result.json")
        models[arm]={n:model[n] for n in ("layout","balanced","initial","initial_checkpoint","checkpoint","final_selected","final_head","update_observations","optimizer_steps","trainable_motion","trainable_head","wall_seconds","peak_cuda_bytes","forward_calls","branch_training")}
        models[arm].update(curves=[dict(update=c["update"],weight=c["weight"],scores=q_summary(c["scores"]),predictions=c["predictions"]) for c in model["curves"]],
            final_train_q=q_summary(model["final_scores"]),cache_bindings=[{n:v for n,v in b.items() if n!="receipt"} for b in model["cache_bindings"]],
            first_history=model["history"][0],last_history=model["history"][-1])
    native=read(root/"native/result.json")["records"];roll=read(root/"rollout/result.json");q_scores={a:v["scores"] for a,v in roll["q_scores"].items()}
    means=[]
    for arm in arms:
        for role in ("train","dev_exposed"):
            for variant in cfg["eval_variants"]:
                rows=[r for r in native if r["arm"]==arm and r["role"]==role and r["variant"]==variant and r["control"]=="correct"]
                means.append(dict(arm=arm,role=role,variant=variant,executions=len(rows),source_groups=len({r["source_group"] for r in rows}),
                    native_passes=sum(r["native_ok"] for r in rows),
                    metrics={n:float(np.mean([r[n] for r in rows])) for n in ("root_rmse_m","pose_error_rad","activity_mps","jerk_p95_mps3","jerk_max_mps3","q_t19_unknown","actual_t19_unknown","actual_over_q")},
                    crossing_jerk_max_mean=float(np.mean([r["hard_domains"]["crossing"]["max"] for r in rows])),
                    editable_jerk_p95_mean=float(np.mean([r["hard_domains"]["editable"]["p95"] for r in rows])),
                    fixed_jerk_max_mean=float(np.mean([r["hard_domains"]["fixed"]["max"] for r in rows])),
                    foot_skate_p95_mean=float(np.mean([r["native_audit"]["foot_skate_mps_p95"] for r in rows])),
                    support_drift_p95_mean=float(np.mean([r["native_audit"]["foot_support_drift_m_p95"] for r in rows]))))
    stages=[]
    for b in roll["bindings"]:
        trace_receipt=read(checked(b["sampler"]))
        selected=[r for r in trace_receipt["records"] if "stages" in r]
        if selected:stages.append(dict(arm=b["arm"],case=b["case"],variant=b["variant"],records=selected))
    registered=[{n:row[n] for n in ("case","role","left","right","arrays","audio225","source_physical","support_frames","swing_frames","full_source_floor","historical_exposure")} for row in data["records"]]
    result=dict(schema=cfg["schema"],code_commit=commit(),config=cfg,data=dict(records=registered,candidate_counts=data["candidate_counts"],source_native_passes=data["source_native_passes"],
        train_groups=data["train_groups"],dev_groups=data["dev_groups"],train_cases=data["train_cases"],dev_cases=data["dev_cases"],train_probe=data["train_probe"],
        follow_cases=data["follow_cases"],probe_selection=data["probe_selection"],support_admitted=data["support_admitted"],source_population=ref(root/"data/result.json")),
        training=models,means=means,native=native,final_q={a:q_summary(s) for a,s in q_scores.items()},stage_readouts=stages,
        contrasts=contrasts(cfg,native,q_scores),checks=checks,
        support_calibration=read(root/"support_calibration/result.json") if data["support_admitted"] else None,
        videos=read(root/"videos/protocol.json")["records"],human_reviewed=False,contact_truth="UNKNOWN",
        branch_training_comparison_executed=False,free_future_root_tested=False,whole_song_generations=0,operational_qualified=False)
    summary=destination/"r21_summary.json";write(summary,result);write(destination/"r21_checks.json",checks)
    evidence={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*.json")) if p.name in ("result.json","protocol.json","independent_checks.json")}
    write(destination/"r21_receipt.json",dict(code_commit=commit(),execution_commit=(original_path/"execution_commit.txt").read_text().strip(),
        summary_sha256=sha(summary),exit_status=0,original_exit_status=int((original_path/"exit_status.txt").read_text()),
        completion_receipt_root=str(completion_path),evidence=evidence,
        original_receipt_files={p.name:sha(p) for p in original_path.iterdir() if p.is_file()},
        completion_receipt_files={p.name:sha(p) for p in completion_path.iterdir() if p.is_file()},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R21_COLLECTED",sha(summary),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--publish-dir",required=True);p.add_argument("--receipt-root")
    a=p.parse_args();run(a.root,a.publish_dir,a.receipt_root)
