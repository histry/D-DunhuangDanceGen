"""Publish complete results once; all failure and execution identities retained."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit

def qsummary(scores):
    rows=[]
    for case,variant in sorted({(r["case"],r["variant"]) for r in scores}):
        group=[r for r in scores if r["case"]==case and r["variant"]==variant]
        old=[r for r in group if r["t"] in (999,519,119,19)]
        rows.append(dict(case=case,variant=variant,points=len(group),grid_mean=float(np.mean([r["unknown"] for r in group])),
            old4_mean=float(np.mean([r["unknown"] for r in old])),t19=next(r["unknown"] for r in group if r["t"]==19),
            branches={b:{stage:float(np.mean([r["branches"][b][stage]["unknown"] for r in group])) for stage in ("raw","clipped")} for b in ("conditional_music","unconditional_music")}))
    return rows

def compare(cfg,native,q,data):
    lookup={(r["arm"],r["case"],r["variant"]):r for r in native if r["control"]=="correct" and r["role"]=="dev_exposed"}
    qr={a:{(r["case"],r["variant"]):r for r in qsummary(scores)} for a,scores in q.items()}
    sources={r["case"]:r for r in data["records"]};out=[];limits=cfg["qualification"]
    for new,base in (("G1","G0"),("G2","G1")):
        cases=[]
        for case in data["dev_cases"]:
            variants=[]
            for variant in cfg["eval_variants"]:
                a=lookup[new,case,variant];b=lookup[base,case,variant];qa=qr[new][case,variant];qb=qr[base][case,variant]
                rom=lambda r:float(np.mean(r["dynamics"]["joint_range_p95_rad"][1:22]))
                source_rom=float(np.mean(a["source_dynamics"]["joint_range_p95_rad"][1:22]))
                tests=dict(pose=a["pose_error_rad"]<=b["pose_error_rad"]*limits["max_pose_ratio"],
                    root=a["root_rmse_m"]<=b["root_rmse_m"]*limits["max_root_ratio"],
                    activity=a["activity_mps"]>=b["activity_mps"]*limits["min_activity_ratio"],
                    range=rom(a)>=rom(b)*limits["min_activity_ratio"],
                    source_pose=a["pose_error_rad"]<=limits["source_pose_mean_max_rad"],source_root=a["root_rmse_m"]<=limits["source_root_rmse_max_m"],
                    source_activity=a["activity_mps"]>=sources[case]["source_activity_mps"]*limits["min_source_activity_ratio"],
                    source_range=rom(a)>=source_rom*limits["min_source_range_ratio"],
                    internal=a["hard_domains"]["editable"]["p95"]<=b["hard_domains"]["editable"]["p95"]*limits["max_internal_jerk_ratio"],
                    q_grid=qa["grid_mean"]<=qb["grid_mean"]*limits["max_q_ratio"],q_old4=qa["old4_mean"]<=qb["old4_mean"]*limits["max_q_ratio"],
                    sample=a["actual_t19_unknown"]<=b["actual_t19_unknown"]*(1-limits["min_actual_gain"]),
                    boundary=a["hard_domains"]["crossing"]["max"]<=b["hard_domains"]["crossing"]["max"]*(1-limits["min_boundary_gain"]),
                    foot_skate=a["native_audit"]["foot_skate_mps_p95"]<=b["native_audit"]["foot_skate_mps_p95"]*(1-limits["min_foot_skate_gain"]),
                    drift=a["native_audit"]["foot_support_drift_m_p95"]<=b["native_audit"]["foot_support_drift_m_p95"]*limits["max_drift_ratio"])
                tests={k:bool(v) for k,v in tests.items()}
                variants.append(dict(variant=variant,tests=tests,failed=[k for k,v in tests.items() if not v],passed=all(tests.values())))
            cases.append(dict(case=case,source_group=sources[case]["left"]["source_group"],variants=variants,passed=all(r["passed"] for r in variants)))
        n=sum(r["passed"] for r in cases)
        out.append(dict(new=new,base=base,cases=cases,passed_dev_groups=n,research_contrast_passed=n>=limits["required_improved_dev_groups"],used_for_selection=False))
    return out

def run(root,out,receipt_root,failed_receipts=()):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for n in ("r22_summary.json","r22_checks.json","r22_receipt.json"):assert not (out/n).exists(),"Preserve previous public files; use fresh destination"
    receipt_root=Path(receipt_root);assert (receipt_root/"exit_status.txt").read_text().strip()=="0"
    cfg=read(root/"data/audit_protocol.json")["config"];data=read(root/"data/result.json");models={a:read(root/f"models/{a}/result.json") for a in cfg["arms"]}
    native=read(root/"native/result.json")["records"];rollout=read(root/"rollout/result.json");q={a:r["scores"] for a,r in rollout["q_scores"].items()}
    means=[]
    for arm in cfg["arms"]:
        for role in ("train","dev_exposed"):
            for variant in cfg["eval_variants"]:
                rows=[r for r in native if r["arm"]==arm and r["role"]==role and r["variant"]==variant and r["control"]=="correct"]
                means.append(dict(arm=arm,role=role,variant=variant,source_groups=len({r["source_group"] for r in rows}),executions=len(rows),
                    native_passes=sum(r["native_ok"] for r in rows),metrics={k:float(np.mean([r[k] for r in rows])) for k in ("root_rmse_m","pose_error_rad","activity_mps","jerk_p95_mps3","jerk_max_mps3","q_t19_unknown","actual_t19_unknown","actual_over_q")},
                    crossing_jerk_max_mean=float(np.mean([r["hard_domains"]["crossing"]["max"] for r in rows])),
                    editable_jerk_p95_mean=float(np.mean([r["hard_domains"]["editable"]["p95"] for r in rows])),
                    foot_skate_p95_mean=float(np.mean([r["native_audit"]["foot_skate_mps_p95"] for r in rows])),
                    support_drift_p95_mean=float(np.mean([r["native_audit"]["foot_support_drift_m_p95"] for r in rows]))))
    checks=read(root/"independent_checks.json")
    result=dict(schema=cfg["schema"],code_commit=commit(),config=cfg,data_audit=read(root/"data/audit.json"),
        data=dict(records=data["records"],train_groups=data["train_groups"],dev_groups=data["dev_groups"],train_cases=data["train_cases"],dev_cases=data["dev_cases"],train_probe=data["train_probe"]),
        training=models,means=means,native=native,final_q={a:qsummary(rows) for a,rows in q.items()},raw_final_q=q,
        contrasts=compare(cfg,native,q,data),lodge=read(root/"lodge/result.json"),checks=checks,videos=read(root/"videos/protocol.json")["records"],
        lodge_videos=read(root/"videos/lodge_protocol.json"),human_reviewed=False,contact_truth="UNKNOWN",whole_song_generations=0,operational_qualified=False)
    summary=out/"r22_summary.json";write(summary,result);write(out/"r22_checks.json",checks)
    evidence={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*.json")) if p.name in ("result.json","protocol.json","audit.json","audit_protocol.json","independent_checks.json")}
    receipts=[receipt_root]+[Path(p) for p in failed_receipts]
    write(out/"r22_receipt.json",dict(collection_commit=commit(),model_execution_commits=sorted({read(root/f"models/{a}/protocol.json")["code_commit"] for a in cfg["arms"]}),
        data_execution_commit=read(root/"data/audit_protocol.json")["code_commit"],lodge_execution_commit=read(root/"lodge/protocol.json")["code_commit"],
        completion_launcher_commit=(receipt_root/"execution_commit.txt").read_text().strip(),summary_sha256=sha(summary),evidence=evidence,
        receipts={str(folder):{p.name:sha(p) for p in folder.iterdir() if p.is_file()} for folder in receipts},
        exit_status=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R22_COLLECTED",sha(summary),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--publish-dir",required=True);p.add_argument("--receipt-root",required=True)
    p.add_argument("--failed-receipts",nargs="*",default=[]);a=p.parse_args();run(a.root,a.publish_dir,a.receipt_root,a.failed_receipts)
