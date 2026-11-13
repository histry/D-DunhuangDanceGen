"""Seal source/bridge evidence without promoting failed or proxy outcomes."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import sha,write
from dunhuang.research.source_capacity.common import read

def source_criteria(single,cfg):
    """Explain the already frozen decision; never changes its gate or population."""
    rows=[];qual={(r["pair"],r["music"],r["source_label"]):r for r in single["qualification"]["records"]}
    for row in single["records"]:
        if row["arm"]!="S3":continue
        m=row["source"];q=qual[(row["pair"],row["music"],row["source_label"])]
        criteria={}
        for metric,threshold in (("mean_source_rad","source_mean_rad"),("phase_pose_rad","key_pose_mean_rad"),("root_relative_turn_rad","root_turn_mean_rad"),("wrist_rotation_rad","wrist_rotation_rad"),("trunk_rotation_rad","trunk_rotation_rad"),("wrist_elbow_root_frame_m","wrist_elbow_root_frame_m")):
            criteria[metric]=m[metric]<=cfg[threshold]
        criteria["activity_ratio"]=cfg["activity_ratio"][0]<=m["activity_ratio"]<=cfg["activity_ratio"][1]
        criteria["hold_proxy_coverage"]=None if m["hold_proxy_coverage"] is None else m["hold_proxy_coverage"]>=cfg["hold_proxy_coverage"]
        criteria["own_gain_rad"]=q["own_gain_rad"]>=cfg["own_gain_rad"]
        criteria["directional_gain_rad"]=q["directional_gain_rad"]>=cfg["relative_gain_rad"]
        assert all(v is not False for v in criteria.values())==q["source_ok"]
        rows.append(dict(pair=row["pair"],cohort=row["cohort"],music=row["music"],source_label=row["source_label"],criteria=criteria,native_ok=row["native_ok"]))
    return dict(records=rows,failure_counts={key:sum(r["criteria"][key] is False for r in rows) for key in rows[0]["criteria"]},
        not_applicable_counts={key:sum(r["criteria"][key] is None for r in rows) for key in rows[0]["criteria"]},scope="EXPLANATION_OF_REGISTERED_GATE")

def run(root):
    root=Path(root);receipt=Path(str(root)+"_receipts")
    assert (receipt/"exit_status.txt").read_text().strip()=="0"
    single=read(root/"single/result.json");bridge=read(root/"bridge/result.json");checks=read(root/"independent_checks.json");groups={}
    for arm in ("baseline","S0","S1","S2","S3"):
        rows=[r for r in single["records"] if r["arm"]==arm];measures=[r["source"] for r in rows if r["source"] is not None]
        if arm=="baseline":measures=[m for r in rows for m in r["sources"].values()]
        groups[arm]=dict(generations=len(rows),native_passes=sum(r["native_ok"] for r in rows),native_failure_reasons={})
        for r in rows:
            for reason in r["native_reasons"]:groups[arm]["native_failure_reasons"][reason]=groups[arm]["native_failure_reasons"].get(reason,0)+1
        for key in ("mean_source_rad","phase_pose_rad","wrist_rotation_rad","trunk_rotation_rad","wrist_elbow_root_frame_m","activity_ratio","generated_hold_seconds","source_hold_seconds","root_relative_turn_rad"):
            values=[m[key] for m in measures];groups[arm][key]=dict(mean=float(np.mean(values)),minimum=float(np.min(values)),maximum=float(np.max(values))) if values else None
        for key in ("joint_jerk_mps3_p95","foot_skate_mps_p95"):
            values=[r["native_audit"][key] for r in rows];groups[arm][key]=dict(mean=float(np.mean(values)),minimum=float(np.min(values)),maximum=float(np.max(values)))
    evidence={};commits={}
    phases=["style_audit","boundary_audit","prepared","single","single_replay","bridge","bridge_replay","whole","whole_replay","joint","joint_replay","videos"]
    if (root/"phase_proxy/result.json").is_file():phases+=["phase_proxy","phase_proxy_replay"]
    if (root/"videos_wrist_axes/protocol.json").is_file():phases.append("videos_wrist_axes")
    for phase in phases:
        commits[phase]=read(root/phase/"protocol.json")["code_commit"]
        for name in ("protocol.json","result.json","replay.json","pairs.json","music.json","candidates.json"):
            file=root/phase/name
            if file.is_file():evidence[phase+"/"+name]=sha(file)
    summary=dict(schema="r8_reperformance_bridge_summary",baseline="2a7941859c381a0308b7288797b4f443a7b6c711",
        groups=groups,source_criterion_breakdown=source_criteria(single,read(root/"single/protocol.json")["config"]["gate"]),source_qualification=single["qualification"],music_response=single["music_response"],single_gate_passed=single["single_event_gate_passed"],
        bridge=bridge,boundary_audit=read(root/"boundary_audit/result.json"),style_three_way_cases=len(read(root/"style_audit/result.json")["records"]),
        whole=read(root/"whole/result.json"),joint=read(root/"joint/result.json"),evidence=evidence,independent_checks=checks,
        neural_weight_updates=0,human_reviewed=False,cultural_authenticity="UNKNOWN",operational_qualified=False)
    if "phase_proxy" in phases:summary["phase_proxy"]=read(root/"phase_proxy/result.json")
    write("reports/r8_summary.json",summary);write("reports/r8_checks.json",checks)
    sealed=dict(schema="r8_sealed_receipt",execution_commit=(receipt/"execution_commit.txt").read_text().strip(),stage_execution_commits=commits,
        verifier_commit=checks["code_commit"],summary_sha256=sha("reports/r8_summary.json"),root=str(root.resolve()),exit_status=0,
        receipts={p.name:sha(p) for p in receipt.iterdir() if p.is_file()},started_utc=(receipt/"started_utc.txt").read_text().strip(),
        finished_utc=(receipt/"finished_utc.txt").read_text().strip(),previous_failed_roots_retained=True,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        no_formal_sampling_adapter_native_changes=not checks["formal_changed_paths"],neural_weight_updates=0,operational_qualified=False)
    post=Path(str(root)+"_post_receipts")
    if post.is_dir():
        assert (post/"exit_status.txt").read_text().strip()=="0"
        sealed["postprocess_receipts"]={p.name:sha(p) for p in post.iterdir() if p.is_file()}
        sealed["postprocess_execution_commit"]=(post/"execution_commit.txt").read_text().strip()
    write("reports/r8_receipt.json",sealed)
    print("R8_SEALED",sealed["summary_sha256"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();run(a.out)
