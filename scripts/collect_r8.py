"""Seal source/bridge evidence without promoting failed or proxy outcomes."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import sha,write
from dunhuang.research.source_capacity.common import read

def run(root):
    root=Path(root);receipt=Path(str(root)+"_receipts")
    assert (receipt/"exit_status.txt").read_text().strip()=="0"
    single=read(root/"single/result.json");bridge=read(root/"bridge/result.json");checks=read(root/"independent_checks.json");groups={}
    for arm in ("baseline","S0","S1","S2","S3"):
        rows=[r for r in single["records"] if r["arm"]==arm];measures=[r["source"] for r in rows if r["source"] is not None]
        groups[arm]=dict(generations=len(rows),native_passes=sum(r["native_ok"] for r in rows))
        for key in ("mean_source_rad","phase_pose_rad","wrist_rotation_rad","trunk_rotation_rad","wrist_elbow_root_frame_m","activity_ratio","generated_hold_seconds","source_hold_seconds"):
            values=[m[key] for m in measures];groups[arm][key]=dict(mean=float(np.mean(values)),minimum=float(np.min(values)),maximum=float(np.max(values))) if values else None
        for key in ("joint_jerk_mps3_p95","foot_skate_mps_p95"):
            values=[r["native_audit"][key] for r in rows];groups[arm][key]=dict(mean=float(np.mean(values)),minimum=float(np.min(values)),maximum=float(np.max(values)))
    evidence={};commits={}
    phases=["style_audit","boundary_audit","prepared","single","single_replay","bridge","bridge_replay","whole","whole_replay","joint","joint_replay","videos"]
    if (root/"phase_proxy/result.json").is_file():phases.append("phase_proxy")
    for phase in phases:
        commits[phase]=read(root/phase/"protocol.json")["code_commit"]
        for name in ("protocol.json","result.json","replay.json","pairs.json","music.json","candidates.json"):
            file=root/phase/name
            if file.is_file():evidence[phase+"/"+name]=sha(file)
    summary=dict(schema="r8_reperformance_bridge_summary",baseline="2a7941859c381a0308b7288797b4f443a7b6c711",
        groups=groups,source_qualification=single["qualification"],music_response=single["music_response"],single_gate_passed=single["single_event_gate_passed"],
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
    write("reports/r8_receipt.json",sealed)
    print("R8_SEALED",sealed["summary_sha256"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();run(a.out)
