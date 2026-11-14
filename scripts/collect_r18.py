"""Collect evidence after successful tmux completion; preserve original failed roots."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);receipt=Path(str(root)+"_receipts")
    assert (receipt/"exit_status.txt").read_text().strip()=="0"
    cfg=read(root/"p0/protocol.json")["config"];checks=read(root/("independent_checks_after_sync.json" if (root/"independent_checks_after_sync.json").exists() else "independent_checks.json"))
    native=read(root/"native/result.json");gate=read(root/"tiny_gate.json");p0=read(root/"p0/result.json")
    attention=[]
    for mode in ("self","full","unique"):
        for task in cfg["tasks"]:
            rows=[r for r in p0["records"] if r["mode"]==mode and r["task"]==task and r["control"]=="correct"]
            attention.append(dict(mode=mode,task=task,cases=len(rows),
                overlap_mass=float(np.mean([r["attention"]["overlap_mass"] for r in rows])),
                unique_mass=float(np.mean([r["attention"]["unique_mass"] for r in rows])),
                unknown_mse=float(np.mean([r["unknown_mse"] for r in rows]))))
    models={}
    for phase in ["tiny"]+(["main"] if checks["main_executed"] else []):
        models[phase]={}
        for mode in ("self","unique"):
            m=read(root/phase/mode/"result.json")
            models[phase][mode]={k:m[k] for k in ("checkpoint","motion_parameters","head_parameters","optimizer_steps",
                "initial_motion","final_motion","initial_head","final_head","frozen_digest","task_counts","fk_parity_max_m")}
            models[phase][mode].update(first=m["history"][0],last=m["history"][-1])
    arms=[]
    for phase in ["tiny"]+(["main"] if checks["main_executed"] else []):
        for task in cfg["tasks"]:
            for arm in ("D0","D1","D2"):
                rows=[r for r in native["records"] if r["phase"]==phase and r["task"]==task and r["arm"]==arm and r["control"]=="correct"]
                arms.append(dict(phase=phase,task=task,arm=arm,cases=len(rows),native_passes=sum(r["native_ok"] for r in rows),
                    means={k:float(np.mean([r[k] for r in rows])) for k in ("root_rmse_m","foot_rmse_m","jerk_max_mps3",
                        "jerk_p95_mps3","activity_mps","pose_error_rad","raw_unknown_mse")}))
    evidence={}
    for p in sorted(root.rglob("*.json")):
        if "_replay" not in str(p.relative_to(root)) and p.name in ("protocol.json","result.json","tiny_gate.json","independent_checks.json"):
            evidence[str(p.relative_to(root))]=sha(p)
    receipt_files={p.name:sha(p) for p in sorted(receipt.iterdir()) if p.is_file()}
    diagnostic_receipt=Path(str(root)+"_diagnostic_receipts")
    assert (diagnostic_receipt/"exit_status.txt").read_text().strip()=="0"
    diagnostic_files={p.name:sha(p) for p in sorted(diagnostic_receipt.iterdir()) if p.is_file()}
    result=dict(schema="r18_task_denoising_evidence_v1",code_commit=commit(),config=cfg,attention_summary=attention,
        p0_records=p0["records"],training=models,tiny_gate=gate,arms=arms,
        objective_diagnostic=read(root/"objective_diagnostic/result.json") if (root/"objective_diagnostic/result.json").exists() else None,
        records=native["records"],qualification=native["qualification"],checks=checks,
        videos=read(root/"videos/protocol.json")["records"],human_reviewed=False,semantic_event_truth="UNKNOWN",
        limitations=["Tiny is two Train sequences, not generalization","Exposed Eval may be foundation-pretrained",
            "Progressive training still uses real past teacher forcing","Attention mass is not useful-content causality",
            "Paired reconstruction does not establish a unique correct FREE dance",
            "No D3, ChangE full-style or whole-song qualification"],new_whole_song_generations=0,operational_qualified=False)
    summary=Path("reports/r18_summary.json");write(summary,result);write("reports/r18_checks.json",checks)
    write("reports/r18_receipt.json",dict(code_commit=commit(),summary_sha256=sha(summary),evidence=evidence,
        execution_commit=(receipt/"execution_commit.txt").read_text().strip(),exit_status=0,receipt_files=receipt_files,
        diagnostic_execution_commit=(diagnostic_receipt/"execution_commit.txt").read_text().strip(),
        diagnostic_exit_status=0,diagnostic_receipt_files=diagnostic_files,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,new_whole_song_generations=0,operational_qualified=False))
    print("R18_COLLECTED",sha(summary),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
