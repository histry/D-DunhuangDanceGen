"""Publish scientific summaries without committing weights, cached states or videos."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);receipt=Path(str(root)+"_receipts")
    assert (receipt/"exit_status.txt").read_text().strip()=="0"
    cfg=read(root/"interface/protocol.json")["config"];checks=read(root/"independent_checks.json")
    arms=["S0","S1","A0","A1","A2"]+(["A3"] if checks["A3_executed"] else [])
    models={}
    for arm in arms:
        r=read(root/"models"/arm/"result.json")
        models[arm]={k:r[k] for k in ("layout","checkpoint","initial","final_selected","final_head","initial_scores","scores",
            "update_observations","optimizer_steps","motion_parameters","head_parameters","null_adaptation","wall_seconds",
            "peak_allocated_cuda_bytes","predictor_forward_calls","motion_token_evaluations")}
        models[arm].update(first=r["history"][0],last=r["history"][-1],
            cache_bindings=[{k:v for k,v in b.items() if k!="receipt"} for b in r["cache_bindings"]])
    native={part:read(root/(part+"_native")/"result.json")["records"] for part in ("p0","structure","final")}
    means=[]
    for role in ("train","dev_exposed"):
        for arm in ["S0","S1","A0","A1","A2"]+(["A3"] if checks["A3_executed"] else []):
            rows=[r for r in native["final"] if r["arm"]==arm and r["role"]==role]
            means.append(dict(role=role,arm=arm,executions=len(rows),source_groups=len({r["case"] for r in rows}),
                native_passes=sum(r["native_ok"] for r in rows),
                metrics={k:float(np.mean([r[k] for r in rows])) for k in ("root_rmse_m","foot_rmse_m","jerk_max_mps3","jerk_p95_mps3",
                    "activity_mps","pose_error_rad","root_orientation_error_rad")},
                interior_jerk_p95_mean=float(np.mean([r["temporal_domains"]["interior"]["p95"] for r in rows])),
                actual_unknown_mean=float(np.mean([r["raw_losses"]["unknown"] for r in rows]))))
    stages=[]
    for part in ("p0","structure","final"):
        for b in read(root/(part+"_rollout")/"result.json")["bindings"]:
            if b["case"]!=0 or b["seed"]!=0 or b["control"]!="correct":continue
            receipt=read(b["sampler"]["path"])
            selected=[r for r in receipt["records"] if r["t"] in cfg["stage_snapshots"] and "stages" in r]
            if selected:stages.append(dict(part=part,arm=b["arm"],layout=b["layout"],records=selected))
    result=dict(schema=cfg["schema"],code_commit=commit(),config=cfg,interface=read(root/"interface/result.json"),
        structure_gate=read(root/"structure_gate.json"),cache_gate=read(root/"cache_gate.json"),
        temporal_calibration=read(root/"temporal_calibration/result.json"),training=models,means=means,
        native=native,stage_readouts=stages,checks=checks,videos=read(root/"videos/protocol.json")["records"],
        human_reviewed=False,semantic_event_truth="UNKNOWN",whole_song_generations=0,operational_qualified=False,
        limitations=["Train2; exposed DEV2 cannot be called blind test","True future endpoint motion is supplied in INBETWEEN",
            "Joint layout changes attention/music span and frozen null embedding; it is a structural package",
            "A3 cached x_t states are not generated history and are not q-distributed",
            "Stage associations alone do not prove unique causation","Fixed observation conflicts do not disappear by training"])
    evidence={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*.json")) if p.name in
        ("protocol.json","result.json","structure_gate.json","cache_gate.json","independent_checks.json")}
    p=Path("reports/r20_summary.json");write(p,result);write("reports/r20_checks.json",checks)
    write("reports/r20_receipt.json",dict(code_commit=commit(),summary_sha256=sha(p),evidence=evidence,
        execution_commit=(receipt/"execution_commit.txt").read_text().strip(),exit_status=0,
        receipt_files={p.name:sha(p) for p in receipt.iterdir() if p.is_file()},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R20_COLLECTED",sha(p),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
