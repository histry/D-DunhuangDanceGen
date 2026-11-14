"""Publish compact R12 results and hashes; bulky arrays/checkpoints remain outputs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);checks=read(root/"independent_checks.json");assert checks["optimizer_steps"]==0
    capacity=read(root/"capacity/result.json");analysis=read(root/"analysis/result.json");cache=read(root/"cache/result.json")
    receipts={};evidence={};models={}
    for name in ("capacity","train","generate","final","chain"):
        p=Path(str(root)+"_"+name+"_receipts");assert (p/"exit_status.txt").read_text().strip()=="0"
        receipts[name]=dict(code_commit=(p/"execution_commit.txt").read_text().strip(),files={f.name:sha(f) for f in p.iterdir() if f.is_file()})
    for phase in ("capacity","capacity_replay","cache","cache_replay","B3","B2","B3_replay","B2_replay","generation","generation_replay","analysis","analysis_replay","diagnosis","diagnosis_replay","chain","chain_replay"):
        for name in ("protocol.json","result.json","interface_gate.json"):
            p=root/phase/name
            if p.exists():evidence[phase+"/"+name]=sha(p)
    for arm in ("B3","B2"):
        r=read(root/arm/"result.json")
        models[arm]=dict(checkpoint=r["checkpoint"],parameters=r["parameters"],optimizer_steps=r["optimizer_steps"],monitor=r["monitor"],
            first_loss=r["history"][0],last_loss=r["history"][-1],term_gradients=[v for v in r["history"] if v["term_head_gradient"]])
    groups={}
    for arm in ("original","B0","B1","B3","B2"):
        rs=[r for r in analysis["records"] if r["arm"]==arm]
        groups[arm]=dict(cases=len(rs),native_passes=sum(r["native_ok"] for r in rs),roi_native_passes=sum(r["roi_native_ok"] for r in rs),
            **{k:float(np.mean([r[k] for r in rs])) for k in ("root_rmse_m","pose_error_rad","feet_rmse_m")},
            root_step_m=float(np.mean([v["root_step_m"] for r in rs for v in r["seams"]])),
            feet_step_m=float(np.mean([v["foot_step_m"] for r in rs for v in r["seams"]])),
            raw_known_root_error_m=float(np.mean([r["raw_known_root_error_m"] for r in rs])) if arm in ("B0","B3","B2") else None)
    summary=dict(schema="r12_explicit_motion_bridge_evidence",baseline="75868e812e31adf2173e4f62c564c56c4c13d219",
        selected_site=capacity["selected_site"],capacity_scores=capacity["scores"],config=read(root/"capacity/protocol.json")["config"],
        training_windows=cache["statistics_training_windows"],training_groups=len(cache["statistics_groups"]),monitor_groups=len(cache["monitor_groups"]),
        models=models,groups=groups,case_results=analysis["records"],control_results=analysis["controls"],qualification=analysis["qualification"],
        checks=checks,evidence=evidence,receipts=receipts,diagnosis=read(root/"diagnosis/result.json"),root_stage_chain=read(root/"chain/result.json"),optimizer_steps=800,
        new_neural_generations=24+24+12+12,new_whole_song_generations=0,human_reviewed=False,operational_qualified=False,
        limitations=["P0 is one matched finite feature step, not global entry incapacity","B0 cache is off-policy for trained Bridge; 12 windows and four saved steps only",
        "same-state soft controls do not measure independent counterfactual rollout quality","4 exposed Val and 4 exposed R8 cases; no new independent test",
        "Bridge study is not proof of ChangE event reinterpretation or cultural fidelity"])
    write("reports/r12_summary.json",summary);write("reports/r12_checks.json",checks)
    write("reports/r12_receipt.json",dict(code_commit=commit(),summary_sha256=sha("reports/r12_summary.json"),receipts=receipts,evidence=evidence,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,new_whole_song_generations=0,operational_qualified=False))
    print("R12_SEALED",sha("reports/r12_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
