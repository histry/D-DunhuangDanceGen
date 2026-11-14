"""Compact public evidence; all trial ledgers and Native arrays remain in outputs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import sha,write
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);checks=read(root/"independent_checks.json")
    pre=read(root/"preflight/result.json");p=read(root/"probes/result.json");n=read(root/"native/result.json")
    evidence={phase+"/"+f:sha(root/phase/f) for phase in ("preflight","preflight_replay","probes","probes_replay","native","native_replay") for f in ("protocol.json","result.json")}
    receipts={}
    for path in (Path(str(root)+"_receipts"),Path(str(root)+"_final_receipts")):
        assert (path/"exit_status.txt").read_text().strip()=="0"
        receipts[path.name]=dict(code_commit=(path/"execution_commit.txt").read_text().strip(),files={f.name:sha(f) for f in path.iterdir() if f.is_file()})
    failed=Path("outputs/root_pose_r15_01_receipts")
    if failed.exists():
        receipts[failed.name]=dict(code_commit=(failed/"execution_commit.txt").read_text().strip(),
            exit_status=(failed/"exit_status.txt").read_text().strip(),files={f.name:sha(f) for f in failed.iterdir() if f.is_file()},
            reason="Pre-GPU absolute/relative manifest path mismatch, corrected without changing scientific population or budgets")
    arms={}
    keys=("root_rmse_m","feet_rmse_m","pose_error_rad","root_orientation_rad","activity_mps")
    for arm in ("J0","J1","J2","J3","J4"):
        rr=[r for r in n["states"] if r["arm"]==arm];trials=[r for r in n["trials"] if r["arm"]==arm]
        arms[arm]=dict(states=len(rr),accepted_states=sum(r["accepted"] for r in rr),
            qualified_groups=n["qualification"]["qualified_groups"].get(arm,[]),
            selected_with_fallback={k:float(np.mean([r["metrics"][k] for r in rr])) for k in keys},
            selected_with_fallback_native_passes=sum(r["metrics"]["native_ok"] for r in rr),
            finite_trials=len(trials),trial_native_passes=sum(r["metrics"]["native_ok"] for r in trials),
            protected_trials=sum(r["eligible"] for r in trials),
            rejection_counts={key:sum(not r["protection"][key] for r in trials) for key in trials[0]["protection"]} if trials else {},
            trust_rejected=sum(not r["response"]["within_radius"] for r in trials))
    result=dict(schema="r15_root_pose_capacity_evidence_v1",baseline="bfb269dfb5714a2c012d10d57f4511509c82a621",
        config=read(root/"preflight/protocol.json")["config"],necessary_bounds=pre["records"],
        frozen_final_responsibility=pre["frozen_final_responsibility"],arms=arms,qualification=n["qualification"],checks=checks,
        states=n["states"],group_results=n["groups"],evidence=evidence,receipts=receipts,
        numerical_valid=checks["numeric_gradients_passed"],oracle_capacity_passed=checks["oracle_capacity_passed"],
        new_population_training_admitted=False,optimizer_steps=0,analytical_heads_fitted=0,new_full_window_generations=0,
        new_whole_song_generations=0,human_reviewed=False,semantic_event_truth="UNKNOWN",operational_qualified=False,
        limitations=["Train oracle unknown targets are not inference conditions","Single-state x_t probes are not full guided rollouts",
            "Exposed eight Train groups do not establish independent generalization","Finite trust search failure is not a global infeasibility proof",
            "Local mechanism and complete Native qualification are separate"])
    write("reports/r15_summary.json",result);write("reports/r15_checks.json",checks)
    write("reports/r15_receipt.json",dict(code_commit=commit(),summary_sha256=sha("reports/r15_summary.json"),evidence=evidence,receipts=receipts,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,new_whole_song_generations=0,operational_qualified=False))
    print("R15_SEALED",sha("reports/r15_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
