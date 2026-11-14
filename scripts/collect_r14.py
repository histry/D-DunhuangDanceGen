"""Compact public R14 evidence; large ledgers and model arrays stay in outputs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit

def mean(rr,key):return float(np.mean([r[key] for r in rr]))

def run(root):
    root=Path(root);check=read(root/"independent_checks.json");a=read(root/"audit/result.json")
    c=read(root/"conflict/result.json");r=read(root/"readout/result.json");n=read(root/"capacity_native/result.json")
    turn=read(root/"turn/result.json");capacity=read(root/"capacity/result.json");receipts={};evidence={}
    for stage in ("audit","readout","capacity","final"):
        p=Path(str(root)+"_"+stage+"_receipts");assert (p/"exit_status.txt").read_text().strip()=="0"
        receipts[stage]=dict(code_commit=(p/"execution_commit.txt").read_text().strip(),
            files={f.name:sha(f) for f in p.iterdir() if f.is_file()})
    for phase in ("audit","audit_replay","conflict","conflict_replay","readout","readout_replay","turn","turn_replay","capacity","capacity_replay","capacity_native","capacity_native_replay"):
        for name in ("protocol.json","result.json"):
            evidence[phase+"/"+name]=sha(root/phase/name)
    cases=[]
    for i in range(4):
        for arm in ("MCD","R12B3"):
            rows=[v for v in a["records"] if v["case"]==i and v["arm"]==arm];last=rows[-1]
            angles=[v for v in turn["records"] if v["case"]==i and v["arm"]==arm]
            cases.append(dict(case=i,arm=arm,states=len(rows),final_raw_left=last["raw_left"],final_raw_right=last["raw_right"],
                all_step_raw_left={k:float(np.mean([v["raw_left"][k] for v in rows])) for k in last["raw_left"]},
                all_step_raw_right={k:float(np.mean([v["raw_right"][k] for v in rows])) for k in last["raw_right"]},
                final_root_so3=angles[-1],final_graft_seams=last["graft_seams"]))
    family={}
    for f in ("A0","A1","A2","A3","A4"):
        rr=[x for x in r["group_scores"] if x["family"]==f]
        family[f]=dict(root_rmse_group_mean_m=mean(rr,"root_rmse_m"),orientation_group_mean_rad=mean(rr,"root_orientation_rad"),
            group_scores=rr,qualification=r["qualification"].get(f))
    strata=[]
    for f in family:
        for axis,label in enumerate(("root_displacement","relative_turn","endpoint_acceleration")):
            for level in range(3):
                rr=[x for x in r["records"] if x["family"]==f and x["difficulty_bins"][axis]==level]
                strata.append(dict(family=f,axis=label,level=level,windows=len(rr),
                    root_rmse_m=mean(rr,"root_rmse_m") if rr else None,scope="descriptive correlated windows; cuts fitted only on fit groups"))
    conflict=dict(states=len(c["records"]),parameter_dimension=c["selected_parameter_dimension"],
        positive_common_first_order_states=sum(min(x["normalized_first_order_descent"])>0 for x in c["records"]),
        jointly_nonregressive_finite_trials=sum(v["all_nonregression"] for x in c["records"] for v in x["trials"]),
        total_finite_trials=sum(len(x["trials"]) for x in c["records"]),
        worsening_counts={k:sum(v["losses"][k]>x["losses"][k]*(1+1e-5)+1e-10 for x in c["records"] for v in x["trials"]) for k in c["records"][0]["losses"]},
        limitations="Only two pre-registered finite RMS budgets; failure does not prove absence of smaller useful steps or global capacity")
    derivative=dict(probes=len(capacity["derivatives"]),max_relative_error=max(x["derivative_relative_error"] for x in capacity["derivatives"]),
        numeric_passed=n["numeric_gradient_passed"])
    result=dict(schema="r14_boundary_state_evidence_v1",baseline="e0b7100b9097e7f68b7639742298326990e94f00",
        config=read(root/"audit/protocol.json")["config"],capacity_config=read(root/"capacity/protocol.json")["config"],
        cases=cases,necessary_bounds=[{k:v for k,v in b.items() if k!="row"} for b in a["necessary_bounds"]],
        necessary_excluded_cases=a["necessary_excluded_cases"],unknown_cases=a["unknown_cases"],
        gradient_conflict=conflict,readout=family,difficulty_strata=strata,selected_family=r["selected_family"],
        representation_gate_passed=r["reference_capacity_study_admitted"],
        fit=dict(windows=r["fit_windows"],groups=r["fit_groups"]),head_holdout=dict(windows=r["holdout_windows"],groups=r["holdout_groups"]),
        parameters_per_head=r["parameters_per_head"],analytical_heads_fitted=5,neural_optimizer_steps=0,
        reference_capacity=n,reference_derivatives=derivative,checks=check,evidence=evidence,receipts=receipts,
        baseline_window_generations=8,guided_full_window_generations=0,new_whole_song_generations=0,human_reviewed=False,
        semantic_event_truth="UNKNOWN",operational_qualified=False,new_population_training_admitted=False,
        limitations=["Boundary states are deterministic observed-motion features, not new information",
            "Head-heldout groups may be exposed to R13 and foundation training",
            "Single-state capacity probes are not guided full-chain evidence",
            "Native necessary exclusions apply only to unchanged edit[60,90) and global max jerk",
            "No ChangE style or full-song qualification"])
    write("reports/r14_summary.json",result);write("reports/r14_checks.json",check)
    write("reports/r14_receipt.json",dict(code_commit=commit(),summary_sha256=sha("reports/r14_summary.json"),
        evidence=evidence,receipts=receipts,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        new_whole_song_generations=0,operational_qualified=False))
    print("R14_SEALED",sha("reports/r14_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
