"""Compact published evidence; full traces and per-frame data stay under outputs."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);a=read(root/"audit/result.json");b=read(root/"analysis/result.json");checks=read(root/"independent_checks.json")
    receipts={}
    for label in ("audit","trace","post","final"):
        folder=Path(str(root)+"_"+label+"_receipts");assert (folder/"exit_status.txt").read_text().strip()=="0"
        receipts[label]=dict(code_commit=(folder/"execution_commit.txt").read_text().strip(),files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    groups={}
    for arm in ("original","zero","C1","C2"):
        rows=[r for r in a["records"] if r["arm"]==arm];seams=[s for r in rows for s in r["seams"]]
        groups[arm]=dict(cases=len(rows),native_passes=sum(r["native_ok"] for r in rows),root_step_mean_m=float(np.mean([s["root_step_m"] for s in seams])),
            root_rotation_step_mean_rad=float(np.mean([s["root_rotation_step_rad"] for s in seams])),feet_step_mean_m=float(np.mean([s["foot_step_m"] for s in seams])),
            decomposition_component_rms_m={k:float(np.mean([s["components"][k]["rms_m"] for s in seams])) for k in ("root_translation","root_rotation","local_pose")})
    conditions={}
    for population in ("val","r8"):
        for arm in ("A0","A1","A2","A3","A4","A5"):
            rows=[r for r in b["case_means"] if r["arm"]==arm and r["case"].startswith("val_")==(population=="val")]
            conditions[population+"/"+arm]=dict(cases=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in ("mse","root_rmse_m","pose_error_rad","feet_rmse_m","boundary_root_rmse_m","boundary_feet_rmse_m","root_response_rms_m","pose_response_rad")})
    evidence={};stages={}
    for phase in ("audit","audit_replay","trace","trace_replay","analysis","analysis_replay","analysis_cached","inspector"):
        protocol=read(root/phase/"protocol.json");stages[phase]=protocol["code_commit"]
        for file in ("protocol.json","result.json","checks.json"):
            p=root/phase/file
            if p.is_file():evidence[phase+"/"+file]=sha(p)
    local_qa=root/"inspector/local_static_checks.json"
    inspector_qa=read(local_qa) if local_qa.exists() else dict(browser_visual_qa="NOT_RUN")
    if local_qa.exists():evidence["inspector/local_static_checks.json"]=sha(local_qa)
    curves=[]
    for name in dict.fromkeys(r["case"] for r in b["records"]):
        rows=[r for r in b["records"] if r["case"]==name and r["arm"]=="A0"]
        curves.append(dict(case=name,first=rows[0],last=rows[-1],root_error_nonmonotone_steps=sum(v["root_rmse_m"]>u["root_rmse_m"] for u,v in zip(rows,rows[1:]))))
    p1=read(root/"trace/protocol.json");traces=[dict(case=x["case"],trace=x["trace"],trace_info=x["trace_info"],reference=x["reference"]) for x in p1["bindings"]]
    summary=dict(schema="r11_frozen_state_and_shared_trajectory_evidence",baseline="4496e435c4f8d4333d4d888b67be2875bf1d8e2b",
        config=read(root/"audit/protocol.json")["config"],state_groups=groups,oracles=a["oracles"],qualification=b["qualification"],conditions=conditions,
        case_condition_means=b["case_means"],inspector_validation=inspector_qa,a0_first_last=curves,traces=traces,checks=checks,stage_commits=stages,evidence=evidence,receipts=receipts,
        optimizer_steps=0,new_neural_generations=16,new_counterfactual_complete_rollouts=0,shared_state_predictions=4800,new_whole_song_generations=0,
        human_reviewed=False,operational_qualified=False,
        interpretation=["seam decomposition is numerical, not causal","A0 drives sampling; other arms are fixed-state forward interventions","soft incremental effect does not identify the entire hard-context pathway","single-side masks and fixed gauge replacements can be outside original training support","four Val and four exposed R8 cases; 400 timesteps are not independent samples","Oracle floating geometry pass is not byte identity or physical admission"])
    write("reports/r11_summary.json",summary);write("reports/r11_checks.json",checks)
    write("reports/r11_receipt.json",dict(code_commit=commit(),summary_sha256=sha("reports/r11_summary.json"),stage_commits=stages,evidence=evidence,receipts=receipts,
        formal_changed_paths=checks["formal_changed_paths"],temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,optimizer_steps=0,new_whole_song_generations=0,operational_qualified=False))
    print("R11_SEALED",sha("reports/r11_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
