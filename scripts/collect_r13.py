"""Public compact report, lineage and receipts; training histories stay under outputs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import sha,write
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);check=read(root/"independent_checks.json");stages=check["stages"];evidence={};receipts={};training={};groups={}
    for stage in ["audit","tiny"]+(["main"] if "main" in stages else [])+["final"]:
        p=Path(str(root)+"_"+stage+"_receipts");assert (p/"exit_status.txt").read_text().strip()=="0"
        receipts[stage]=dict(code_commit=(p/"execution_commit.txt").read_text().strip(),files={f.name:sha(f) for f in p.iterdir() if f.is_file()})
    phases=["official","official_replay","teacher","teacher_replay","teacher_native","teacher_native_replay","native_detail","native_detail_replay","inspection","inspection_replay"]
    for stage in stages:
        phases.extend([stage,stage+"_replay",stage+"_rollout",stage+"_rollout_replay",stage+"_native",stage+"_native_replay"])
        r=read(root/stage/"result.json");training[stage]=dict(checkpoint=r["checkpoint"],parameters=r["parameters"],optimizer_steps=r["optimizer_steps"],
            initial_digest=r["initial_digest"],final_digest=r["final_digest"],frozen_digest=r["frozen_digest"],ledger_sha256=sha(root/stage/"result.json"),
            first=r["history"][0],last=r["history"][-1],self_policy_snapshots=r["self_snapshots"],
            training_population_windows=len(read(root/stage/"protocol.json")["fit_population"]))
        nr=read(root/(stage+"_native")/"result.json");g={}
        for arm in ("MCD","R12B3"):
            rows=[r for r in nr["records"] if r["arm"]==arm and r["driver"]=="correct"]
            g[arm]=dict(cases=len(rows),native_passes=sum(r["native_ok"] for r in rows),
                root_step_mean_m=float(np.mean([j["root_step_m"] for r in rows for j in r["joins"]])),
                foot_step_max_mean_m=float(np.mean([j["foot_step_max_m"] for r in rows for j in r["joins"]])),
                pose_error_rad=float(np.mean([r["pose_error_rad"] for r in rows])),
                raw_known_root_rmse_m=float(np.mean([r["raw_final"]["known_root_rmse_m"] for r in rows])))
        groups[stage]=g
    for phase in phases:
        for file in ("protocol.json","result.json","interface.json"):
            p=root/phase/file
            if p.exists():evidence[phase+"/"+file]=sha(p)
    main=None
    if "main" in stages:
        # Main's four fit-window q probes are diagnostics, NOT the P1 admission test.
        oldmain=read(root/"main_native/result.json");main=dict(oldmain)
        main["fit_probe_diagnostic"]={key:main.pop(key) for key in ("q_records","tiny_cases","tiny_fit_passed_cases","tiny_training_admitted","context_control_wins")}
        main["fit_probe_diagnostic"]["scope"]="Four already-fitted windows; not heldout evaluation and not the P1 gate"
        main["heldout_scope"]="Four adaptation-heldout Train music groups; foundation-pretraining exposure not independently excluded"
        wins={}
        rows=[r for r in main["records"] if r["arm"]=="MCD" and r["driver"]=="correct"]
        for control in ("removed","removed_left","removed_right","wrong_left","wrong_right"):
            totals={key:0 for key in ("known_root_rmse_m","known_pose_rad","reconstruction")}
            for row in rows:
                for key in totals:
                    a=np.mean([v[key] for v in row["shared_state_controls"] if v["control"]=="correct"])
                    b=np.mean([v[key] for v in row["shared_state_controls"] if v["control"]==control])
                    totals[key]+=int(a<b)
            wins[control]=totals
        main["heldout_shared_state_control_wins"]=wins
    result=dict(schema="r13_motion_mask_conditioned_denoiser_evidence",baseline="04f817b73487d85b6408a5af0ce32180a5ffb835",
        config=read(root/"teacher/protocol.json")["config"],official=read(root/"official/result.json"),teacher=read(root/"teacher_native/result.json"),
        teacher_feature_probes=read(root/"teacher/result.json")["records"],training=training,groups=groups,
        tiny=read(root/"tiny_native/result.json"),main=main,
        native_detail=read(root/"native_detail/result.json"),inspection=read(root/"inspection/result.json"),checks=check,evidence=evidence,receipts=receipts,new_whole_song_generations=0,human_reviewed=False,operational_qualified=False,
        limitations=["Tiny fit is memorization, not generalization","Historical B3 is not a matched parameter or training-data ablation",
            "Self-policy snapshots are stale after subsequent updates, not fully on-policy","Official DanceFusion runtime was not reconstructed",
            "No new ChangE style training or whole-song qualification"])
    write("reports/r13_summary.json",result);write("reports/r13_checks.json",check)
    write("reports/r13_receipt.json",dict(code_commit=commit(),summary_sha256=sha("reports/r13_summary.json"),evidence=evidence,receipts=receipts,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,new_whole_song_generations=0,operational_qualified=False))
    print("R13_SEALED",sha("reports/r13_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
