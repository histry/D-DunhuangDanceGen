"""Publish bounded summaries while complete traces and snapshots remain in outputs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);receipt=Path(str(root)+"_receipts")
    assert (receipt/"exit_status.txt").read_text().strip()=="0"
    cfg=read(root/"calibration/protocol.json")["config"];checks=read(root/"independent_checks.json");gate=read(root/"gate.json")
    arms=cfg["arms"]+(["U"] if gate["unique_admitted"] else []);models={}
    for arm in arms:
        m=read(root/"models"/arm/"result.json")
        models[arm]={k:m[k] for k in ("checkpoint","optimizer_steps","initial_motion","final_motion","initial_head","final_head",
            "frozen_digest","initial_scores","scores","update_observations","fk_parity_max_m","motion_parameters","head_parameters")}
        models[arm].update(first=m["history"][0],last=m["history"][-1])
    native=read(root/"native/result.json");means=[]
    for arm in ["D0"]+arms:
        rows=[r for r in native["records"] if r["arm"]==arm and r["control"]=="correct"]
        means.append(dict(arm=arm,cases=len(rows),native_passes=sum(r["native_ok"] for r in rows),
            metrics={k:float(np.mean([r[k] for r in rows])) for k in ("root_rmse_m","foot_rmse_m","jerk_max_mps3","jerk_p95_mps3",
                "activity_mps","pose_error_rad","raw_unknown_mse_147","raw_observed_root_rmse_m")}))
    evidence={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*.json"))
        if p.name in ("protocol.json","result.json","gate.json","independent_checks.json")}
    result=dict(schema=cfg["schema"],code_commit=commit(),config=cfg,calibration=read(root/"calibration/result.json"),
        training=models,gate=gate,means=means,records=native["records"],checks=checks,
        videos=read(root/"videos/protocol.json")["records"],actual_train_groups=2,heldout_generation_groups=0,
        human_reviewed=False,semantic_event_truth="UNKNOWN",whole_song_generations=0,operational_qualified=False,
        limitations=["Only two Train sources; no generalization claim","N0 vs N1 is an objective-package ablation",
            "Four training timesteps, not full noise coverage","Paired FK targets need not pass Native",
            "Local gradient/finite-update evidence is not a unique causal decomposition","No generated-history or new ChangE training"])
    p=Path("reports/r19_summary.json");write(p,result);write("reports/r19_checks.json",checks)
    write("reports/r19_receipt.json",dict(code_commit=commit(),summary_sha256=sha(p),evidence=evidence,
        execution_commit=(receipt/"execution_commit.txt").read_text().strip(),exit_status=0,
        receipt_files={p.name:sha(p) for p in receipt.iterdir() if p.is_file()},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,whole_song_generations=0,operational_qualified=False))
    print("R19_COLLECTED",sha(p),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
