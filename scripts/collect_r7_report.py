"""Publish compact results with immutable phase and launcher receipts."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write
from dunhuang.research.source_capacity.common import read

def run(root,receipts,kind):
    root=Path(root);receipts=Path(receipts);assert (receipts/"exit_status.txt").read_text().strip()=="0"
    checks=read(root/"public_checks.json")
    if kind=="source":
        main=root/"control";r=read(main/"result.json")
        summary=dict(schema="source_contrast_r7_summary",scope="EXPOSED_TRAIN_OFFLINE_TWO_REFERENCE_IDENTITY",
            qualification=r["qualification"],new_projection_solves=r["new_projection_solves"],
            frozen_prediction_parities=r["frozen_prediction_parities"],
            numeric_trusted=sum(s["numerics"]["numeric_trusted"] for s in r["solves"]),
            primary_numeric_denominator=len(r["solves"]),new_solve_status_counts={},
            independent_evaluation=r["independent_evaluation"],optimizer_updates=0,full_generation_calls=0,
            model_training_started=False,operational_qualified=False,human_reviewed=False)
        for s in r["solves"]:
            if s["new_solve"]:summary["new_solve_status_counts"][s["status"]]=summary["new_solve_status_counts"].get(s["status"],0)+1
        phases=["preflight","preflight_replay","control","control_replay"]
    else:
        main=root/"generation";r=read(main/"result.json")
        summary=dict(schema="exposed_full_song_r7_summary",scope="EXPOSED_DUNHUANG_TRANSFER_TEST",
            music=read(root/"music/result.json"),generation=r,features_replay=read(root/"features/replay.json"),
            optimizer_updates=0,human_reviewed=False,operational_qualified=False)
        phases=["music","music_replay","features","generation","generation_replay","videos"]
    evidence={}
    execution={}
    for phase in phases:
        folder=root/phase
        execution[phase]=read(folder/"protocol.json")["code_commit"]
        for name in ("protocol.json","result.json","arrays.npz","replay.json","inventory.json","generation_plans.json"):
            file=folder/name
            if file.is_file():evidence[phase+"/"+name]=sha(file)
    evidence["public_checks.json"]=sha(root/"public_checks.json")
    summary["checks"]=checks;summary["evidence"]=evidence
    target=Path("reports")/("source_contrast" if kind=="source" else "exposed_full_song")
    write(str(target)+"_summary.json",summary)
    receipt=dict(schema="r7_sealed_execution_receipt",execution_commit=(receipts/"execution_commit.txt").read_text().strip(),
        verifier_commit=checks["code_commit"],stage_execution_commits=execution,summary_sha256=sha(str(target)+"_summary.json"),
        root=str(root.resolve()),exit_status=0,receipt_sha256={p.name:sha(p) for p in receipts.iterdir() if p.is_file()},
        started_utc=(receipts/"started_utc.txt").read_text().strip(),finished_utc=(receipts/"finished_utc.txt").read_text().strip(),
        prior_failed_roots_retained=True,optimizer_updates=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        operational_qualified=False)
    write(str(target)+"_receipt.json",receipt);write(str(target)+"_checks.json",checks)
    print("R7_SEALED_REPORT",kind,sha(str(target)+"_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipts",required=True);p.add_argument("--kind",choices=["source","full_song"],required=True);a=p.parse_args();run(a.out,a.receipts,a.kind)
