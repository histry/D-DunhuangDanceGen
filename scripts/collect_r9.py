"""Seal completed R9 inference, finite projection and B2 training separately."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read
def run(main,b2):
    main=Path(main);b2=Path(b2);receipts={}
    for label,root in (("main",main),("b2",b2)):
        folder=Path(str(root)+"_receipts");assert int((folder/"exit_status.txt").read_text())==0
        receipts[label]=dict(code_commit=(folder/"execution_commit.txt").read_text().strip(),started=(folder/"started_utc.txt").read_text().strip(),finished=(folder/"finished_utc.txt").read_text().strip(),
            files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    post=Path(str(main)+"_post_receipts");assert int((post/"exit_status.txt").read_text())==0
    receipts["post"]=dict(code_commit=(post/"execution_commit.txt").read_text().strip(),files={p.name:sha(p) for p in post.iterdir() if p.is_file()})
    source=read(main/"source/result.json");boundary=read(main/"boundary/result.json");training=read(b2/"train/result.json")
    assert training["optimizer_steps"]==300 and read(b2/"train_replay/result.json")["optimizer_steps"]==0
    checks=read(main/"independent_checks.json");bchecks=read(b2/"independent_checks.json")
    analysis=read(main/"analysis/result.json");media=read(main/"analysis/protocol.json")
    evidence={};commits={}
    for prefix,root,phases in (("r9",main,["training_preflight","boundary","boundary_replay","source","source_replay","analysis","analysis_replay"]),("b2",b2,["data_replay","train","train_replay","generation","generation_replay","application","application_replay"])):
        for phase in phases:
            commits[prefix+"/"+phase]=read(root/phase/"protocol.json")["code_commit"]
            for name in ("protocol.json","result.json","replay.json","rows.json","media_checks.json"):
                file=root/phase/name
                if file.is_file():evidence[prefix+"/"+phase+"/"+name]=sha(file)
    summary=dict(schema="r9_separated_source_boundary_training_summary",baseline="0911637f76372a54d3c315e3088f8d86abfeaa6a",groups=analysis["groups"],root_traces=analysis["traces"],foot_factors=analysis["foot_factors"],
        boundary=boundary,training_data=read(main/"training_preflight/result.json"),b2_training=training,b2_validation_generation=read(b2/"generation/result.json"),b2_original_context_application=read(b2/"application/result.json"),
        checks=checks,b2_checks=bchecks,evidence=evidence,stage_commits=commits,receipts=receipts,
        inference_only_source_updates=0,adapter_training_steps=300,backbone_training_steps=0,new_whole_song_generations=0,source_truth="observed fitted rotations; missing Root translation/contact/fingers UNKNOWN",
        human_reviewed=False,cultural_authenticity="UNKNOWN",operational_qualified=False)
    write("reports/r9_summary.json",summary)
    write("reports/r9_checks.json",dict(main=checks,b2=bchecks))
    failed=Path("outputs/feasible_r9_01_receipts/exit_status.txt");assert failed.read_text().strip()=="1"
    write("reports/r9_receipt.json",dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),summary_sha256=sha("reports/r9_summary.json"),
        stage_commits=commits,receipts=receipts,first_engineering_failure_retained=str(failed),temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        formal_changed_paths=checks["formal_changed_paths"],new_whole_song_generations=0,operational_qualified=False))
    print("R9_SEALED",sha("reports/r9_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--main",required=True);p.add_argument("--b2",required=True);a=p.parse_args();run(a.main,a.b2)
