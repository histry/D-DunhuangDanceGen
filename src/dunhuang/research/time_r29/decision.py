"""A bounded coverage study is distinct from physical qualification and whole-song release."""
import argparse
from pathlib import Path
from dunhuang.research.state_r27.common import read,write,sha,commit
def run(root):
    root=Path(root);path=root/"coverage_admission.json";assert not path.exists()
    result=read(root/"evaluation_aist/result.json");g={x["role"]:x for x in result["gates"] if x["arm"]=="B1"}
    admitted=g["train"]["source_pass"] and not g["holdout"]["source_pass"]
    write(path,dict(code_commit=commit(),coverage_admitted=admitted,Train_source_gate=g["train"]["recordings_passed"],
        holdout_source_gate=g["holdout"]["recordings_passed"],Native_gate_not_used_for_research_coverage=True,
        coverage_optimizer_steps=0,required_action="Seal registered16-record/8-group cohort and run one fixed1200-update study" if admitted else "No population coverage training; retain main/FineDance controls",
        evidence_sha256=sha(root/"evaluation_aist/result.json"),whole_song_generations=0,operational_qualified=False))
    print("R29_COVERAGE_ADMISSION",admitted,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
