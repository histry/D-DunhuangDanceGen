"""Post-small-set research gate; no choice of weights or budget from exposed ChangE."""
import argparse
from pathlib import Path
from dunhuang.research.state_r27.common import read,write,sha,commit
def run(root):
    root=Path(root);path=root/"population_admission.json";assert not path.exists()
    evaluation=read(root/"evaluation/result.json");fit=read(root/"models/T1/result.json")["fit"]["passed"]
    gates=[r for r in evaluation["gates"] if r["arm"]=="T1"]
    admitted=fit and all(r["source_pass"] and r["proxy_pass"] for r in gates)
    write(path,dict(code_commit=commit(),T1_fit=fit,T1_gates=gates,expanded_training_admitted=admitted,
        additional_training_steps=0,required_next_step="Register finite new-population budget before any additional training" if admitted else "Stop enlargement; retain all failed source and proxy gates",
        evaluation_sha256=sha(root/"evaluation/result.json"),whole_song_generations=0,operational_qualified=False))
    print("R28_POPULATION_ADMISSION",admitted,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);args=p.parse_args();run(args.root)
