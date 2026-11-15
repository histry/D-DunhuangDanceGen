"""One sealed branch, selected only from synthetic Train positive controls."""
import argparse
from pathlib import Path
from .common import *

def run(cfg,root):
    root=Path(root);out=root/"decision.json";assert not out.exists()
    model=read(root/"models/P/result.json");evaluation=read(root/"positive/result.json")
    fit=model["fit"]["passed"];sample=evaluation["gates"]["P"]["passed"]
    route="capacity" if not fit else "actual_state" if not sample else "transfer_only"
    write(out,dict(route=route,single_step_fit=fit,complete_sampling_fit=sample,P_optimizer_steps=model["optimizer_steps"],
        code_commit=commit(),P_result_sha256=sha(root/"models/P/result.json"),positive_result_sha256=sha(root/"positive/result.json"),
        selection_scope="MODEL_GENERATED_TRAIN_ONLY_NO_DEV_OR_CHANGE_READ",Q2_admitted=False,whole_song_generations=0,operational_qualified=False))
    print("R27_ROUTE",route,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);a=p.parse_args();run(read(a.config),a.root)
