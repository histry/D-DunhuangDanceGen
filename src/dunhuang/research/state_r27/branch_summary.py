"""Freeze post-intervention choices before viewing any AIST/ChangE transfer."""
import argparse
from pathlib import Path
import numpy as np
from .common import *
from .qualification import fitting_gate

def run(root):
    root=Path(root);decision=read(root/"decision.json");route=decision["route"];out=root/"branch_summary.json";assert not out.exists()
    models=["P"];eligible=False;values={}
    if route=="capacity":models+=["P2"]
    elif route=="actual_state":
        models+=["Q0","Q1"];post=read(root/"post_positive/result.json");q=read(root/"models/Q1/result.json");initial=read(root/"models/P/result.json")["curves"][0]["records"]
        fit=fitting_gate(q["curves"][-1]["records"],initial)
        means={a:float(np.mean([x["metrics"]["body22_rad"] for x in post["records"] if x["arm"]==a and x["mode"]=="correct"])) for a in ("Q0","Q1")}
        eligible=fit["passed"] and not post["gates"]["Q1"]["passed"] and means["Q1"]<=.95*means["Q0"]
        values=dict(Q1_fit_relative_to_original_C1=fit,body_means=means)
    write(out,dict(route=route,models_for_transfer=models,Q2_eligible=eligible,eligibility=values,selection_scope="MODEL_GENERATED_TRAIN_ONLY",code_commit=commit(),whole_song_generations=0,operational_qualified=False))
    print("R27_BRANCH_SUMMARY",route,models,"Q2",eligible,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
