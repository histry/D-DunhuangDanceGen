"""Freeze post-intervention choices before viewing any AIST/ChangE transfer."""
import argparse
from pathlib import Path
import numpy as np
from .common import *
from .qualification import fitting_gate

def run(root):
    root=Path(root);decision=read(root/"decision.json");route=decision["route"];out=root/"branch_summary.json";assert not out.exists()
    models=["P"];eligible=False;values={}
    assert route=="transfer_only","Other registered branches require a separate implementation and publication"
    write(out,dict(route=route,models_for_transfer=models,Q2_eligible=eligible,eligibility=values,selection_scope="MODEL_GENERATED_TRAIN_ONLY",code_commit=commit(),whole_song_generations=0,operational_qualified=False))
    print("R27_BRANCH_SUMMARY",route,models,"Q2",eligible,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
