"""Independent exact-result verification. Existing outputs and failed receipts stay intact."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import write,checked
from dunhuang.research.local_r23.common import read,same_arrays,forbid_optimizers,commit

def run(root):
    root=Path(root);out=root/"independent_checks.json";assert not out.exists();forbid_optimizers()
    count=0;reports=0
    for stage in ["samples","gradients","represent","external","physical_tracking","native"]:
        a=root/stage;b=root/(stage+"_replay")
        assert read(a/"result.json")==read(b/"result.json"),stage
        for p in a.rglob("*.npz"):
            same_arrays(p,b/p.relative_to(a));count+=1
        for p in a.rglob("*physical.json"):
            assert read(p)==read(b/p.relative_to(a));reports+=1
    sample=read(root/"samples/result.json");assert sample["sampler_calls"]==36 and len(sample["bindings"])==64
    assert read(root/"gradients/result.json")["actual_pre_snapshots"]==9
    rep=read(root/"represent/result.json");assert rep["latent_least_squares_fits"]==18 and len(rep["records"])==24
    assert read(root/"external/result.json")["full_native_status"]=="NOT_EVALUABLE"
    tracking=read(root/"physical_tracking/result.json");assert tracking["tracking_rollouts"]==0
    frozen=["src/dunhuang/research/branch_r22","src/dunhuang/research/context_r21","src/dunhuang/research/temporal_r20",
        "src/dunhuang/pipeline.py","src/dunhuang/planning","src/dunhuang/edge_backend.py","src/dunhuang/sampling.py","inputs/lodge_r22/official"]
    diff=subprocess.check_output(["git","diff","c6bebeb1603f2b3d1cf35170d7d3de044fadbb31","--"]+frozen,text=True)
    assert not diff
    for b in sample["bindings"]:
        if not b["reused_R22"]:
            t=np.load(root/"samples"/b["trace_file"]);assert np.array_equal(t["sample"][t["hard"]],t["condition"][t["hard"]])
    assert count==144 and reports==88,(count,reports)
    write(out,dict(code_commit=commit(),array_bundles_exact=count,physical_reports_exact=reports,
        sampler_calls=36,replay_sampler_calls=36,reused_cfg2_without_new_sampling=28,
        actual_pre_gradient_readouts=9,representation_latent_fits=18,replay_representation_latent_fits=18,
        optimizer_steps=0,whole_song_generations=0,frozen_changed_paths=[],operational_qualified=False))
    print("R23_INDEPENDENT_CHECKS",count,reports,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);run(p.parse_args().root)
