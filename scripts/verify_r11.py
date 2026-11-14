"""Fresh-process verification of saved R11 science; no optimizer or sampler."""
import argparse,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit
from dunhuang.research.primitive_reperformance.study import ARRAYS

def run(root):
    root=Path(root);assert read(root/"audit/result.json")==read(root/"audit_replay/result.json")
    assert read(root/"analysis/result.json")==read(root/"analysis_replay/result.json")
    if (root/"analysis_cached").exists():assert read(root/"analysis/result.json")==read(root/"analysis_cached/result.json")
    assert read(root/"audit/result.json")["oracle_passed"]
    oracle=0
    for b in read(root/"audit/protocol.json")["bindings"]:
        folder=Path(checked(b["oracle"])).parent
        other=root/"audit_replay"/folder.name
        for f in ("oracle_native.npy","oracle_fk.npy"):
            assert np.array_equal(np.load(folder/f),np.load(other/f));oracle+=1
    n=0;trace=0;steps=0
    for b in read(root/"trace/protocol.json")["bindings"]:
        folder=Path(checked(b["trace"])).parent;other=root/"trace_replay"/folder.name
        for f in ARRAYS:assert np.array_equal(np.load(folder/f),np.load(other/f));n+=1
        for name in ("prediction","latent"):
            assert np.array_equal(np.load(folder/"trace.npz")[name],np.load(other/"trace.npz")[name]);trace+=1
        assert read(folder/"trace.json")==read(other/"trace.json")
        assert read(folder/"physical.json")==read(other/"physical.json")
        parent=Path(checked(b["reference"])).parent
        for f in ARRAYS:assert np.array_equal(np.load(folder/f),np.load(parent/f))
        assert read(folder/"physical.json")==read(parent/"physical.json")
        for row in read(folder/"trace.json")["steps"]:
            assert row["rng_unchanged"] and row["a0_features_exact"]
            assert [r["arm"] for r in row["arms"]]==["A0","A1","A2","A3","A4","A5"]
            assert all(len(a["residual_rms_by_branch"])==2 for a in row["arms"])
            assert all(v==0 for v in row["arms"][-1]["residual_rms_by_branch"]);steps+=1
    assert (n,trace,steps)==(48,16,400)
    formal=subprocess.check_output(["git","diff","--name-only","4496e43","--","src/dunhuang","configs"],text=True).splitlines()
    formal=[v for v in formal if not v.startswith("src/dunhuang/research/")];assert not formal
    result=dict(code_commit=commit(),oracle_arrays=oracle,full_motion_arrays=n,trace_arrays=trace,shared_trajectory_steps=steps,
        original_C2_a0_arrays_equal=True,analysis_exact_replay=True,cached_analysis_exact=(root/"analysis_cached/result.json").is_file(),rng_unchanged=True,a0_features_exact=True,formal_changed_paths=formal,
        optimizer_steps=0,new_whole_song_generations=0,operational_qualified=False)
    write(root/"independent_checks.json",result);print("R11_VERIFIED",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
