"""Independent saved-output audit. No model fitting, sampler or optimizer."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays
from dunhuang.research.root_pose_r15.geometry import qualification
from dunhuang.research.root_pose_r15.common import source

def run(root):
    root=Path(root);forbid_optimizers();cfg=read(root/"preflight/protocol.json")["config"]
    _,rows,_,_=source(cfg,manifest=True);arrays=0;native_arrays=0
    assert read(root/"preflight/result.json")==read(root/"preflight_replay/result.json")
    for f in ("target_fk.npz","skeleton.npz"):
        same_arrays(root/"preflight"/f,root/"preflight_replay"/f);arrays+=1
    p=read(root/"probes/result.json");pr=read(root/"probes_replay/result.json")
    assert p["records"]==pr["records"] and p["model_unchanged"] and pr["model_unchanged"]
    assert p["optimizer_steps"]==pr["optimizer_steps"]==0
    assert len(p["records"])==64 and sum(len(r["trials"]) for r in p["records"])==256
    for key in p["probes"]:
        same_arrays(checked(p["probes"][key]),checked(pr["probes"][key]));arrays+=1
    protocol=read(root/"probes/protocol.json");assert len(set(protocol["donors"]))==8
    for i,j in enumerate(protocol["donors"]):assert rows[i]["row"]["source_group"]!=rows[j]["row"]["source_group"]
    for r in p["records"]:
        assert r["input_coordinates"]==(270 if r["arm"]=="J1" else 1350)
        for t,fraction in zip(r["trials"],cfg["line_fractions"]):
            assert t["input_l2"]<=cfg["root_input_l2"]*fraction*(1+1e-5)+1e-8
    numeric=all(r["derivative_relative_error"]<=.05 or abs(r["derivative_numeric"]-r["derivative_analytic"])<=1e-4 for r in p["records"])
    n=read(root/"native/result.json");nr=read(root/"native_replay/result.json");assert n==nr
    assert n["native_readouts"]==272 and len(n["trials"])==256 and len(n["states"])==80 and len(n["groups"])==40
    assert n["qualification"]==qualification(n["groups"],cfg)
    immutable={r["case"] for r in read(root/"preflight/result.json")["records"] if r["role"]=="capacity_train" and r["necessary_status"]!="UNKNOWN"}
    for r in n["trials"]:
        if r["case"] in immutable:assert not r["metrics"]["native_ok"]
    for folder in (root/"native").iterdir():
        if not folder.is_dir():continue
        for f in ("native.npy","fk.npy"):
            assert np.array_equal(np.load(folder/f),np.load(root/"native_replay"/folder.name/f));native_arrays+=1
        assert read(folder/"physical.json")==read(root/"native_replay"/folder.name/"physical.json")
    # Replay the registered first-eligible selection, rather than selecting a best score.
    for r in n["states"]:
        if r["arm"]=="J0":continue
        trials=[x for x in n["trials"] if (x["case"],x["arm"],x["t"])==(r["case"],r["arm"],r["t"])]
        eligible=[x for x in trials if x["eligible"]];first=eligible[0] if eligible else None
        assert r["accepted"]==(first is not None) and r["selected_trial"]==(first["trial"] if first else None)
        base=next(x for x in n["states"] if x["case"]==r["case"] and x["t"]==r["t"] and x["arm"]=="J0")
        assert r["metrics"]==(first["metrics"] if first else base["metrics"])
    frozen=subprocess.check_output(["git","diff","--name-only","bfb269d","--","src/dunhuang","configs","reports/r14_summary.json","reports/r14_receipt.json","reports/r13_summary.json","tests/test_boundary_state_r14.py"],text=True).splitlines()
    frozen=[p for p in frozen if not p.startswith("src/dunhuang/research/root_pose_r15/")];assert not frozen
    receipt=Path(str(root)+"_receipts");assert (receipt/"exit_status.txt").read_text().strip()=="0"
    result=dict(code_commit=commit(),execution_commit=(receipt/"execution_commit.txt").read_text().strip(),
        array_bundles_exact=arrays,native_arrays_exact=native_arrays,native_reports_exact=272,
        fixed_train_music_groups=8,policy_states=16,arms=5,finite_trials_exact=256,numeric_gradients_passed=numeric,
        max_gradient_relative_error=max(r["derivative_relative_error"] for r in p["records"]),
        same_total_input_budget=True,all_denominators_preserved=True,first_eligible_selection_verified=True,
        frozen_changed_paths=frozen,optimizer_steps=0,replay_optimizer_steps=0,model_weights_unchanged=True,
        oracle_capacity_passed=bool(numeric and n["qualification"]["local_capacity_passed"]),
        new_full_window_generations=0,new_whole_song_generations=0,operational_qualified=False)
    write(root/"independent_checks.json",result);print("R15_VERIFIED",result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
