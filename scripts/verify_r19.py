"""Verify sealed pre/post AdamW observations and all zero-optimizer replays."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write,sha
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.gate import assess

def run(root,out=None):
    root=Path(root);forbid_optimizers();cfg=read(root/"calibration/protocol.json")["config"]
    assert read(root/"calibration/result.json")==read(root/"calibration_replay/result.json")
    same_arrays(root/"calibration/norms.npz",root/"calibration_replay/norms.npz");bundles=1
    gate=read(root/"gate.json");models={};update_pairs=0;steps=0
    arms=cfg["arms"]+(["U"] if gate["unique_admitted"] else [])
    for arm in arms:
        p=root/"models"/arm;r=read(p/"result.json");rr=read(root/"models"/(arm+"_replay")/"result.json")
        assert r["optimizer_steps"]==cfg["steps"] and rr["optimizer_steps"]==0
        checked(r["calibration"]);assert r["calibration"]["sha256"]==sha(root/"calibration/result.json")
        for name in ("initial_scores","scores","initial_motion","initial_head","final_motion","final_head","frozen_digest","ledger","update_observations"):
            assert r[name]==rr[name],(arm,name)
        assert r["initial_motion"]!=r["final_motion"] and r["fk_parity_max_m"]<=1e-5
        for name in ("initial_q.npz","q_eval.npz"):same_arrays(p/name,root/"models"/(arm+"_replay")/name);bundles+=1
        assert [x["step"] for x in r["update_observations"]]==cfg["snapshots"]
        for observation in r["update_observations"]:
            checked(observation["pre"]);checked(observation["post"])
            assert observation["readout"]["actual_update_norm"]>0;update_pairs+=1
        models[arm]=r;steps+=r["optimizer_steps"]
    assert len({m["initial_motion"] for m in models.values()})==len({m["initial_head"] for m in models.values()})==1
    assert all(m["ledger"]==models["N0"]["ledger"] for m in models.values())
    assert all(m["motion_parameters"]==models["N0"]["motion_parameters"] and m["head_parameters"]==models["N0"]["head_parameters"] for m in models.values())
    assert assess(cfg,models)=={k:v for k,v in gate.items() if k!="code_commit"}
    roll=read(root/"rollout/result.json");rep=read(root/"rollout_replay/result.json");assert len(roll["bindings"])==len(rep["bindings"])
    noises={}
    for a,b in zip(roll["bindings"],rep["bindings"]):
        assert (a["case"],a["arm"],a["control"])==(b["case"],b["arm"],b["control"])
        same_arrays(checked(a["trace"]),checked(b["trace"]));bundles+=1
        receipt=read(checked(a["sampler"]));assert receipt==read(checked(b["sampler"]))
        assert receipt["all_edge_calls_local150"] and not receipt["output_pose_paste"]
        noises.setdefault(a["case"],set()).add(receipt["noise_sha256"])
        tr=np.load(checked(a["trace"]));assert tr["sample"].shape==(225,151) and not tr["known"][:,:4].any()
        assert np.array_equal(tr["sample"][tr["hard"]],tr["condition"][tr["hard"]])
    assert all(len(v)==1 for v in noises.values())
    assert read(root/"native/result.json")==read(root/"native_replay/result.json")
    arrays=0;reports=0
    for folder in (root/"native").iterdir():
        if not folder.is_dir():continue
        for f in folder.iterdir():
            other=root/"native_replay"/folder.name/f.name
            if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other));arrays+=1
            else:assert read(f)==read(other);reports+=1
    readouts=root/"aligned_readouts"
    if readouts.exists():
        a=read(readouts/"result.json");b=read(root/"aligned_readouts_replay/result.json")
        assert a["records"]==b["records"] and a["unique_full_responses"]==b["unique_full_responses"]
        assert a["optimizer_steps"]==b["optimizer_steps"]==0
        for folder in readouts.iterdir():
            if not folder.is_dir():continue
            for f in folder.iterdir():
                other=root/"aligned_readouts_replay"/folder.name/f.name
                if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other));arrays+=1
                else:assert read(f)==read(other);reports+=1
    frozen=subprocess.check_output(["git","diff","--name-only",cfg["baseline"],"--","src","configs","reports/r18_summary.json","reports/r18_receipt.json"],text=True).splitlines()
    frozen=[p for p in frozen if not p.startswith("src/dunhuang/research/objective_r19/")];assert not frozen,frozen
    result=dict(code_commit=commit(),array_bundles_exact=bundles,native_arrays_exact=arrays,physical_reports_exact=reports,
        actual_adamw_update_pairs_exact=update_pairs,first_sampler_calls=len(roll["bindings"]),replay_sampler_calls_exact=len(roll["bindings"]),
        training_optimizer_steps=steps,replay_optimizer_steps=0,actual_train_source_groups=2,heldout_generation_groups=0,
        matched_initialization_ledger_and_noise=True,frozen_changed_paths=frozen,unique_executed="U" in models,
        whole_song_generations=0,operational_qualified=False)
    target=Path(out) if out else root/"independent_checks.json";assert not target.exists();write(target,result);print("R19_VERIFIED",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out");a=p.parse_args();run(a.root,a.out)
