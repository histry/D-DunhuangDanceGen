"""Fresh replay verifier: never construct an optimizer or mutate scientific outputs."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays

def run(root):
    root=Path(root).resolve();forbid_optimizers();cfg=read(root/"interface/protocol.json")["config"]
    assert read(root/"interface/result.json")==read(root/"interface_replay/result.json")
    same_arrays(root/"interface/predictions.npz",root/"interface_replay/predictions.npz");bundles=1
    assert read(root/"temporal_calibration/result.json")==read(root/"temporal_calibration_replay/result.json")
    same_arrays(root/"temporal_calibration/norms.npz",root/"temporal_calibration_replay/norms.npz");bundles+=1
    gate=read(root/"cache_gate.json");arms=["S0","S1","A0","A1","A2"]+(["A3"] if gate["cache_admitted"] else [])
    models={};steps=0;updates=0;cache_calls=0
    for arm in arms:
        a=read(root/"models"/arm/"result.json");b=read(root/"models"/(arm+"_replay")/"result.json")
        for name in ("initial","initial_scores","scores","final_selected","final_head","update_observations","ledger"):
            assert a[name]==b[name],(arm,name)
        assert a["optimizer_steps"]==cfg["steps"] and b["optimizer_steps"]==0
        for name in ("initial_q.npz","q_eval.npz"):
            same_arrays(root/"models"/arm/name,root/"models"/(arm+"_replay")/name);bundles+=1
        for item in a["update_observations"]:
            checked(item["pre"]);checked(item["post"]);assert item["readout"]["actual_update_norm"]>0;updates+=1
            assert "decoder_last" in item["readout"]["layers_backward"]
            assert item["readout"]["parameter_stats"]["decoder_last"]["trainable_elements"]==0
        for original in a["cache_bindings"]:
            path=Path(checked(original["trace"]));relative=path.relative_to(root/"models"/arm)
            same_arrays(path,root/"models"/(arm+"_replay")/relative);bundles+=1;cache_calls+=1
            checked(original["weight"]);checked(original["data_arrays"]);checked(original["scaler"])
        models[arm]=a;steps+=a["optimizer_steps"]
    assert models["S0"]["initial"]["selected"]==models["S1"]["initial"]["selected"]
    assert models["S0"]["initial"]["head"]==models["S1"]["initial"]["head"]
    assert models["S0"]["initial"]["frozen_except_null"]==models["S1"]["initial"]["frozen_except_null"]
    assert models["S0"]["ledger"]==models["S1"]["ledger"]
    assert all(models[x]["initial"]==models["A0"]["initial"] for x in arms if x.startswith("A"))
    def neutral(rows):return [{k:v for k,v in row.items() if k not in ("x_t_sha256","use_cache")} for row in rows]
    assert neutral(models["A1"]["ledger"])==neutral(models["A2"]["ledger"])
    if "A3" in models:
        assert neutral(models["A2"]["ledger"])==neutral(models["A3"]["ledger"])
        assert sum(r["use_cache"] for r in models["A3"]["ledger"])==cfg["steps"]//2
    sampler_calls=0;arrays=0;reports=0;noise={}
    for part in ("structure","gap","p0","final"):
        a=read(root/(part+"_rollout")/"result.json");b=read(root/(part+"_rollout_replay")/"result.json")
        assert len(a["bindings"])==len(b["bindings"])
        for x,y in zip(a["bindings"],b["bindings"]):
            assert (x["case"],x["arm"],x["seed"],x["control"])==(y["case"],y["arm"],y["seed"],y["control"])
            same_arrays(checked(x["trace"]),checked(y["trace"]));bundles+=1;sampler_calls+=1
            receipt=read(checked(x["sampler"]));assert receipt==read(checked(y["sampler"]))
            assert receipt["single_call_joint225"]==(x["layout"]=="joint")
            if x["control"]=="correct":noise.setdefault((part,x["case"],x["seed"]),set()).add(receipt["noise_sha256"])
            tr=np.load(checked(x["trace"]));assert tr["sample"].shape==(225,151)
            assert np.array_equal(tr["sample"][tr["hard"]],tr["condition"][tr["hard"]])
        if part=="gap":continue
        assert read(root/(part+"_native")/"result.json")==read(root/(part+"_native_replay")/"result.json")
        for folder in (root/(part+"_native")).iterdir():
            if not folder.is_dir():continue
            for f in folder.iterdir():
                other=root/(part+"_native_replay")/folder.name/f.name
                if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other));arrays+=1
                else:assert read(f)==read(other);reports+=1
    assert all(len(v)==1 for v in noise.values())
    frozen=subprocess.check_output(["git","diff","--name-only",cfg["baseline"],"--","src","configs","reports/r19_summary.json","reports/r19_receipt.json"],text=True).splitlines()
    frozen=[p for p in frozen if not p.startswith("src/dunhuang/research/temporal_r20/")];assert not frozen
    result=dict(code_commit=commit(),array_bundles_exact=bundles,native_arrays_exact=arrays,physical_reports_exact=reports,
        update_pairs_exact=updates,training_optimizer_steps=steps,replay_optimizer_steps=0,first_sampler_calls=sampler_calls,
        cache_sampler_calls=cache_calls,replay_sampler_calls_exact=sampler_calls+cache_calls,
        final_seed_count=len(cfg["eval_seeds"]),train_groups=2,dev_exposed_groups=2,
        selected_layout=read(root/"structure_gate.json")["layout"],A3_executed="A3" in models,
        frozen_changed_paths=frozen,whole_song_generations=0,operational_qualified=False)
    out=root/"independent_checks.json";assert not out.exists();write(out,result);print("R20_VERIFIED",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
