"""Fresh saved-science verification with zero training/optimizer calls."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays

def run(root):
    root=Path(root);forbid_optimizers();data=read(root/"data/result.json");cfg=read(root/"data/protocol.json")["config"]
    bundles=0;arrays=0;physics=0
    fit={r["left"]["source_group"] for r in data["records"] if r["role"]=="fit"}
    held={r["left"]["source_group"] for r in data["records"] if r["role"]=="eval"}
    assert len(fit)==8 and len(held)==4 and not fit&held
    again=read(root/"data_replay/result.json")
    for a,b in zip(data["records"],again["records"]):
        assert (a["role"],a["case"],a["left"])==(b["role"],b["case"],b["left"])
        same_arrays(checked(a["arrays"]),checked(b["arrays"]));bundles+=1
        assert a["feature_overlap"]==b["feature_overlap"]
    same_arrays(root/"data/parity.npz",root/"data_replay/parity.npz");bundles+=1
    audit=read(root/"audit/result.json");ar=read(root/"audit_replay/result.json")
    assert audit["optimizer_steps"]==ar["optimizer_steps"]==0 and audit["model_unchanged"] and ar["model_unchanged"]
    assert len(audit["factors"])==32 and len(audit["music"])==8
    noise={}
    for part in ("factors","music"):
        for a,b in zip(audit[part],ar[part]):
            same_arrays(checked(a["trace"]),checked(b["trace"]));bundles+=1
            r=read(checked(a["sampler"]));assert r==read(checked(b["sampler"]))
            key=(part,a["case"]);noise.setdefault(key,set()).add(r["noise_sha256"])
    assert all(len(s)==1 for s in noise.values())
    models={}
    for mode in ("cross","self"):
        a=read(root/"models"/mode/"result.json");b=read(root/"models"/(mode+"_replay")/"result.json")
        assert a["q_scores"]==b["q_scores"] and a["final_digest"]==b["final_digest"]
        assert a["base_digest"]==b["base_digest"] and a["base_unchanged"] and b["base_unchanged"]
        assert a["optimizer_steps"]==cfg["training_steps"] and b["optimizer_steps"]==0
        assert a["fk_parity_max_m"]<=1e-5
        assert len(a["ledger"])==cfg["training_steps"]
        assert all(r["group"] in fit for r in a["ledger"])
        same_arrays(root/"models"/mode/"q_eval.npz",root/"models"/(mode+"_replay")/"q_eval.npz");bundles+=1
        checked(a["checkpoint"]);models[mode]=a
    assert models["cross"]["parameters"]==models["self"]["parameters"]
    assert models["cross"]["initial_digest"]==models["self"]["initial_digest"]
    assert models["cross"]["ledger"]==models["self"]["ledger"]
    assert models["cross"]["base_digest"]==models["self"]["base_digest"]
    rollout=read(root/"rollout/result.json");rr=read(root/"rollout_replay/result.json")
    assert len(rollout["bindings"])==len(rr["bindings"])==36
    noise={}
    for a,b in zip(rollout["bindings"],rr["bindings"]):
        assert (a["case"],a["arm"],a["task"])==(b["case"],b["arm"],b["task"])
        same_arrays(checked(a["trace"]),checked(b["trace"]));bundles+=1
        x=read(checked(a["sampler"]));y=read(checked(b["sampler"]))
        xp=x.pop("prefix");yp=y.pop("prefix");assert x==y
        if xp:assert xp["sha256"]==yp["sha256"]
        noise.setdefault((a["case"],a["task"]),set()).add(x["noise_sha256"])
        z=np.load(checked(a["trace"]));assert z["sample"].shape==(225,151)
        assert not z["known"][:,:4].any()
        if a["task"]=="free":assert not z["known"][:,:13].any() and not z["hard"].any()
        if a["task"]=="progressive":
            prefix=np.load(checked(xp))["sample"][:75]
            assert np.array_equal(z["sample"][:75,4:],prefix[:,4:]) and not z["known"][75:,:13].any()
        assert np.array_equal(z["sample"][z["hard"]],z["condition"][z["hard"]])
    assert all(len(s)==1 for s in noise.values())
    for phase in ("audit_native","native"):
        assert read(root/phase/"result.json")==read(root/(phase+"_replay")/"result.json")
        for folder in (root/phase).iterdir():
            if not folder.is_dir():continue
            for f in folder.iterdir():
                other=root/(phase+"_replay")/folder.name/f.name
                if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other));arrays+=1
                else:assert read(f)==read(other);physics+=1
    frozen=subprocess.check_output(["git","diff","--name-only","70988d9","--","src/dunhuang","configs","reports/r16_summary.json","reports/r16_receipt.json","tests/test_continuous_r16.py"],text=True).splitlines()
    frozen=[p for p in frozen if not p.startswith("src/dunhuang/research/global_clock_r17/")];assert not frozen
    receipts={}
    for stage in ("audit","train"):
        p=Path(str(root)+"_"+stage+"_receipts");assert (p/"exit_status.txt").read_text().strip()=="0"
        receipts[stage]=(p/"execution_commit.txt").read_text().strip()
    result=dict(code_commit=commit(),array_bundles_exact=bundles,native_arrays_exact=arrays,physical_reports_exact=physics,
        first_sampler_calls=76,replay_sampler_calls_exact=76,fit_music_groups=8,adapter_heldout_exposed_music_groups=4,
        models_parameters=models["cross"]["parameters"],matched_initialization_and_ledger=True,music_and_base_frozen=True,
        training_optimizer_steps=2*cfg["training_steps"],replay_optimizer_steps=0,optimizer_steps=0,
        global_noise_clock_verified=True,free_future_root_truth_not_observed=True,generated_prefix_hash_and_preservation_verified=True,
        frozen_changed_paths=frozen,receipts=receipts,new_whole_song_generations=0,operational_qualified=False)
    write(root/"independent_checks.json",result);print("R17_VERIFIED",result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
