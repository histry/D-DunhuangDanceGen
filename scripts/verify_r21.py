"""Read-only cross-process ledger/array/update verification; fresh report destination."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays

def run(root,out=None):
    root=Path(root).resolve();out=Path(out) if out else root/"independent_checks.json"
    assert not out.exists();forbid_optimizers();data=read(root/"data/result.json");cfg=read(root/"data/protocol.json")["config"]
    other=read(root/"data_replay/result.json");bundles=0
    assert data["candidate_ledger"]==other["candidate_ledger"]
    for a,b in zip(data["records"],other["records"]):
        same_arrays(checked(a["arrays"]),checked(b["arrays"]));bundles+=1
        assert read(checked(a["source_physical"]))==read(checked(b["source_physical"]))
        assert a["audio225"]["sha256"]==b["audio225"]["sha256"]
    same_arrays(root/"data/parity.npz",root/"data_replay/parity.npz");bundles+=1
    for case in (0,1):
        same_arrays(root/"neutrality"/(str(case)+".npz"),root/"neutrality_replay"/(str(case)+".npz"));bundles+=1
        assert read(root/"neutrality"/(str(case)+".json"))==read(root/"neutrality_replay"/(str(case)+".json"))
    if data["support_admitted"]:
        same_arrays(root/"support_calibration/norms.npz",root/"support_calibration_replay/norms.npz");bundles+=1
        assert read(root/"support_calibration/result.json")==read(root/"support_calibration_replay/result.json")
    arms=[a for a in cfg["arms"] if a!="F" or data["support_admitted"]];models={};steps=0;updates=0;cache_calls=0
    for arm in arms:
        a=read(root/"models"/arm/"result.json");b=read(root/"models"/(arm+"_replay")/"result.json")
        assert a["optimizer_steps"]==(cfg["follow_steps"] if arm in ("Q","M","F") else cfg["steps"]) and b["optimizer_steps"]==0
        for n in ("initial","final_selected","final_head","ledger","update_observations","final_scores"):assert a[n]==b[n],(arm,n)
        assert len(a["curves"])==len(b["curves"])
        for x,y in zip(a["curves"],b["curves"]):
            assert x["scores"]==y["scores"] and x["update"]==y["update"]
            same_arrays(checked(x["predictions"]),checked(y["predictions"]));bundles+=1;checked(x["weight"])
        same_arrays(root/"models"/arm/"final_q.npz",root/"models"/(arm+"_replay")/"final_q.npz");bundles+=1
        for item in a["update_observations"]:
            checked(item["pre"]);checked(item["post"]);assert item["readout"]["actual_update_norm"]>0
            assert item["readout"]["parameter_stats"]["decoder_last"]["trainable_elements"]==0;updates+=1
        for x in a["cache_bindings"]:
            relative=Path(checked(x["trace"])).resolve().relative_to((root/"models"/arm).resolve())
            same_arrays(checked(x["trace"]),root/"models"/(arm+"_replay")/relative);bundles+=1;cache_calls+=1
            checked(x["weight"]);checked(x["data_arrays"]);checked(x["scaler"])
        models[arm]=a;steps+=a["optimizer_steps"]
    assert all(models[a]["initial"]["selected"]==models["B0"]["initial"]["selected"] and models[a]["initial"]["head"]==models["B0"]["initial"]["head"] and models[a]["initial"]["frozen_except_null"]==models["B0"]["initial"]["frozen_except_null"] for a in ("B1","B2","B3"))
    assert all(models[a]["ledger"]==models["B0"]["ledger"] for a in ("B1","B2","B3"))
    neutral=lambda rows:[{k:v for k,v in x.items() if k not in ("x_t_sha256","use_cache")} for x in rows]
    for arm in ("M","F"):
        if arm not in models:continue
        assert models[arm]["initial"]==models["Q"]["initial"]
        assert neutral(models[arm]["ledger"])==neutral(models["Q"]["ledger"])
    assert sum(x["use_cache"] for x in models["M"]["ledger"])==cfg["follow_steps"]//2
    for arm in ("M","F"):
        if arm not in models:continue
        for b in [x for x in models[arm]["cache_bindings"] if x["refresh"]==0]:
            original=next(x for x in models["Q"]["cache_bindings"] if x["refresh"]==0 and x["case"]==b["case"])
            same_arrays(checked(b["trace"]),checked(original["trace"]))
    a=read(root/"rollout/result.json");b=read(root/"rollout_replay/result.json")
    for arm in arms:
        assert a["q_scores"][arm]["scores"]==b["q_scores"][arm]["scores"]
        same_arrays(checked(a["q_scores"][arm]["predictions"]),checked(b["q_scores"][arm]["predictions"]));bundles+=1
    noise={}
    for x,y in zip(a["bindings"],b["bindings"]):
        assert all(x[k]==y[k] for k in ("arm","case","variant","control","seed"))
        same_arrays(checked(x["trace"]),checked(y["trace"]));bundles+=1
        sa=read(checked(x["sampler"]));assert sa==read(checked(y["sampler"]))
        if x["control"]=="correct":noise.setdefault((x["case"],x["variant"]),set()).add(sa["noise_sha256"])
        with np.load(checked(x["trace"])) as tr:assert np.array_equal(tr["sample"][tr["hard"]],tr["condition"][tr["hard"]])
    assert all(len(x)==1 for x in noise.values()) and len(a["bindings"])==len(b["bindings"])
    n=read(root/"native/result.json");assert n==read(root/"native_replay/result.json");native_arrays=0;reports=0
    for folder in (root/"native").iterdir():
        if not folder.is_dir():continue
        for p in folder.iterdir():
            other=root/"native_replay"/folder.name/p.name
            if p.suffix==".npy":assert np.array_equal(np.load(p),np.load(other));native_arrays+=1
            else:assert read(p)==read(other);reports+=1
    populations={}
    for record in n["records"]:
        if record["control"]=="correct":
            proxy=record["source_proxy_readouts"];populations.setdefault((record["case"],record["variant"]),set()).add((proxy["support_elements"],proxy["velocity_elements"]))
        domains=record["hard_domains"];assert sum(x["templates"] for x in domains.values())==222
    assert all(len(v)==1 for v in populations.values())
    frozen=subprocess.check_output(["git","diff","--name-only",cfg["baseline"],"--","src","configs","reports/r20_release","docs/R20_PROTOCOL.md","reports/R20双端联合视野与采样时序实验报告.md"],text=True).splitlines()
    frozen=[p for p in frozen if not p.startswith("src/dunhuang/research/context_r21/")];assert not frozen,frozen
    result=dict(code_commit=commit(),array_bundles_exact=bundles,update_pairs_exact=updates,training_optimizer_steps=steps,replay_optimizer_steps=0,
        source_records=len(data["records"]),train_groups=len(data["train_groups"]),dev_exposed_groups=len(data["dev_groups"]),
        first_evaluation_sampler_calls=len(a["bindings"])+2,cache_sampler_calls=cache_calls,
        replay_sampler_calls_exact=len(a["bindings"])+2+cache_calls,native_arrays_exact=native_arrays,physical_reports_exact=reports,
        source_arrays_rebuilt=len(data["records"]),frozen_changed_paths=frozen,F_executed="F" in models,
        whole_song_generations=0,operational_qualified=False)
    write(out,result);print("R21_VERIFIED",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out");a=p.parse_args();run(a.root,a.out)
