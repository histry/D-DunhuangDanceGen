"""Fresh-process R22 verification; no optimizer and no output overwrite."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,same_arrays,forbid_optimizers
from dunhuang.research.context_r21.data import same_audio_pcm

def run(root,out=None):
    root=Path(root);out=Path(out) if out else root/"independent_checks.json";assert not out.exists()
    forbid_optimizers();data=read(root/"data/result.json");other=read(root/"data_replay/result.json")
    assert read(root/"data/audit.json")==read(root/"data_replay/audit.json")
    assert data["candidate_ledger"]==other["candidate_ledger"];bundles=0
    for a,b in zip(data["records"],other["records"]):
        same_arrays(checked(a["arrays"]),checked(b["arrays"]));bundles+=1
        assert same_audio_pcm(checked(a["audio225"]),checked(b["audio225"]))
    same_arrays(root/"data/rebuilt/parity.npz",root/"data_replay/rebuilt/parity.npz");bundles+=1
    physical_candidates=0
    for p in (root/"data/candidate_physical").glob("*.json"):
        assert read(p)==read(root/"data_replay/candidate_physical"/p.name);physical_candidates+=1
    assert read(root/"sanity/result.json")==read(root/"sanity_replay/result.json")
    same_arrays(root/"sanity/sample.npz",root/"sanity_replay/sample.npz");bundles+=1
    cfg=read(root/"data/audit_protocol.json")["config"];models={};initial=[];ledgers=[];steps=updates=0
    for arm in cfg["arms"]:
        a=read(root/f"models/{arm}/result.json");b=read(root/f"models/{arm}_replay/result.json")
        assert a["initial"]==b["initial"] and a["ledger"]==b["ledger"] and b["optimizer_steps"]==0
        assert len(a["ledger"])==cfg["steps"]==a["optimizer_steps"]
        assert {r["case"] for r in a["ledger"]}==set(data["train_cases"])
        initial.append(a["initial"]);ledgers.append(a["ledger"]);steps+=a["optimizer_steps"]
        assert a["update_observations"]==b["update_observations"];updates+=len(a["update_observations"])
        assert a["final_selected"]==b["final_selected"] and a["final_head"]==b["final_head"]
        for x,y in zip(a["curves"],b["curves"]):
            assert x["scores"]==y["scores"];same_arrays(checked(x["predictions"]),checked(y["predictions"]));bundles+=1
        assert a["final_scores"]==b["final_scores"];same_arrays(checked(a["final_q"]),checked(b["final_q"]));bundles+=1
        models[arm]=a
    assert all(x==initial[0] for x in initial) and all(x==ledgers[0] for x in ledgers)
    assert all(c["scores"]==models["G0"]["curves"][0]["scores"] for a in models.values() for c in a["curves"][:1])
    a=read(root/"rollout/result.json");b=read(root/"rollout_replay/result.json")
    for arm in cfg["arms"]:
        assert a["q_scores"][arm]["scores"]==b["q_scores"][arm]["scores"]
        same_arrays(checked(a["q_scores"][arm]["predictions"]),checked(b["q_scores"][arm]["predictions"]));bundles+=1
    noise={}
    for x,y in zip(a["bindings"],b["bindings"]):
        assert all(x[k]==y[k] for k in ("arm","case","variant","control","seed","music_control_binding"))
        same_arrays(checked(x["trace"]),checked(y["trace"]));bundles+=1
        sa=read(checked(x["sampler"]));assert sa==read(checked(y["sampler"]))
        noise.setdefault((x["case"],x["variant"]),set()).add(sa["noise_sha256"])
        with np.load(checked(x["trace"])) as tr:assert np.array_equal(tr["sample"][tr["hard"]],tr["condition"][tr["hard"]])
    assert len(a["bindings"])==len(b["bindings"])==72 and all(len(x)==1 for x in noise.values())
    assert read(root/"native/result.json")==read(root/"native_replay/result.json")
    arrays=reports=0
    for folder in (root/"native").iterdir():
        if not folder.is_dir():continue
        for p in folder.iterdir():
            other=root/"native_replay"/folder.name/p.name
            if p.suffix==".npy":assert np.array_equal(np.load(p),np.load(other));arrays+=1
            else:assert read(p)==read(other);reports+=1
    lodge=read(root/"lodge/result.json");lodge_b=read(root/"lodge_replay/result.json")
    same_arrays(checked(lodge["arrays"]),checked(lodge_b["arrays"]));bundles+=1
    for key in ("checkpoint","common22","native24_rejection","batched_sampler_calls","generated_trajectories"):assert lodge[key]==lodge_b[key]
    assert lodge["full_native_status"]=="NOT_EVALUABLE" and not lodge["comparable_to_R22_native"]
    paths=subprocess.check_output(["git","diff","--name-only",cfg["baseline"],"--","src","configs","reports/r21_release","docs/R21_PROTOCOL.md","reports/R21联合上下文边界与支撑实验报告.md"],text=True).splitlines()
    frozen=[p for p in paths if not p.startswith("src/dunhuang/research/branch_r22/")];assert not frozen,frozen
    result=dict(code_commit=commit(),array_bundles_exact=bundles,actual_update_pairs_exact=updates,training_optimizer_steps=steps,
        replay_optimizer_steps=0,source_records_rebuilt=28,source_audio_pcm_exact=28,candidate_native_reports_exact=physical_candidates,
        evaluation_sampler_calls=74,replay_evaluation_sampler_calls=74,final_trajectories_exact=72,native_arrays_exact=arrays,physical_reports_exact=reports,
        lodge_batched_sampler_calls=3,lodge_trajectories=6,lodge_replay_batched_sampler_calls=3,lodge_arrays_exact=True,
        frozen_changed_paths=frozen,whole_song_generations=0,operational_qualified=False)
    write(out,result);print("R22_VERIFIED",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out");a=p.parse_args();run(a.root,a.out)
