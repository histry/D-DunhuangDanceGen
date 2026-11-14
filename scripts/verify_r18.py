"""Fresh optimizer-free science verification, with separate training counters."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays
from dunhuang.research.task_denoising_r18.engine import population
from dunhuang.research.task_denoising_r18.gate import assess

def run(root,out=None):
    root=Path(root);forbid_optimizers();cfg=read(root/"p0/protocol.json")["config"];pre,fit,ev=population(cfg)
    arrays=0;bundles=0;reports=0;samplers=0;updates=0
    assert read(root/"p0/result.json")==read(root/"p0_replay/result.json")
    same_arrays(root/"p0/predictions.npz",root/"p0_replay/predictions.npz");bundles+=1
    gate=read(root/"tiny_gate.json");main=(root/"main").exists();assert main==gate["main_admitted"]
    initial={};noise_groups={}
    for phase in ["tiny"]+(["main"] if main else []):
        models={}
        for mode in ("self","unique"):
            p=root/phase/mode;r=read(p/"result.json");rr=read(root/phase/(mode+"_replay")/"result.json")
            assert r["scores"]==rr["scores"] and r["frozen_digest"]==rr["frozen_digest"]
            assert r["final_motion"]==rr["final_motion"] and r["final_head"]==rr["final_head"]
            assert r["optimizer_steps"]==cfg[phase+"_steps"] and rr["optimizer_steps"]==0
            assert r["initial_motion"]!=r["final_motion"] and r["fk_parity_max_m"]<=1e-5
            assert all(v["updates"]==cfg[phase+"_steps"]//3 and v["ranking_updates"]==cfg[phase+"_steps"]//6
                and v["motion_nonzero_gradients"]==v["updates"] for v in r["task_counts"].values())
            checked(r["checkpoint"]);same_arrays(p/"q_eval.npz",root/phase/(mode+"_replay")/"q_eval.npz");bundles+=1
            assert all(e["group"] in {f["left"]["source_group"] for f in fit} for e in r["ledger"])
            models[mode]=r;updates+=r["optimizer_steps"]
            if phase=="tiny":assert assess(r,cfg)==gate["models"][mode]
        a,b=models["self"],models["unique"];assert a["ledger"]==b["ledger"]
        assert a["initial_motion"]==b["initial_motion"] and a["initial_head"]==b["initial_head"]
        assert a["motion_parameters"]==b["motion_parameters"] and a["head_parameters"]==b["head_parameters"]
        initial[phase]=(a["initial_motion"],a["initial_head"])
        for binding in read(root/(phase+"_rollout")/"result.json")["bindings"]:
            name=Path(binding["trace"]["path"]).parent.name
            path=checked(binding["trace"]);other=root/(phase+"_rollout_replay")/name/"trace.npz"
            same_arrays(path,other);bundles+=1;samplers+=1
            r=read(checked(binding["sampler"]));rr=read(root/(phase+"_rollout_replay")/name/"sampler.json")
            p=r.pop("prefix");pp=rr.pop("prefix");assert r==rr
            noise_groups.setdefault((phase,binding["case"],binding["task"]),set()).add(r["noise_sha256"])
            assert r["all_edge_calls_local150"] and not r["output_pose_paste"]
            if p:assert p["sha256"]==pp["sha256"]
            tr=np.load(path);assert tr["sample"].shape==(225,151)
            assert np.array_equal(tr["sample"][tr["hard"]],tr["condition"][tr["hard"]])
            if binding["task"]=="free":assert not tr["known"][:,:13].any()
            if p:
                prefix=np.load(checked(p))["sample"][:75]
                assert np.array_equal(tr["sample"][:75,4:],prefix[:,4:]) and not tr["known"][75:,:13].any()
    assert all(len(group)==1 for group in noise_groups.values()),"Matched-noise contract differs across arms"
    if main:assert initial["tiny"]==initial["main"],"Main must restart, not select/warm-start Tiny"
    assert read(root/"native/result.json")==read(root/"native_replay/result.json")
    for folder in (root/"native").iterdir():
        if not folder.is_dir():continue
        for f in folder.iterdir():
            other=root/"native_replay"/folder.name/f.name
            if f.suffix==".npy":assert np.array_equal(np.load(f),np.load(other));arrays+=1
            else:assert read(f)==read(other);reports+=1
    diagnostics=root/"objective_diagnostic"
    if diagnostics.exists():
        assert read(diagnostics/"result.json")==read(root/"objective_diagnostic_replay/result.json")
        same_arrays(diagnostics/"gradient_grams.npz",root/"objective_diagnostic_replay/gradient_grams.npz");bundles+=1
        assert read(diagnostics/"result.json")["optimizer_steps"]==0
    frozen=subprocess.check_output(["git","diff","--name-only",cfg["baseline"],"--","src","configs","reports/r17_summary.json","reports/r17_receipt.json"],text=True).splitlines()
    frozen=[p for p in frozen if not p.startswith("src/dunhuang/research/task_denoising_r18/")];assert not frozen,frozen
    result=dict(code_commit=commit(),array_bundles_exact=bundles,native_arrays_exact=arrays,physical_reports_exact=reports,
        first_sampler_calls=samplers,replay_sampler_calls_exact=samplers,training_optimizer_steps=updates,replay_optimizer_steps=0,
        registered_fit_music_groups=8,actual_trained_music_groups=8 if main else 2,
        p0_exposed_eval_groups=4,heldout_actual_generation_groups=4 if main else 0,tiny_is_generalization=False,main_executed=main,
        matched_initialization_and_ledger=True,matched_sampler_noise_verified=True,unselected_and_music_frozen=True,frozen_changed_paths=frozen,
        generated_history_training=False,D3_executed=False,new_whole_song_generations=0,operational_qualified=False)
    target=Path(out) if out else root/"independent_checks.json"
    assert not target.exists(),"Independent checks must use a fresh output"
    write(target,result);print("R18_VERIFIED",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out");a=p.parse_args();run(a.root,a.out)
