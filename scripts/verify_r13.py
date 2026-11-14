"""Independent saved-science verification; no optimizer may be constructed."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers
from dunhuang.research.source_tasks.common import same_arrays

def run(root):
    root=Path(root);forbid_optimizers();assert read(root/"official/result.json")==read(root/"official_replay/result.json")
    a=read(root/"teacher/result.json");b=read(root/"teacher_replay/result.json")
    assert a["records"]==b["records"] and a["hidden_target_poison_exact"] and b["optimizer_steps"]==0
    same_arrays(root/"teacher/probes.npz",root/"teacher_replay/probes.npz")
    for binding in a["bindings"]:same_arrays(checked(binding["rollout"]),root/"teacher_replay"/Path(binding["rollout"]["path"]).name)
    assert read(root/"teacher_native/result.json")==read(root/"teacher_native_replay/result.json")
    stages=["tiny"]+(["main"] if (root/"main/result.json").exists() else [])
    rollouts=0;native_arrays=0;train_steps=0
    for stage in stages:
        first=read(root/stage/"result.json");again=read(root/(stage+"_replay")/"result.json")
        assert first["q_scores"]==again["q_scores"] and first["final_digest"]==again["final_digest"]
        assert first["frozen_digest"]==again["frozen_digest"] and again["optimizer_steps"]==0
        assert first["selected_motion_backbone_updated"] and first["music_encoder_frozen"]
        assert read(root/stage/"interface.json")==read(root/(stage+"_replay")/"interface.json")
        same_arrays(root/stage/"q_eval.npz",root/(stage+"_replay")/"q_eval.npz");same_arrays(root/stage/"initial.npz",root/(stage+"_replay")/"initial.npz")
        train_steps+=first["optimizer_steps"]
        g=read(root/(stage+"_rollout")/"result.json");gr=read(root/(stage+"_rollout_replay")/"result.json")
        for left,right in zip(g["bindings"],gr["bindings"]):
            assert left["case"]==right["case"] and left["arm"]==right["arm"] and left["driver"]==right["driver"]
            same_arrays(checked(left["trace"]),checked(right["trace"]));rollouts+=1
            assert read(Path(checked(left["trace"])).parent/"sampler.json")==read(Path(checked(right["trace"])).parent/"sampler.json")
        nr=read(root/(stage+"_native")/"result.json");assert nr==read(root/(stage+"_native_replay")/"result.json")
        for r in nr["records"]:
            name=str(r["case"])+"_"+r["arm"]+"_"+r["driver"]
            for f in ("native.npy","fk.npy"):
                assert np.array_equal(np.load(root/(stage+"_native")/name/f),np.load(root/(stage+"_native_replay")/name/f));native_arrays+=1
            assert read(root/(stage+"_native")/name/"physical.json")==read(root/(stage+"_native_replay")/name/"physical.json")
    tiny=read(root/"tiny_native/result.json")
    assert "main" not in stages or tiny["tiny_training_admitted"]
    assert tiny["tiny_training_admitted"] or not (root/"main").exists()
    frozen=subprocess.check_output(["git","diff","--name-only","04f817b","--","src/dunhuang/research/motion_bridge","src/dunhuang/research/condition_dependency","reports/r12_summary.json","reports/r12_checks.json","reports/r12_receipt.json"],text=True).splitlines();assert not frozen
    formal=subprocess.check_output(["git","diff","--name-only","04f817b","--","src/dunhuang","configs"],text=True).splitlines()
    formal=[p for p in formal if not p.startswith("src/dunhuang/research/")];assert not formal
    receipts={}
    for stage in ["audit","tiny"]+(["main"] if "main" in stages else []):
        p=Path(str(root)+"_"+stage+"_receipts");assert (p/"exit_status.txt").read_text().strip()=="0"
        receipts[stage]=(p/"execution_commit.txt").read_text().strip()
    result=dict(code_commit=commit(),frozen_r12_changed_paths=frozen,formal_changed_paths=formal,teacher_probe_exact=True,
        teacher_rollouts_exact=4,denoiser_rollouts_exact=rollouts,native_arrays_exact=native_arrays,native_results_exact=True,
        q_eval_exact=True,self_policy_caches_replayed=sum(len(read(root/stage/"result.json")["self_caches"]) for stage in stages),
        training_optimizer_steps=train_steps,replay_optimizer_steps=0,optimizer_steps=0,music_encoder_frozen=True,stages=stages,
        new_whole_song_generations=0,operational_qualified=False,receipts=receipts)
    write(root/"independent_checks.json",result);print("R13_VERIFIED",result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
