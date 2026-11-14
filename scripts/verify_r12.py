"""Fresh-process replay verification; never constructs an optimizer."""
import argparse,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit
from dunhuang.research.source_tasks.common import same_arrays,forbid_optimizers

def run(root):
    root=Path(root);forbid_optimizers()
    a=read(root/"capacity/result.json");assert a==read(root/"capacity_replay/result.json")
    same_arrays(root/"capacity/predictions.npz",root/"capacity_replay/predictions.npz")
    assert a["training_admitted"] and a["val_tensor_reads"]==0
    cache=read(root/"cache/result.json");other=read(root/"cache_replay/result.json")
    assert np.array_equal(np.load(checked(cache["inv_rms"])),np.load(checked(other["inv_rms"])))
    for x,y in zip(cache["bindings"],other["bindings"]):
        same_arrays(checked(x["trajectory"]),checked(y["trajectory"]))
        assert x["row"]==y["row"] and x["noise_sha256"]==y["noise_sha256"]
    for arm in ("B3","B2"):
        orig=read(root/arm/"result.json");replay=read(root/(arm+"_replay")/"result.json")
        assert orig["optimizer_steps"]==400 and replay["optimizer_steps"]==0
        assert orig["monitor"]==replay["monitor"] and orig["backbone_digest"]==a["backbone_digest"]==replay["backbone_digest"]
        same_arrays(root/arm/"monitor.npz",root/(arm+"_replay")/"monitor.npz")
    g=read(root/"generation/result.json");gr=read(root/"generation_replay/result.json")
    arrays=0
    for x,y in zip(g["bindings"],gr["bindings"]):
        assert x["case"]==y["case"] and x["arm"]==y["arm"]
        for key in ("native","raw_last"):
            assert np.array_equal(np.load(checked(x[key])),np.load(checked(y[key])));arrays+=1
        same_arrays(checked(x["trace"]),checked(y["trace"]))
        assert read(checked(x["trace_info"]))==read(checked(y["trace_info"]))
        folder=Path(checked(x["native"])).parent;other=Path(checked(y["native"])).parent
        assert np.array_equal(np.load(folder/"normalized.npy"),np.load(other/"normalized.npy"));arrays+=1
        assert read(folder/"sampler.json")==read(other/"sampler.json")
    result=read(root/"analysis/result.json");assert result==read(root/"analysis_replay/result.json")
    fk_arrays=0
    for r in result["records"]:
        p=r["case"]+"_"+r["arm"]
        assert np.array_equal(np.load(root/"analysis"/p/"fk.npy"),np.load(root/"analysis_replay"/p/"fk.npy"));fk_arrays+=1
        for f in ("physical.json","roi_physical.json"):assert read(root/"analysis"/p/f)==read(root/"analysis_replay"/p/f)
    formal=subprocess.check_output(["git","diff","--name-only","75868e8","--","src/dunhuang","configs"],text=True).splitlines()
    formal=[p for p in formal if not p.startswith("src/dunhuang/research/")];assert not formal
    frozen=subprocess.check_output(["git","diff","--name-only","75868e8","--","src/dunhuang/research/condition_dependency","src/dunhuang/research/state_bridge","reports/r10_summary.json","reports/r11_summary.json"],text=True).splitlines();assert not frozen
    receipts={}
    for phase in ("capacity","train","generate"):
        p=Path(str(root)+"_"+phase+"_receipts")
        assert (p/"exit_status.txt").read_text().strip()=="0"
        receipts[phase]=(p/"execution_commit.txt").read_text().strip()
    check=dict(code_commit=commit(),capacity_exact_replay=True,cache_trajectory_arrays=len(cache["bindings"])*4,
        monitor_arrays=192,generation_arrays=arrays,generation_trace_arrays=len(g["bindings"])*2,native_fk_arrays=fk_arrays,
        native_jsons_exact=True,optimizer_steps=0,replay_optimizer_steps=0,original_backbone_unchanged=True,
        formal_changed_paths=formal,frozen_changed_paths=frozen,receipts=receipts,new_whole_song_generations=0,operational_qualified=False)
    write(root/"independent_checks.json",check);print("R12_VERIFIED",check,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
