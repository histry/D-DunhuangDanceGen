"""Read-only independent replay seals, protected-context checks and original Native outcomes."""
import argparse,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write,sha
from dunhuang.research.source_capacity.common import read
from dunhuang.research.boundary_context.common import Data

def run(root):
    root=Path(root);train=read(root/"train/result.json");replay=read(root/"train_replay/result.json");assert train["validation"]==replay["validation"] and replay["optimizer_steps"]==0
    assert train["backbone_before"]==train["backbone_after"]==replay["backbone_before"]==replay["backbone_after"]
    assert train["optimizer_steps"]==300;checked(train["checkpoint"])
    result=read(root/"generation/result.json");assert result==read(root/"generation_replay/result.json")
    p=read(root/"generation/protocol.json");pre=read(checked(p["preflight"]));assets=read(checked(pre["assets"]));data=Data(np.load(checked(assets["scaler"])));count=0
    for b in p["bindings"]:
        x=np.load(checked(b["native"]));target=data.native(b["target"]);o=target[0,4:7].copy();o[1]=0;target[:,4:7]-=o
        assert np.array_equal(x[:60],target[:60]) and np.array_equal(x[90:],target[90:])
        folder=Path(b["native"]["path"]).parent;other=root/"generation_replay"/folder.name
        for f in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy","edited_native.npy","edited_fk.npy"):
            assert np.array_equal(np.load(folder/f),np.load(other/f));count+=1
        assert read(folder/"edited_physical.json")==read(other/"edited_physical.json")
    assert count==128
    app=read(root/"application/result.json");assert app==read(root/"application_replay/result.json");app_arrays=0
    for binding in read(root/"application/protocol.json")["bindings"]:
        x=np.load(checked(binding["native"]));old=np.load(checked(binding["original"]))
        assert np.array_equal(x[:60],old[:60]) and np.array_equal(x[90:],old[90:])
        folder=Path(binding["native"]["path"]).parent;other=root/"application_replay"/folder.name
        for file in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy","edited_native.npy","edited_fk.npy"):
            assert np.array_equal(np.load(folder/file),np.load(other/file));app_arrays+=1
    assert app_arrays==64
    sealed=dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),frozen_adapter_sha256=train["checkpoint"]["sha256"],
        optimizer_training_steps=300,replay_optimizer_steps=0,exact_generation_arrays=count,exact_application_arrays=app_arrays,protected_context_exact=True,original_native_unchanged=True,operational_qualified=False,new_whole_song_generations=0)
    write(root/"independent_checks.json",sealed);print("B2_INDEPENDENT",sealed,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();run(a.out)
