"""Independent saved artifact verifier; no neural generation and no optimizer."""
import argparse,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write,sha
from dunhuang.research.source_capacity.common import read
def run(root):
    root=Path(root);source=read(root/"source/result.json");replay=read(root/"source_replay/result.json")
    assert source["records"]==replay["records"];count=0
    p=read(root/"source/protocol.json")
    for b in p["bindings"]:
        folder=Path(checked(b["native"])).parent
        if b["arm"]=="T0":continue
        other=root/"source_replay"/str(b["pair"])/b["source"]/b["arm"]
        for f in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy"):
            assert np.array_equal(np.load(folder/f),np.load(other/f));count+=1
        g=read(folder/"generation.json");assert g["weights_before"]==g["weights_after"] and g["adapter"] is None
    assert count==240
    boundary=read(root/"boundary/result.json");assert boundary==read(root/"boundary_replay/result.json");bc=0;rootresidual=0.
    for b in read(root/"boundary/protocol.json")["bindings"]:
        x=np.load(checked(b["native"]));old=np.load(checked(b["original"]));a,z=b["config"]["edit"]
        assert np.array_equal(x[:a],old[:a]) and np.array_equal(x[z:],old[z:])
        for t in (a,z):
            residual=np.diff(x[t-3:t+3,4:7].astype(float),n=3,axis=0)
            rootresidual=max(rootresidual,float(abs(residual).max()))
        other=root/"boundary_replay"/b["name"]
        for f in ("native.npy","fk.npy"):
            assert np.array_equal(np.load(Path(b["native"]["path"]).parent/f),np.load(other/f));bc+=1
    changed=subprocess.check_output(["git","diff","--name-only","0911637","--","src/dunhuang","configs"],text=True).splitlines()
    formal=[p for p in changed if not p.startswith("src/dunhuang/research/")]
    assert not formal
    result=dict(exact_new_source_arrays=count,frozen_trace_arrays=source["traced_frozen_arrays_exact"],boundary_arrays=bc,
        independent_root_d3_stencil_max_m=rootresidual,code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        formal_changed_paths=formal,new_neural_weight_updates=0,new_whole_song_generations=0,operational_qualified=False)
    write(root/"independent_checks.json",result);print("R9_INDEPENDENT",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();run(a.out)
