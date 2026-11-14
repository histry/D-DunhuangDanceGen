"""Independent saved scientific artifact replay; no optimizer or new sampling."""
import argparse,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read

def run(audit,training):
    audit=Path(audit);training=Path(training);assert read(audit/"p0/result.json")==read(audit/"p0_replay/result.json")
    counts={}
    for phase in ("root","hermite"):
        assert read(audit/phase/"result.json")==read(audit/(phase+"_replay")/"result.json");n=0
        for b in read(audit/phase/"protocol.json")["bindings"]:
            x=np.load(checked(b["native"]));old=np.load(checked(b["original"]))
            if phase=="root":
                assert np.array_equal(x[:,7:],old[:,7:]) and np.array_equal(x[:,:4],old[:,:4]) and np.array_equal(x[:,5],old[:,5])
                assert np.array_equal(x[:3],old[:3]) and np.array_equal(x[-3:],old[-3:])
            else:assert np.array_equal(x[:60],old[:60]) and np.array_equal(x[90:],old[90:])
            folder=Path(b["native"]["path"]).parent;other=audit/(phase+"_replay")/folder.name
            for file in ("native.npy","fk.npy"):assert np.array_equal(np.load(folder/file),np.load(other/file));n+=1
        counts[phase]=n
    for arm in ("C0","C1"):
        x=read(training/arm/"result.json");y=read(training/(arm+"_replay")/"result.json")
        assert x["validation"]==y["validation"] and y["optimizer_steps"]==0 and x["optimizer_steps"]==200
        assert x["backbone_digest"]==y["backbone_digest"] and not x["backbone_updated"];checked(x["checkpoint"])
    g=read(training/"generation/result.json");assert g==read(training/"generation_replay/result.json");n=0
    for b in read(training/"generation/protocol.json")["bindings"]:
        folder=Path(checked(b["native"])).parent;other=training/"generation_replay"/folder.name
        for f in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy","edited_native.npy","edited_fk.npy"):
            assert np.array_equal(np.load(folder/f),np.load(other/f));n+=1
        assert read(folder/"edited_physical.json")==read(other/"edited_physical.json")
    assert n==256
    formal=subprocess.check_output(["git","diff","--name-only","4423535","--","src/dunhuang","configs"],text=True).splitlines()
    formal=[v for v in formal if not v.startswith("src/dunhuang/research/")];assert not formal
    result=dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),capacity_arrays=counts,neural_arrays=n,p0_exact_replay=True,
        training_steps_per_arm=200,replay_optimizer_steps=0,formal_changed_paths=formal,whole_song_generations=0,operational_qualified=False)
    write(training/"independent_checks.json",result);print("R10_VERIFIED",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--audit",required=True);p.add_argument("--training",required=True);a=p.parse_args();run(a.audit,a.training)
