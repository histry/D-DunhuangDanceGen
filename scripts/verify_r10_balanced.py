"""No optimization: replay all C2 artifacts and match registered frozen C1 noise."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);assert read(root/"units/result.json")==read(root/"units_replay/result.json")
    assert np.array_equal(np.load(root/"units/inv_rms.npy"),np.load(root/"units_replay/inv_rms.npy"))
    units=read(root/"units/result.json");assert units["validation_tensor_reads"]==0 and units["test_tensor_reads"]==0
    a=read(root/"C2/result.json");b=read(root/"C2_replay/result.json")
    assert a["optimizer_steps"]==200 and b["optimizer_steps"]==0 and a["validation"]==b["validation"]
    assert a["parameters"]==341888 and a["backbone_digest"]==b["backbone_digest"] and not a["backbone_updated"]
    checked(a["checkpoint"])
    g=read(root/"generation/result.json");assert g==read(root/"generation_replay/result.json");n=0
    for binding in read(root/"generation/protocol.json")["bindings"]:
        folder=Path(checked(binding["native"])).parent;other=root/"generation_replay"/folder.name
        for file in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy","edited_native.npy","edited_fk.npy"):
            assert np.array_equal(np.load(folder/file),np.load(other/file));n+=1
        assert read(folder/"edited_physical.json")==read(other/"edited_physical.json")
        old=Path("outputs/content_r10_01/generation")/(binding["case"]+"_C1")/"generation.json"
        assert read(folder/"generation.json")["sampler"]["noise_sha256"]==read(old)["sampler"]["noise_sha256"]
    assert n==64
    write(root/"independent_checks.json",dict(code_commit=commit(),neural_arrays=n,train_units_exact_replay=True,units_val_test_tensor_reads=0,
        training_steps=200,replay_optimizer_steps=0,parameters=341888,backbone_updated=False,whole_song_generations=0,operational_qualified=False))
    print("R10_C2_VERIFIED",n,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
