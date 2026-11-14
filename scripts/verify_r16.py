"""Fresh independent array/ledger verification, no fitting or sampling."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers,same_arrays
from dunhuang.research.continuous_r16.contract import views,continuous_pair
from dunhuang.research.continuous_r16.gate import qualification

def run(root):
    root=Path(root);forbid_optimizers();pre=read(root/"preflight/protocol.json");cfg=pre["config"];bundles=0;arrays=0
    assert read(root/"preflight/result.json")==read(root/"preflight_replay/result.json")
    for f in (root/"preflight").glob("*.npz"):same_arrays(f,root/"preflight_replay"/f.name);bundles+=1
    for b in pre["pairs"]:
        continuous_pair(b["left"],b["right"]);z=np.load(checked(b["arrays"]))
        assert np.array_equal(z["window_motion"],views(z["encoded"])) and np.array_equal(z["window_music"],views(z["music"]))
        assert np.array_equal(z["window_motion"][0,75:],z["window_motion"][1,:75])
        assert np.array_equal(z["values"][~z["known"]],np.zeros_like(z["values"][~z["known"]]))
    assert len({b["left"]["source_group"] for b in pre["pairs"]})==cfg["cases"]
    a=read(root/"samples/result.json");b=read(root/"samples_replay/result.json");assert len(a["bindings"])==len(b["bindings"])==24
    noise={}
    for x,y in zip(a["bindings"],b["bindings"]):
        assert (x["case"],x["arm"])==(y["case"],y["arm"]);same_arrays(checked(x["trace"]),checked(y["trace"]));bundles+=1
        rec=read(checked(x["sampler"]));assert rec==read(checked(y["sampler"]))
        noise.setdefault(x["case"],set()).add(rec["noise_sha256"])
        z=np.load(checked(x["trace"]));assert z["raw"].shape==(50,150,151)
        assert not z["condition"][~z["known"]].any()
        if x["arm"]!="C0":
            assert not z["hard"].any() and np.array_equal(z["sample"][0],z["raw"][-1])
            assert not rec["post_generation_pose_paste"]
        else:assert np.array_equal(z["sample"][z["hard"]],z["condition"][z["hard"]])
    assert all(len(v)==1 for v in noise.values()) and a["model_unchanged"] and b["model_unchanged"]
    n=read(root/"native/result.json");assert n==read(root/"native_replay/result.json")
    assert n["qualification"]==qualification(n["records"],cfg,read(root/"native/protocol.json")["native_jerk_limit"])
    for folder in (root/"native").iterdir():
        if not folder.is_dir():continue
        for f in ("native.npy","fk.npy","halo.npy","halo_fk.npy"):
            assert np.array_equal(np.load(folder/f),np.load(root/"native_replay"/folder.name/f));arrays+=1
        for f in ("physical.json","halo_physical.json"):assert read(folder/f)==read(root/"native_replay"/folder.name/f)
        case=int(folder.name.split("_",1)[0]);z=np.load(checked(pre["pairs"][case]["arrays"]))
        halo=np.load(folder/"halo.npy");assert np.array_equal(halo[:8],z["halo"][:8]) and np.array_equal(halo[158:],z["halo"][158:])
    frozen=subprocess.check_output(["git","diff","--name-only","da412fa","--","src/dunhuang","configs","reports/r15_summary.json","reports/r15_receipt.json","reports/r14_summary.json","tests/test_root_pose_r15.py"],text=True).splitlines()
    frozen=[p for p in frozen if not p.startswith("src/dunhuang/research/continuous_r16/")];assert not frozen
    receipt=Path(str(root)+"_receipts");assert (receipt/"exit_status.txt").read_text().strip()=="0"
    result=dict(code_commit=commit(),execution_commit=(receipt/"execution_commit.txt").read_text().strip(),
        array_bundles_exact=bundles,native_arrays_exact=arrays,native_reports_exact=48,sampler_calls_exact=24,
        matched_noise_verified=True,global_225_motion_music_clock_verified=True,c1_no_output_pose_paste=True,
        real_external_frames_unchanged=True,models_unchanged=True,frozen_changed_paths=frozen,optimizer_steps=0,
        replay_optimizer_steps=0,cross_window_training_admitted=n["qualification"]["cross_window_training_admitted"],
        new_whole_song_generations=0,operational_qualified=False)
    write(root/"independent_checks.json",result);print("R16_VERIFIED",result,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
