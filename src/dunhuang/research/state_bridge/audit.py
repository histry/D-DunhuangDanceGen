"""R11 P0: saved arrays, oracle codec, unchanged Native; zero new sampling/training."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter,matrices
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.condition_dependency.codec import encode,restore
from dunhuang.research.bridge_inpainting.study import audit_native
from .common import cases,native_environment
from .metrics import states,seam,rotvec

def run(cfg,out,replay=None):
    out=fresh(out);items=cases(cfg);native_environment(items[0]["request"]);records=[];oracles=[];bindings=[]
    for item in items:
        name=item["name"];old=item["original"];old_fk,old_physical=audit_native(old)
        encoded,o=encode(old[:150]);scaler=np.load(checked(item["request"]["scaler"]))
        normalized=encoded*scaler["scale"]+scaler["bias"];decoded=restore(MotionAdapter.to_native(((normalized-scaler["bias"])/scaler["scale"]).astype(np.float32)),o)
        oracle=old.copy();oracle[60:90]=decoded[60:90];ofk,op=audit_native(oracle)
        angle=np.linalg.norm(rotvec(matrices(old[60:90,7:].reshape(-1,24,6)).swapaxes(-1,-2)@matrices(oracle[60:90,7:].reshape(-1,24,6))),axis=-1)
        oe=dict(case=name,native_max_abs=float(abs(oracle-old).max()),fk_max_abs_m=float(abs(ofk-old_fk).max()),pose_max_rad=float(angle.max()),
            exact_bytes=bool(np.array_equal(oracle,old)),outside_exact=bool(np.array_equal(oracle[:60],old[:60]) and np.array_equal(oracle[90:],old[90:])),
            normalized_edit_supported=bool((abs(normalized[60:90,4:])<=1+32*np.finfo(np.float32).eps).all()),affine_no_clipping=True)
        oe["passed"]=bool(oe["native_max_abs"]<=cfg["oracle_native_atol"] and oe["fk_max_abs_m"]<=cfg["oracle_fk_atol_m"] and oe["pose_max_rad"]<=cfg["oracle_angle_atol_rad"] and oe["outside_exact"])
        oracles.append(oe);folder=out/name;folder.mkdir();np.save(folder/"oracle_native.npy",oracle);np.save(folder/"oracle_fk.npy",ofk);write(folder/"oracle_physical.json",op)
        binding=dict(case=name,oracle=ref(folder/"oracle_native.npy"),oracle_fk=ref(folder/"oracle_fk.npy"))
        for arm in ("original","zero","C1","C2"):
            if arm=="original":x=old;fk=old_fk;physical=old_physical
            else:
                source=item["parent_folder"] if arm=="C2" else Path(cfg["comparison"])/"generation"/(name+"_"+arm)
                x=np.load(source/"edited_native.npy");fk,physical=audit_native(x)
                assert np.array_equal(fk,np.load(source/"edited_fk.npy")) and physical==read(source/"edited_physical.json")
                assert np.array_equal(x[:60],old[:60]) and np.array_equal(x[90:],old[90:])
            records.append(dict(case=name,arm=arm,states=states(x,fk,cfg["key_frames"]),seams=[seam(x,fk,t,cfg["decomposition_atol_m"]) for t in (60,90)],
                native_ok=physical["absolute"]["ok"],failure_reasons=physical["absolute"]["reasons"]))
        bindings.append(binding)
        if replay:
            for f in ("oracle_native.npy","oracle_fk.npy"):assert np.array_equal(np.load(folder/f),np.load(Path(replay)/name/f))
            assert op==read(Path(replay)/name/"oracle_physical.json")
    result=dict(records=records,oracles=oracles,oracle_passed=all(r["passed"] for r in oracles),optimizer_steps=0,new_neural_generations=0,new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,parent=ref(Path(cfg["parent"])/"generation/protocol.json"),bindings=bindings))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R11_ORACLE_AND_STATE",len(oracles),result["oracle_passed"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
