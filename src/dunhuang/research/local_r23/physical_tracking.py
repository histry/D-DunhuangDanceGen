"""Explicit PHC input pair and simulator readiness; never pretend kinematic export is tracking."""
import argparse,importlib.util,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.contracts import matrices,Q
from .common import inputs,finish,arrays,read,fresh,forbid_optimizers

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();root,pc,pop=inputs(cfg)
    source=next(r for r in pop["records"] if r["case"]==24);z=np.load(checked(source["arrays"]))
    oldnative=read(root/"native/protocol.json")["saved"]
    index=next(k for k,b in oldnative.items() if (b["arm"],b["case"],b["variant"],b["control"])==("G2",24,0,"correct"))
    generated=np.load(root/"native"/index/"native.npy")
    pairs={}
    for name,x in (("source",z["native"]),("G2",generated)):
        R=matrices(x[:,7:].reshape(225,24,6));R=R.copy();R[:,0]=Q@R[:,0]
        pairs[name+"_native"]=x;pairs[name+"_root_zup"]=x[:,4:7]@Q.T;pairs[name+"_local_rotation_zup"]=R
    arrays(out,"tracking_input_pair.npz",pairs,replay)
    # Check every available server interpreter separately; do not contaminate an EDGE environment.
    modules=["isaacgym","isaaclab","mujoco","smpl_sim"];environments={}
    for folder in sorted(Path("/home/disk/lsm/conda_envs").glob("*")):
        py=folder/"bin/python"
        if py.is_file():
            code="import importlib.util,json;print(json.dumps({m:importlib.util.find_spec(m) is not None for m in "+repr(modules)+"}))"
            p=subprocess.run([str(py),"-c",code],capture_output=True,text=True)
            import json
            environments[folder.name]=dict(returncode=p.returncode,modules=json.loads(p.stdout) if p.returncode==0 else None)
    contracts=Path("inputs/phc_r23/contract.json")
    # Readiness cannot be inferred merely from a simulator package; body, policy and retargeting must be verified.
    status="NOT_EXECUTED_MISSING_VERIFIED_PHC_RUNTIME"
    finish(out,dict(status=status,environment_inventory=environments,verified_runtime_contract_present=contracts.is_file(),
        input_pair=["case24_native_source","case24_G2_cfg2"],source_and_generated_world_clock_identical=True,
        export_coordinate_conversion="Y-up native to SMPL Z-up via Q, root global rotation Q@R0; PHC upright heading retarget remains unverified",
        pair_native_source=source["arrays"],tracking_rollouts=0,teacher_pairs=0,distillation_updates=0,
        missing=["Verified PHC source/policy/simulator/body runtime and upright coordinate retarget contract"],
        public_runtime_reference="https://github.com/ZhengyiLuo/PHC",contacts_truth="UNKNOWN",optimizer_steps=0),cfg,replay)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
