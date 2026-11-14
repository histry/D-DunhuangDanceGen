"""Read-only frozen responsibilities and necessary edit-domain checks."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.boundary_state_r14.bounds import immutable_jerk_bound
from .common import source

def run(cfg,out,replay=None):
    out=fresh(out);assets,rows,monitor,hashes=source(cfg,manifest=True);native_environment(assets)
    from contracts.physical_quality import PhysicalQualityLimits
    from motion_geometry.smpl24 import PARENTS,OFFSETS
    limit=PhysicalQualityLimits.from_environment().joint_jerk_mps3_max
    data=Data(np.load(checked(assets["scaler"])));arrays={};records=[]
    for role,population in (("old_monitor",[b["row"] for b in monitor]),("capacity_train",[b["row"] for b in rows])):
        for i,row in enumerate(population):
            x=data.native(row);fk,audit=audit_native(x)
            bound=immutable_jerk_bound(fk,limit,cfg["edit"],cfg["fps"])
            assert abs(bound["native_global_max"]-audit["audit"]["joint_jerk_mps3_max"])<1e-3
            records.append(dict(role=role,case=i,row=row,original_native_ok=audit["absolute"]["ok"],
                native_reasons=audit["absolute"]["reasons"],**bound))
            arrays[role+"_"+str(i)]=fk
    old=read(Path(cfg["r14_root"])/"audit/result.json")
    for r,b in zip(records[:4],old["necessary_bounds"]):
        assert r["necessary_status"]==b["necessary_status"] and r["witnesses"]==b["witnesses"]
    responsibility=[r for r in old["records"] if r["step"]==49]
    np.savez_compressed(out/"target_fk.npz",**arrays);np.savez(out/"skeleton.npz",parents=PARENTS,offsets=OFFSETS)
    result=dict(records=records,frozen_final_responsibility=responsibility,all_old_cases_retained=True,
        immutable_exclusions=sum(r["necessary_status"]!="UNKNOWN" for r in records),other_metrics="UNKNOWN",
        local_mechanism_gate_separate_from_native=True,optimizer_steps=0,new_full_window_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        threshold=limit,verified_r14_files=len(hashes),target_fk=ref(out/"target_fk.npz"),skeleton=ref(out/"skeleton.npz")))
    if replay:
        from dunhuang.research.source_tasks.common import same_arrays
        assert result==read(Path(replay)/"result.json")
        for name in ("target_fk.npz","skeleton.npz"):same_arrays(out/name,Path(replay)/name)
    print("R15_PREFLIGHT",len(records),result["immutable_exclusions"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
