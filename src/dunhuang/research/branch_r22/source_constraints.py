"""Additional read-only source endpoint proof; never changes population or gates."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,forbid_optimizers
from dunhuang.research.masked_denoiser.metrics import decode

def root_bounds(native,hard,limits,fps=30):
    x=np.asarray(native);h=np.asarray(hard,dtype=bool);assert x.ndim==2 and x.shape[1]==151 and h.shape==x.shape
    known=h[:,4:7].all(-1);root=x[:,4:7].astype(np.float64)
    fixed_pairs=known[:-1]&known[1:]
    v=float(np.max(abs(np.diff(root[:,1]))[fixed_pairs])*fps) if fixed_pairs.any() else None
    endpoints=bool(known[0] and known[-1]);distance=float(np.linalg.norm((root[-1]-root[0])[[0,2]])) if endpoints else None
    speed=distance/((len(x)-1)/fps) if endpoints else None
    tests=dict(root_horizontal_net_displacement_m=None if distance is None else distance>limits["root_horizontal_net_displacement_m"],
        root_horizontal_drift_speed_mps=None if speed is None else speed>limits["root_horizontal_drift_speed_mps"],
        root_vertical_speed_mps_max=None if v is None else v>limits["root_vertical_speed_mps_max"])
    return dict(endpoint_exact=endpoints,net_displacement_m=distance,net_drift_speed_mps=speed,fixed_vertical_speed_lower_bound_mps=v,
        unavoidable_failures=[k for k,value in tests.items() if value],statuses=tests,
        complete_feasibility="EXCLUDED_BY_FIXED_ROOT" if any(v is True for v in tests.values()) else "UNKNOWN",
        floor_support_or_global_distribution_bound_claimed=False)

def run(root,out,replay=None):
    root=Path(root);out=Path(out);assert not out.exists();forbid_optimizers()
    pre=read(root/"data/protocol.json");data=read(root/"data/result.json");cfg=pre["config"]
    scaler=np.load(checked(pre["assets"]["scaler"]));rows=[]
    for r in data["records"]:
        z=np.load(checked(r["arrays"]));represented=decode(z["encoded"],scaler,z["native"])
        physical=read(checked(r["source_physical"]))["absolute"];variants={}
        for variant,(left,right) in enumerate(cfg["contexts"]):
            h=np.zeros((225,151),bool);h[:left,4:]=True;h[225-right:,4:]=True
            original=root_bounds(z["native"],h,physical["limits"])
            assert abs(original["net_drift_speed_mps"]-physical["audit"]["root_horizontal_drift_speed_mps"])<1e-6
            variants[str(variant)]=root_bounds(represented,h,physical["limits"])
        rows.append(dict(case=r["case"],role=r["role"],source_group=r["left"]["source_group"],variants=variants))
    result=dict(records=rows,scope="SUPPLEMENTAL_SOURCE_ONLY_DIAGNOSTIC; no generation read, no population or threshold changes",
        optimizer_steps=0,code_commit=commit(),whole_song_generations=0,operational_qualified=False)
    if replay:assert result==read(replay)
    write(out,result);print("R22_FIXED_ROOT_AUDIT",[(r["case"],r["role"]) for r in rows if any(v["unavoidable_failures"] for v in r["variants"].values())],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(a.root,a.out,a.replay_from)
