"""Compare quintic state reference with sealed R9 same intervals and finite radii."""
import argparse,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.io import checked,write,sha,pinned
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_state.study import activity
from dunhuang.research.bridge_inpainting.study import audit_native,source_error,region
from .hermite import reference

def run(cfg,out,replay=None):
    out=fresh(out);root=Path(cfg["r8"]);p=read(root/"boundary_audit/protocol.json");assets=read(checked(p["assets"]));sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    candidates=read(checked(p["candidates"]));bp=read(root/"bridge/protocol.json");records=[];bindings=[]
    a,b=60,90
    for c in candidates:
        old=np.load(checked(c["native"]));plan=read(checked(c["event_plan"]));original_fk=np.load(checked(c["fk"]));curve,meta=reference(old,a,b)
        for arm in ("inpaint","event_inpaint"):
            binding=next(v for v in bp["bindings"] if v["name"]==c["name"] and v["arm"]==arm);raw=np.load(checked(binding["motion"]))
            for domain in cfg["domains"]:
                name=c["name"]+"_"+arm+"_"+domain["name"];folder=out/name;folder.mkdir()
                row=dict(name=name,case=c["name"],arm=arm,domain=domain["name"],reference=meta,research_pass=False,native_ok=False)
                if curve is not None:
                    x=raw.copy();x[a:b,4:]=curve[a:b,4:]
                    delta=float(np.linalg.norm(x[a:b,4:7]-raw[a:b,4:7],axis=-1).max())
                    R=matrices(x[a:b,7:].reshape(-1,24,6));G=matrices(raw[a:b,7:].reshape(-1,24,6))
                    angle=float(Rotation.from_matrix((G.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().max())
                    row.update(root_modification_m=delta,rotation_modification_rad=angle)
                    if delta<=domain["root_cap_m"] and angle<=domain["rotation_cap_rad"]:
                        fk,physics=audit_native(x);_,roi=audit_native(x[:150]);joins=[region(fk,t-3,t+3) for t in (a,b)]
                        oldjoins=[region(original_fk,t-3,t+3) for t in (a,b)]
                        guards=dict(new_joins_nonregression=all(v["maximum"]<=u["maximum"]+1e-6 for v,u in zip(joins,oldjoins)),
                            source_nonregression=source_error(x,plan,a,b)<=source_error(raw,plan,a,b)+1e-6,
                            activity_noncollapse=activity(x,a,b)>=.9*activity(raw,a,b),
                            center_retains_improvement=region(fk,69,81)["p95"]<=.9*region(original_fk,69,81)["p95"])
                        row.update(status="BOUNDED_QUINTIC_REFERENCE",guards={k:bool(v) for k,v in guards.items()},research_pass=all(guards.values()),native_ok=physics["absolute"]["ok"] and roi["absolute"]["ok"],
                            joins=joins,source_before=source_error(raw,plan,a,b),source_after=source_error(x,plan,a,b),activity_before=activity(raw,a,b),activity_after=activity(x,a,b))
                        assert np.array_equal(x[:a],old[:a]) and np.array_equal(x[b:],old[b:]) and np.array_equal(x[:,:4],raw[:,:4])
                        np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",physics);write(folder/"roi_physical.json",roi)
                        bindings.append(dict(name=name,native=ref(folder/"native.npy"),fk=ref(folder/"fk.npy"),original=c["native"],candidate=binding["motion"]))
                        if replay:
                            for f in ("native.npy","fk.npy"):assert np.array_equal(np.load(folder/f),np.load(Path(replay)/name/f))
                            assert physics==read(Path(replay)/name/"physical.json")
                    else:row["status"]="REFERENCE_EXCEEDS_REGISTERED_TRUST_DOMAIN"
                else:row["status"]=meta["status"]
                records.append(row)
    result=dict(records=records,population=len(records),research_passes=sum(v["research_pass"] for v in records),native_passes=sum(v["native_ok"] for v in records),new_neural_generations=0,
        scope="quintic continuous state reference; finite capacity proposal, not a proof that other C2 trajectories cannot exist",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,parent=ref(root/"bridge/protocol.json"),bindings=bindings))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R10_HERMITE",len(records),result["research_passes"],result["native_passes"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
