"""Small fixed real-generation comparison; counterfactuals alter soft input only."""
import argparse,sys,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data,fixed_rows
from dunhuang.research.primitive_reperformance.environment import worker_environment
from dunhuang.research.primitive_reperformance.study import ARRAYS
from dunhuang.research.bridge_inpainting.study import audit_native,region,source_error
from .codec import encode,restore

def run(cfg,training,out,replay=None):
    out=fresh(out);pf=Path(cfg["preflight"]);assets=read(checked(read(pf/"protocol.json")["assets"]))
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    data=Data(np.load(checked(assets["scaler"])));items=[]
    for i,row in enumerate(fixed_rows(read(pf/"rows.json"),"val",4,cfg["data_seed"])):
        items.append(dict(name="val_"+str(i),native=data.native(row),audio=row["audio"],start=row["start_frame"],feature=np.load(checked(row["jukebox"]))[None],target=row))
    root=Path("outputs/reperformance_r8_02");candidates=read(root/"boundary_audit/candidates.json")
    for c in candidates:
        items.append(dict(name=c["name"],native=np.load(checked(c["native"])),audio=c["audio"],start=0,feature=np.load(checked(c["features"])),target=c))
    contexts={}
    for item in items:
        values,o=encode(item["native"][:150]);mask=np.zeros((1,150,151),bool);mask[:,52:60,4:]=True;mask[:,90:98,4:]=True
        f=out/(item["name"]+"_context.npz");np.savez_compressed(f,values=values[None],mask=mask)
        item.update(context=ref(f),origin=o);contexts[item["name"]]=ref(f)
        f=out/(item["name"]+"_features.npy");np.save(f,item["feature"]);item["features"]=ref(f)
    checkpoints={arm:read(Path(training)/arm/"result.json")["checkpoint"] for arm in cfg.get("checkpoint_arms",["C0","C1"])}
    records=[];bindings=[]
    for i,item in enumerate(items):
        old=item["native"];truth_fk,truth_physical=audit_native(old);noise=set()
        arms=cfg.get("generation_arms") or (["zero","C0","C1"]+(["C1_removed","C1_wrong"] if item["name"].startswith("val_") else []))
        for arm in arms:
            folder=out/(item["name"]+"_"+arm);folder.mkdir();wrong=items[(i+1)%4] if item["name"].startswith("val_") else items[4+(i-3)%4]
            q=dict(assets,context=item["context"],features=item["features"],audio=item["audio"],audio_start_frame=item["start"],mode="anchored",anchors=[],frames=150,windows=1,
                output=str(folder.resolve()),research_r10=True,r10_adapter=None if arm=="zero" else checkpoints[arm[:2]],
                soft_control="removed" if arm.endswith("removed") else "wrong" if arm.endswith("wrong") else "observed",wrong_context=wrong["context"],
                root_origin_native=item["origin"].tolist(),seed=cfg["seed"]+100+i,overlap_policy="mean")
            write(folder/"request.json",q)
            for python,module in ((assets["edge_python"],"dunhuang.research.condition_dependency.worker"),(assets["native_python"],"dunhuang.physical")):
                with (folder/(module.split(".")[-1]+".log")).open("w") as log:subprocess.run([python,"-m",module,str(folder/"request.json")],check=True,stdout=log,stderr=subprocess.STDOUT,env=worker_environment())
            sample=restore(np.load(folder/"native.npy"),item["origin"]);x=old.copy();x[60:90]=sample[60:90]
            assert np.array_equal(x[:60],old[:60]) and np.array_equal(x[90:],old[90:])
            fk,physical=audit_native(x);_,roi=audit_native(x[:150]);np.save(folder/"edited_native.npy",x);np.save(folder/"edited_fk.npy",fk);write(folder/"edited_physical.json",physical);write(folder/"roi_physical.json",roi)
            row=dict(case=item["name"],arm=arm,joins=[region(fk,t-3,t+3) for t in (60,90)],original_joins=[region(truth_fk,t-3,t+3) for t in (60,90)],
                root_rmse_m=float(np.sqrt(np.mean((x[60:90,4:7]-old[60:90,4:7])**2))),roi_native_ok=roi["absolute"]["ok"],global_native_ok=physical["absolute"]["ok"],
                audit=physical["audit"],target_native_ok=truth_physical["absolute"]["ok"])
            A=matrices(x[60:90,7:].reshape(-1,24,6));B=matrices(old[60:90,7:].reshape(-1,24,6))
            row["rotation_error_rad"]=float(Rotation.from_matrix((A.swapaxes(-1,-2)@B).reshape(-1,3,3)).magnitude().mean())
            if not item["name"].startswith("val_"):row["source_error_rad"]=source_error(x,read(checked(item["target"]["event_plan"])),60,90)
            g=read(folder/"generation.json");assert g["weights_before"]==g["weights_after"];noise.add(g["sampler"]["noise_sha256"])
            if cfg.get("comparison_generation"):
                original=read(Path(cfg["comparison_generation"])/(item["name"]+"_C1")/"generation.json")
                assert g["sampler"]["noise_sha256"]==original["sampler"]["noise_sha256"]
            records.append(row);bindings.append(dict(case=item["name"],arm=arm,native=ref(folder/"edited_native.npy"),fk=ref(folder/"edited_fk.npy"),target=item["target"]))
            if replay:
                other=Path(replay)/folder.name
                for f in ARRAYS+("edited_native.npy","edited_fk.npy"):assert np.array_equal(np.load(folder/f),np.load(other/f))
                assert physical==read(other/"edited_physical.json") and roi==read(other/"roi_physical.json")
            print("R10_GENERATION",item["name"],arm,row["roi_native_ok"],flush=True)
        assert len(noise)==1
    result=dict(records=records,new_neural_generations=len(records),operational_qualified=False,new_whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,training={k:ref(Path(training)/k/"result.json") for k in checkpoints},bindings=bindings))
    if replay:assert result==read(Path(replay)/"result.json")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--training",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.training,a.out,a.replay_from)
