"""Apply frozen B2 to the exact four R8 generated contexts; no source promotion."""
import argparse,subprocess,sys,copy
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.primitive_reperformance.environment import worker_environment
from dunhuang.research.primitive_reperformance.study import ARRAYS
from dunhuang.research.bridge_inpainting.study import audit_native,region,source_error

def run(parent,training,out,replay=None):
    out=fresh(out);parent=Path(parent);candidates=read(parent/"boundary_audit/candidates.json")
    assets=read(checked(read(parent/"boundary_audit/protocol.json")["assets"]));saved=read(parent/"bridge/protocol.json");trained=read(Path(training)/"result.json")
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    records=[];bindings=[]
    for i,c in enumerate(candidates):
        original=np.load(checked(c["native"]));plan=read(checked(c["event_plan"]));noise=set()
        for arm in ("zero","learned"):
            old=Path(checked(next(b for b in saved["bindings"] if b["name"]==c["name"] and b["arm"]=="inpaint")["motion"])).parent
            q=read(old/"request.json");q.update(research_b2=True,context=c["context"],b2_adapter=trained["checkpoint"] if arm=="learned" else None)
            folder=out/(c["name"]+"_"+arm);folder.mkdir();q["output"]=str(folder.resolve());write(folder/"request.json",q)
            for python,module in ((assets["edge_python"],"dunhuang.research.boundary_context.worker"),(assets["native_python"],"dunhuang.physical")):
                with (folder/(module.split(".")[-1]+".log")).open("w") as log:subprocess.run([python,"-m",module,str(folder/"request.json")],check=True,stdout=log,stderr=subprocess.STDOUT,env=worker_environment())
            if arm=="zero":
                for f in ARRAYS:assert np.array_equal(np.load(folder/f),np.load(old/f)),("R8 frozen inpaint changed",f)
            x=original.copy();x[60:90]=np.load(folder/"native.npy")[60:90];fk,physical=audit_native(x);_,roi=audit_native(x[:150])
            assert np.array_equal(x[:60],original[:60]) and np.array_equal(x[90:],original[90:])
            np.save(folder/"edited_native.npy",x);np.save(folder/"edited_fk.npy",fk);write(folder/"edited_physical.json",physical);write(folder/"roi_physical.json",roi)
            g=read(folder/"generation.json");assert g["weights_before"]==g["weights_after"];noise.add(g["sampler"]["noise_sha256"])
            row=dict(case=c["name"],arm=arm,joins=[region(fk,t-3,t+3) for t in (60,90)],center=region(fk,69,81),
                source_error_rad=source_error(x,plan,60,90),roi_native_ok=roi["absolute"]["ok"],global_native_ok=physical["absolute"]["ok"],source_condition_used=False,
                noise_sha256=g["sampler"]["noise_sha256"])
            records.append(row);bindings.append(dict(case=c["name"],arm=arm,native=ref(folder/"edited_native.npy"),fk=ref(folder/"edited_fk.npy"),original=c["native"]))
            if replay:
                other=Path(replay)/folder.name
                for f in ARRAYS+("edited_native.npy","edited_fk.npy"):assert np.array_equal(np.load(folder/f),np.load(other/f))
                assert read(folder/"roi_physical.json")==read(other/"roi_physical.json") and read(folder/"edited_physical.json")==read(other/"edited_physical.json")
            print("R9_B2_APPLICATION",c["name"],arm,row["roi_native_ok"],flush=True)
        assert len(noise)==1
    result=dict(records=records,new_neural_generations=8,neural_weight_updates=0,new_whole_song_generations=0,source_condition_used=False,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),parent=ref(parent/"bridge/protocol.json"),training=ref(Path(training)/"result.json"),bindings=bindings))
    if replay:assert result==read(Path(replay)/"result.json")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--parent",required=True);p.add_argument("--training",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(a.parent,a.training,a.out,a.replay_from)
