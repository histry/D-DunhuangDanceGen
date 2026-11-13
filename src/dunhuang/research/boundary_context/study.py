"""Eight fixed V3 validation in-betweening cases, full generation and exact replay."""
import argparse,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import MotionAdapter,matrices
from dunhuang.io import checked,write,pinned
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.primitive_reperformance.environment import worker_environment
from dunhuang.research.primitive_reperformance.study import ARRAYS
from dunhuang.research.bridge_inpainting.study import audit_native,region
from .common import Data,fixed_rows

def run(cfg,preflight,training,out,replay=None):
    out=fresh(out);protocol=read(Path(preflight)/"protocol.json");assets=read(checked(protocol["assets"]));trained=read(Path(training)/"result.json")
    rows=fixed_rows(read(Path(preflight)/"rows.json"),"val",cfg["generation_windows"],cfg["seed"])
    assert len(rows)==cfg["generation_windows"];data=Data(np.load(checked(assets["scaler"])))
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    records=[];bindings=[]
    for i,row in enumerate(rows):
        target=data.native(row);origin=target[0,4:7].copy();origin[1]=0;target[:,4:7]-=origin
        a,b=60,90;values=MotionAdapter.to_edge(target)[None];mask=np.zeros((1,150,151),bool);mask[:,52:60,4:]=True;mask[:,90:98,4:]=True
        context=out/(str(i)+"_context.npz");np.savez_compressed(context,values=values,mask=mask);feature=out/(str(i)+"_features.npy");np.save(feature,np.load(checked(row["jukebox"]))[None])
        truth_fk,truth_physical=audit_native(target);noise=set()
        for arm in ("zero","learned"):
            folder=out/(str(i)+"_"+arm);folder.mkdir();q=dict(assets,features=ref(feature),audio=row["audio"],audio_start_frame=row["start_frame"],
                frames=150,windows=1,mode="anchored",anchors=[],context=ref(context),research_b2=True,seed=cfg["seed"]+100+i,overlap_policy="mean",
                output=str(folder.resolve()),b2_adapter=trained["checkpoint"] if arm=="learned" else None)
            write(folder/"request.json",q)
            for python,module in ((assets["edge_python"],"dunhuang.research.boundary_context.worker"),(assets["native_python"],"dunhuang.physical")):
                with (folder/(module.split(".")[-1]+".log")).open("w") as log:subprocess.run([python,"-m",module,str(folder/"request.json")],check=True,stdout=log,stderr=subprocess.STDOUT,env=worker_environment())
            raw=np.load(folder/"native.npy");x=target.copy();x[a:b]=raw[a:b]
            assert np.array_equal(x[:a],target[:a]) and np.array_equal(x[b:],target[b:])
            fk,physical=audit_native(x);np.save(folder/"edited_native.npy",x);np.save(folder/"edited_fk.npy",fk);write(folder/"edited_physical.json",physical)
            R=matrices(x[a:b,7:].reshape(-1,24,6));G=matrices(target[a:b,7:].reshape(-1,24,6))
            geo=float(Rotation.from_matrix((R.swapaxes(-1,-2)@G).reshape(-1,3,3)).magnitude().mean())
            generation=read(folder/"generation.json");noise.add(generation["sampler"]["noise_sha256"])
            record=dict(case=i,arm=arm,source_group=row["source_group"],rotation_error_rad=geo,root_rmse_m=float(np.sqrt(np.mean((x[a:b,4:7]-target[a:b,4:7])**2))),
                joins=[region(fk,t-3,t+3) for t in (a,b)],true_joins=[region(truth_fk,t-3,t+3) for t in (a,b)],
                native_ok=physical["absolute"]["ok"],native_reasons=physical["absolute"]["reasons"],target_native_ok=truth_physical["absolute"]["ok"],
                native_audit=physical["audit"],model_digest=generation["weights_before"],noise_sha256=generation["sampler"]["noise_sha256"])
            assert generation["weights_before"]==generation["weights_after"];records.append(record);bindings.append(dict(case=i,arm=arm,native=ref(folder/"edited_native.npy"),fk=ref(folder/"edited_fk.npy"),target=row))
            if replay:
                prior=Path(replay)/folder.name
                for f in ARRAYS+("edited_native.npy","edited_fk.npy"):assert np.array_equal(np.load(folder/f),np.load(prior/f))
                assert read(folder/"edited_physical.json")==read(prior/"edited_physical.json")
            print("R9_B2_GENERATION",i,arm,record["native_ok"],flush=True)
        assert len(noise)==1
    result=dict(records=records,new_neural_generations=16,neural_weight_updates=0,new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,training=ref(Path(training)/"result.json"),preflight=ref(Path(preflight)/"protocol.json"),bindings=bindings))
    if replay:assert result==read(Path(replay)/"result.json")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--preflight",required=True);p.add_argument("--training",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.preflight,a.training,a.out,a.replay_from)
