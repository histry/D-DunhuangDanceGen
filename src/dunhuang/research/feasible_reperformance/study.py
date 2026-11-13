"""Fixed eight sources, preserved R8 controls, no outcome-dependent selection."""
import argparse,copy,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,sha,write,pinned
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.primitive_reperformance.environment import worker_environment
from dunhuang.research.primitive_reperformance.metrics import measure,angles
from dunhuang.research.primitive_reperformance.study import ARRAYS
from .guide import LOWER

def trace_metrics(file):
    x=np.load(file);p=x["predicted_root"].astype(float);s=x["source_root"].astype(float)
    error=angles(p[:,:1].swapaxes(-1,-2)@p,np.broadcast_to(s[:1].swapaxes(-1,-2)@s,p.shape)).mean(-1)
    delta=p[0,0].T@p[-1,0]
    return dict(initial_prediction_to_final_prediction_rad=float(Rotation.from_matrix(delta).magnitude()),
        first_to_last_predicted_yaw_rad=float(np.angle(np.exp(1j*(np.arctan2(p[-1,0,0,2],p[-1,0,2,2])-np.arctan2(p[0,0,0,2],p[0,0,2,2]))))),
        relative_turn_error_by_step_rad=error.tolist(),source_relative_curve_truth="FITTED_ROTATION_ONLY",
        first_heading_difference_is_not_causal_proof=True)

def launch(q,folder,assets):
    folder.mkdir();q=dict(q,output=str(folder.resolve()));write(folder/"request.json",q)
    for python,mod in ((assets["edge_python"],"dunhuang.research.feasible_reperformance.worker"),(assets["native_python"],"dunhuang.physical")):
        with (folder/(mod.split(".")[-1]+".log")).open("w") as log:
            subprocess.run([python,"-m",mod,str(folder/"request.json")],check=True,stdout=log,stderr=subprocess.STDOUT,env=worker_environment())

def run(cfg,out,replay=None):
    out=fresh(out);root=Path(cfg["parent"]);seal=read("reports/r8_summary.json")
    for name in ("single/protocol.json","single/result.json","prepared/protocol.json","prepared/pairs.json"):
        if name in seal["evidence"] and sha(root/name)!=seal["evidence"][name]:raise ValueError("Frozen R8 seal differs")
    proto=read(root/"single/protocol.json");prior=read(root/"single/result.json")
    assets=read(checked(read(root/"prepared/protocol.json")["assets"]))
    sys.path.insert(0,str(pinned(assets["legacy_root"],assets["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    records=[];bindings=[];traces=[];count=0;exact=0
    for b in proto["bindings"]:
        r=next(r for r in prior["records"] if b["name"]==str(r["pair"])+"_"+str(r["music"])+"_"+r["arm"]+("_"+r["source_label"] if r["source_label"] else ""))
        if r["music"]!=cfg["music"] or r["arm"]!="S2":continue
        q0=read(checked(b["request"]));plan=read(checked(b["plan"]));original=Path(checked(b["native"])).parent
        teacher=(Path(replay)/str(r["pair"])/r["source_label"]/"T0") if replay else out/str(r["pair"])/r["source_label"]/"T0"
        if not replay:
            for arm in ("T0","S3_trace"):
                q=copy.deepcopy(q0)
                if arm=="S3_trace":
                    name=b["name"].replace("_S2_","_S3_");oldb=next(v for v in proto["bindings"] if v["name"]==name);oldfolder=Path(checked(oldb["native"])).parent;q=read(checked(oldb["request"]))
                else:oldfolder=original
                q.update(research_r9=True);folder=out/str(r["pair"])/r["source_label"]/arm;folder.parent.mkdir(parents=True,exist_ok=True)
                launch(q,folder,assets);count+=1
                for file in ARRAYS:
                    assert np.array_equal(np.load(folder/file),np.load(oldfolder/file)),("Trace changed frozen output",file);exact+=1
                assert read(folder/"physical.json")==read(oldfolder/"physical.json")
                traces.append(dict(pair=r["pair"],source=r["source_label"],arm=arm,metrics=trace_metrics(folder/"state_trace.npz"),trace=ref(folder/"state_trace.npz")))
        # T0 is a matched saved control. It is not counted as a newly trained model.
        rows=[("T0",original)]
        for arm,joints in cfg["groups"].items():
            q=copy.deepcopy(q0);q.update(research_r9=True,condition_group=arm,r9_relative_root=arm in cfg["relative_root"])
            q["reperformance_config"]["groups"][arm]=joints
            if arm=="T4":q["r9_lower_teacher"]=ref(teacher/"state_trace.npz")
            folder=out/str(r["pair"])/r["source_label"]/arm;folder.parent.mkdir(parents=True,exist_ok=True)
            launch(q,folder,assets);count+=1;rows.append((arm,folder))
            reference=(Path(replay)/str(r["pair"])/r["source_label"]/arm) if replay else None
            if reference:
                for file in ARRAYS:
                    assert np.array_equal(np.load(folder/file),np.load(reference/file));exact+=1
                assert read(folder/"physical.json")==read(reference/"physical.json")
            if arm=="T4":
                channels=[7+6*j+k for j in LOWER for k in range(6)]
                assert np.array_equal(np.load(folder/"native.npy")[:,channels],np.load(original/"native.npy")[:,channels])
        for arm,folder in rows:
            native=np.load(folder/"native.npy");p=read(folder/"physical.json");g=read(folder/"generation.json")
            assert g["weights_before"]==g["weights_after"] and g["adapter"] is None
            assert g["sampler"]["noise_sha256"]==r["noise_sha256"]
            row=dict(pair=r["pair"],source=r["source_label"],cohort=r["cohort"],arm=arm,source_metrics=measure(native,plan,proto["config"]["guide"]),
                native_ok=p["absolute"]["ok"],native_reasons=p["absolute"]["reasons"],audit=p["audit"],model_digest=g["weights_before"],noise_sha256=g["sampler"]["noise_sha256"])
            records.append(row);bindings.append(dict(pair=r["pair"],source=r["source_label"],arm=arm,native=ref(folder/"native.npy"),physical=ref(folder/"physical.json"),plan=b["plan"]))
            print("R9_SOURCE",r["pair"],r["source_label"],arm,row["native_ok"],flush=True)
    assert len(records)==48 and count==(40 if replay else 56)
    result=dict(records=records,new_neural_generations=count,traced_frozen_arrays_exact=exact,neural_weight_updates=0,operational_qualified=False,new_whole_song_generations=0)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,parent=ref(root/"single/protocol.json"),bindings=bindings,traces=traces))
    if replay:
        expected=read(Path(replay)/"result.json")
        assert records==expected["records"]
        write(out/"replay.json",dict(records_equal=True,exact_arrays=exact,new_neural_generations=count))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
