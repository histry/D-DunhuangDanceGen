"""Source-specific input interventions, geometry and amplitude; no semantic truth."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.research.lodge_adapter_r25.metrics import source_metrics,motion_metrics
from .common import read,fresh,commit,forbid_optimizers
from .reference import events

def timing(motion,reference):
    a,b=events(motion[64:160,7:]),events(reference[64:160,7:])
    pairs=[(x["frame"],y["frame"]) for x,y in zip(a,b) if x["frame"] is not None and y["frame"] is not None]
    return dict(generated=a,source=b,comparable_types=len(pairs),mae_frames=float(np.mean([abs(x-y) for x,y in pairs])) if pairs else None,scope="Kinematic candidate proxy; UNKNOWN retained; not semantic event truth")

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    samples=read(root/"samples/result.json");z=dict(np.load(root/"samples/arrays.npz"))
    oldroot=Path(cfg["r24"]);old=dict(np.load(checked(read(oldroot/"main/protocol.json")["arrays"])))
    hist=dict(np.load(Path(cfg["r25"])/"samples/arrays.npz"));pop=read(root/"data/result.json")
    data={r["case"]:dict(np.load(root/"data"/r["arrays"])) for r in pop["records"]};roles={r["case"]:r["role"] for r in pop["records"]}
    cue=old["cue_mask"][:,7:].all(-1).astype(bool);mask=np.zeros(256,bool);mask[64:160]=True;mask&=~cue;records=[]
    def measure(motion,points,refs,rpoints,metadata):
        errors=[]
        for ref,rp in zip(refs,rpoints):
            er=source_metrics(motion,ref,points,rp,mask)
            a=points[64:160]-points[64:160,:1];b=rp[64:160]-rp[64:160,:1]
            extent=lambda x:float(np.sqrt(np.mean(np.ptp(x[:,[16,17,18,19,20,21]],axis=0)**2)))
            source_extent=extent(b);er["source_relative_arm_range_ratio"]=extent(a)/source_extent if source_extent>1e-8 else None
            errors.append(er)
        records.append(dict(metadata,errors=errors,quality=motion_metrics(motion,points,cue),timing=[timing(motion,r) for r in refs],
            source_truth_scope="Paired observed recording" if metadata["dataset"]=="paired" else "Retimed reference proxy only; no new-music paired truth"))
    for row in samples["receipts"]:
        x=z[row["stem"]+"_139"];p=z[row["stem"]+"_22"]
        for i,item in enumerate(row.get("conditions",row.get("cases",[]))):
            if row["dataset"]=="change":
                music,event=item;measure(x[i],p[i],old["source139"],old["source22"],dict(dataset="change",arm=row["arm"],mode=row["mode"],music=music,event=event,seed=row["seed"]))
            else:
                d=data[item];measure(x[i],p[i],[d["raw139"]],[d["source22"]],dict(dataset="paired",arm=row["arm"],mode=row["mode"],case=item,role=roles[item],seed=row["seed"]))
    for arm,values in [("R24",old)]+[(f"R25_{a}",hist) for a in ("event","phase")]:
        for seed in cfg["eval_seeds"]:
            stem=str(seed) if arm=="R24" else f"change_{arm[4:]}_correct_{seed}"
            for i,(music,event) in enumerate([[m,e] for m in range(2) for e in (-1,0,1)]):
                measure(values[stem+"_139"][i],values[stem+"_22"][i],old["source139"],old["source22"],dict(dataset="change",arm=arm,mode="frozen",music=music,event=event,seed=seed))
    def get(arm,mode,seed,m,e):
        return next(r for r in records if r["dataset"]=="change" and (r["arm"],r["mode"],r["seed"],r["music"],r["event"])==(arm,mode,seed,m,e))
    comparisons=[]
    for arm in ("C1","C2"):
        for seed in cfg["eval_seeds"]:
            for m in range(2):
                tests=[]
                for e in range(2):
                    new=get(arm,"correct",seed,m,e);wrong=get(arm,"swapped",seed,m,e);base=get("C0","correct",seed,m,e);none=get("C0","correct",seed,m,-1)
                    er=new["errors"][e];wr=wrong["errors"][e];be=base["errors"][e];q=new["quality"];qn=none["quality"]
                    values={k+"_gain":er[k]<=(1-cfg["min_gain"])*be[k] for k in ("body22_rad","wrist_relative_shoulder_rmse_m","double_arm_wrist_relation_rmse_m")}
                    values.update(correct_input_body=er["body22_rad"]<wr["body22_rad"],correct_input_wrist=er["wrist_relative_shoulder_rmse_m"]<wr["wrist_relative_shoulder_rmse_m"],
                        activity=q["activity_mps"]>=cfg["min_activity_ratio"]*qn["activity_mps"],
                        global_jerk=q["jerk_p95_mps3"]<=cfg["max_quality_ratio"]*qn["jerk_p95_mps3"],
                        core_jerk=q["regions"]["core_interior_unprompted"]["p95"]<=cfg["max_quality_ratio"]*qn["regions"]["core_interior_unprompted"]["p95"])
                    tests.append(dict(event=e,tests=values,passed=all(values.values())))
                comparisons.append(dict(arm=arm,seed=seed,music=m,events=tests,passed=all(x["passed"] for x in tests)))
    result=dict(records=records,comparisons=comparisons,passed_groups={a:sum(r["passed"] for r in comparisons if r["arm"]==a) for a in ("C1","C2")},
        original_R25_gate=read("reports/r25_release/r25_summary.json")["evaluation"]["passed_groups"],
        optimizer_steps=0,C3_training_admitted=False,full_native_status="NOT_EVALUABLE",contact_truth="UNKNOWN",operational_qualified=False,whole_song_generations=0)
    assert len(records)==202
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(root/"samples/arrays.npz")))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R26_EVALUATION",result["passed_groups"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
