"""Native/readout process: no optimizer, no model selection on exposed Val."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.condition_dependency.codec import origin
from .common import assets_and_population
from .metrics import physical,qmetrics,joins

def run(cfg,source,out,phase,training=None,replay=None):
    out=fresh(out);assets,_,tiny,monitor,_=assets_and_population(cfg);native_environment(assets)
    scaler=np.load(checked(assets["scaler"]));data=Data(scaler);source=Path(source);records=[]
    if phase=="teacher":
        info=read(source/"result.json");good=[]
        for binding in info["bindings"]:
            old=data.native(binding["row"]);z=np.load(checked(binding["rollout"]));case=binding["case"];rows=[]
            for arm,sample in (("B0",z["baseline"][0]),("teacher",z["sample"][0])):
                edited,fk,audit,score=physical(sample,old,scaler);folder=out/(str(case)+"_"+arm);folder.mkdir()
                np.save(folder/"native.npy",edited);np.save(folder/"fk.npy",fk);write(folder/"physical.json",audit)
                row=dict(case=case,arm=arm,**score);records.append(row);rows.append(row)
                if replay:
                    other=Path(replay)/folder.name;assert np.array_equal(edited,np.load(other/"native.npy")) and np.array_equal(fk,np.load(other/"fk.npy"))
                    assert audit==read(other/"physical.json")
            base,teacher=rows
            good.append(teacher["native_ok"] and teacher["pose_error_rad"]<=base["pose_error_rad"]+.01 and
                all(a["foot_step_max_m"]<=b["foot_step_max_m"]+.01 for a,b in zip(teacher["joins"],base["joins"])))
        result=dict(records=records,teacher_usable=all(good),teacher_cases_passed=sum(good),teacher_distillation_admitted=all(good),
            direction_uses_unknown_truth=False,tiny_new_architecture_training_independent=True,optimizer_steps=0,operational_qualified=False)
    else:
        bindings=read(source/"result.json")["bindings"];pop=tiny if phase=="tiny" else monitor
        for binding in bindings:
            old=data.native(binding["row"]);z=np.load(checked(binding["trace"]));folder=out/(str(binding["case"])+"_"+binding["arm"]+"_"+binding["driver"]);folder.mkdir()
            edited,fk,audit,score=physical(z["sample"][0],old,scaler)
            target=MotionAdapter.to_edge(old,origin(old))*scaler["scale"]+scaler["bias"]
            raw=qmetrics(z["raw"][-1],target,scaler)
            row=dict(case=binding["case"],arm=binding["arm"],driver=binding["driver"],**score,raw_final=raw)
            if binding["driver"]=="correct":
                controls=[]
                names=("correct","removed","removed_left","removed_right","wrong_left","wrong_right")
                for j,t in enumerate(z["t"]):
                    for k,name in enumerate(names):controls.append(dict(t=int(t),control=name,**qmetrics(z["prediction"][j,k],target,scaler)))
                row["shared_state_controls"]=controls
            records.append(row);np.save(folder/"native.npy",edited);np.save(folder/"fk.npy",fk);write(folder/"physical.json",audit)
            if replay:
                other=Path(replay)/folder.name;assert np.array_equal(edited,np.load(other/"native.npy")) and np.array_equal(fk,np.load(other/"fk.npy"))
                assert audit==read(other/"physical.json")
        qrows=[];training=Path(training);initial=np.load(training/"initial.npz");pred=np.load(training/"q_eval.npz")
        for i,b in enumerate(tiny):
            old=data.native(b["row"]);target=MotionAdapter.to_edge(old,origin(old))*scaler["scale"]+scaler["bias"]
            for j,t in enumerate(cfg["timesteps"]):
                qrows.append(dict(case=i,t=t,control="initial",**qmetrics(initial[str(j)][i],target,scaler)))
                for control in ("correct","removed","removed_left","removed_right","wrong_left","wrong_right"):
                    qrows.append(dict(case=i,t=t,control=control,**qmetrics(pred[str(j)+"_"+control][i],target,scaler)))
        cases=[]
        for i in range(4):
            def mean(control,key):return float(np.mean([r[key] for r in qrows if r["case"]==i and r["control"]==control]))
            ratios={key:mean("correct",key)/max(mean("initial",key),1e-12) for key in ("known_root_rmse_m","known_pose_rad","reconstruction")}
            def joint(c):return (mean(c,"known_root_rmse_m")/.1)**2+(mean(c,"known_pose_rad")/.3)**2
            cases.append(dict(case=i,ratios=ratios,wrong_left_win=joint("correct")<joint("wrong_left"),wrong_right_win=joint("correct")<joint("wrong_right")))
        g=cfg["tiny_gate"];fits=sum(all(r<=g["maximum_ratio"] for r in c["ratios"].values()) for c in cases)
        wins={side:sum(c[side+"_win"] for c in cases) for side in ("wrong_left","wrong_right")}
        admitted=fits>=g["minimum_cases"] and min(wins.values())>=g["minimum_control_wins"]
        result=dict(records=records,q_records=qrows,tiny_cases=cases,tiny_fit_passed_cases=fits,context_control_wins=wins,
            tiny_training_admitted=admitted,original_native_thresholds_unchanged=True,
            native_passes=sum(r["native_ok"] for r in records if r["arm"]=="MCD" and r["driver"]=="correct"),
            whole_song_admitted=False,optimizer_steps=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,source=ref(source/"result.json"),phase=phase))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R13_ANALYZE",phase,{k:v for k,v in result.items() if k not in ("records","q_records")},flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--source",required=True);p.add_argument("--out",required=True);p.add_argument("--phase",choices=["teacher","tiny","main"],required=True);p.add_argument("--training");p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.source,a.out,a.phase,a.training,a.replay_from)
