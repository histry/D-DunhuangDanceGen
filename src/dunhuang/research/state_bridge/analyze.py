"""Shared-state physical readouts and pre-registered training gate, no optimizer."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.contracts import MotionAdapter,matrices
from dunhuang.research.condition_dependency.codec import encode,restore
from dunhuang.research.source_capacity.common import read,commit,fresh
from .common import cases,native_environment
from .metrics import pair_quality,direction,rotvec,FEET

def mean(rows,key):return float(np.mean([r[key] for r in rows]))
def qualification(rows,cfg,oracle_passed):
    details={};passed=bool(oracle_passed)
    for arm,side in (("A3","right"),("A4","left")):
        a=[r for r in rows if r["arm"]==arm and r["case"].startswith("val_")]
        assert len(a)==cfg["gate"]["val_cases"]
        conditions=dict(mean_reconstruction_better=mean(a,"correct_mse")<mean(a,"mse"),
            paired_wins=sum(r["correct_mse"]<r["mse"] for r in a)>=cfg["gate"]["paired_wins_minimum"])
        gains={}
        for metric in ("root","foot"):
            values=[r[metric+"_gain"] for r in a if r[metric+"_target_norm"]>=cfg["gate"]["nonzero_target_delta_m"] and r[metric+"_gain"] is not None]
            gains[metric]=float(np.median(values)) if values else None
            conditions[metric+"_directional_response"]=bool(values and gains[metric]>=cfg["gate"]["minimum_directional_gain"])
        details[side]=dict(conditions=conditions,paired_wins=sum(r["correct_mse"]<r["mse"] for r in a),cases=len(a),
            mean_correct_mse=mean(a,"correct_mse"),mean_wrong_mse=mean(a,"mse"),median_directional_gains=gains)
        passed=passed and all(conditions.values())
    return dict(training_admitted=bool(passed),details=details,config=cfg["gate"],optimizer_steps=0,
        operational_qualified=False,root_joint_capacity_admitted=False,whole_song_admitted=False,
        action="REGISTER_P2_SEPARATELY" if passed else "STOP_NEW_TRAINING_ROOT_REPAIR_AND_WHOLE_SONG")

def run(cfg,audit,traces,out,replay=None):
    out=fresh(out);items=cases(cfg);native_environment(items[0]["request"])
    from motion.geometry import fk_24_np
    p=read(Path(traces)/"protocol.json");records=[];aggregates=[]
    for item,binding in zip(items,p["bindings"]):
        assert binding["case"]==item["name"];q=item["request"];old=item["original"];target=old[50:100];tfk=fk_24_np(target)
        encoded,o=encode(old[:150]);s=np.load(checked(q["scaler"]));target_norm=encoded[50:100]*s["scale"]+s["bias"]
        wrong=restore(MotionAdapter.to_native(np.load(checked(item["wrong_context"]))["values"][0]),o);wfk=fk_24_np(wrong)
        data=np.load(checked(binding["trace"]));steps=read(checked(binding["trace_info"]))["steps"]
        case_rows=[]
        for step,info in enumerate(steps):
            assert info["a0_features_exact"] and info["rng_unchanged"]
            base=None;base_fk=None
            for a,arm in enumerate(cfg["arms"]):
                z=data["prediction"][step,a];native=restore(MotionAdapter.to_native(((z-s["bias"])/s["scale"]).astype(np.float32)),o)
                hybrid=target.copy();hybrid[10:40]=native[10:40];fk=fk_24_np(hybrid)
                mse=float(np.mean((z[10:40,4:]-target_norm[10:40,4:])**2))
                row=dict(case=item["name"],arm=arm,step=step,t=info["t"],mse=mse,**pair_quality(hybrid,target,fk,tfk))
                if arm=="A0":
                    base=hybrid.copy();base_fk=fk.copy();base_mse=mse
                row["correct_mse"]=base_mse
                row["root_response_rms_m"]=float(np.sqrt(np.mean((hybrid[10:40,4:7]-base[10:40,4:7])**2)))
                row["pose_response_rad"]=float(np.linalg.norm(rotvec(matrices(base[10:40,7:].reshape(-1,24,6)).swapaxes(-1,-2)@matrices(hybrid[10:40,7:].reshape(-1,24,6))),axis=-1).mean())
                if arm in ("A3","A4"):
                    endpoint,known=(39,90) if arm=="A3" else (10,59)
                    root_response=hybrid[endpoint,4:7]-base[endpoint,4:7]
                    root_target=wrong[known,4:7]-old[known,4:7]
                    foot_response=fk[endpoint,FEET]-base_fk[endpoint,FEET]
                    foot_target=wfk[known,FEET]-fk_24_np(old[known:known+1])[0,FEET]
                    root_direction=direction(root_response,root_target);foot_direction=direction(foot_response,foot_target)
                    cr=matrices(old[known,7:13]);wr=matrices(wrong[known,7:13])
                    br=matrices(base[endpoint,7:13]);nr=matrices(hybrid[endpoint,7:13])
                    rotation_target=rotvec(wr@cr.T)
                    row.update(root_direction=root_direction,foot_direction=foot_direction,rotation_direction=direction(rotvec(nr@br.T),rotation_target),
                        rotation_target_chart_safe=bool(np.linalg.norm(rotation_target)<np.pi-.05))
                case_rows.append(row);records.append(row)
        for arm in cfg["arms"]:
            rows=[x for x in case_rows if x["arm"]==arm];r=dict(case=item["name"],arm=arm)
            for key in ("mse","correct_mse","root_rmse_m","pose_error_rad","feet_rmse_m","boundary_root_rmse_m","boundary_feet_rmse_m","root_response_rms_m","pose_response_rad"):
                r[key]=mean(rows,key)
            if arm in ("A3","A4"):
                for metric in ("root","foot"):
                    r[metric+"_gain"]=float(np.mean([v[metric+"_direction"]["projection_gain"] for v in rows])) if all(v[metric+"_direction"]["projection_gain"] is not None for v in rows) else None
                    r[metric+"_target_norm"]=float(rows[0][metric+"_direction"]["target_norm"])
            aggregates.append(r)
    gate=qualification(aggregates,cfg,read(Path(audit)/"result.json")["oracle_passed"])
    result=dict(records=records,case_means=aggregates,qualification=gate,optimizer_steps=0,new_neural_generations=0,new_whole_song_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,traces=ref(Path(traces)/"protocol.json"),audit=ref(Path(audit)/"result.json"),
        normalized_prediction_readout="decode then graft only unknown [60,90); counterfactual is a clean prediction on the A0 state, not a new trajectory"))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R11_GATE",gate,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--audit",required=True);p.add_argument("--traces",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.audit,a.traces,a.out,a.replay_from)
