"""Full original Native and protected-domain readouts for all 64 CFG conditions."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.masked_denoiser.metrics import decode
from dunhuang.research.global_clock_r17.evaluate import score
from dunhuang.research.branch_r22.data import domains
from dunhuang.research.task_denoising_r18.evaluate import dynamics
from .common import inputs,finish,arrays,read,fresh,forbid_optimizers

def qualify(rows,cfg):
    q=cfg["cfg_gate"];cases=sorted({r["case"] for r in rows if r["role"]=="dev_exposed"});results=[]
    for arm in cfg["arms"]:
        for case in cases:
            layouts=[]
            for variant in cfg["eval_variants"]:
                a=next(r for r in rows if (r["arm"],r["case"],r["variant"],r["cfg"])==(arm,case,variant,1))
                b=next(r for r in rows if (r["arm"],r["case"],r["variant"],r["cfg"])==(arm,case,variant,2))
                tests=dict(pose=a["pose_error_rad"]<=q["max_pose_ratio"]*b["pose_error_rad"],root=a["root_rmse_m"]<=q["max_root_ratio"]*b["root_rmse_m"],
                    source_activity=a["source_activity_ratio"]>=q["min_activity_ratio"],source_range=a["source_range_ratio"]>=q["min_range_ratio"],
                    internal=a["domains"]["editable"]["p95"]<=q["max_internal_ratio"]*b["domains"]["editable"]["p95"],
                    boundary=a["domains"]["crossing"]["max"]<=(1-q["min_gain"])*b["domains"]["crossing"]["max"],
                    foot_skate=a["native_audit"]["foot_skate_mps_p95"]<=(1-q["min_gain"])*b["native_audit"]["foot_skate_mps_p95"],
                    drift=a["native_audit"]["foot_support_drift_m_p95"]<=q["max_drift_ratio"]*b["native_audit"]["foot_support_drift_m_p95"],
                    full_native_not_regressed=not b["native_ok"] or a["native_ok"])
                layouts.append(dict(variant=variant,tests=tests,passed=all(tests.values())))
            results.append(dict(arm=arm,case=case,layouts=layouts,passed=all(r["passed"] for r in layouts)))
    return dict(records=results,passed_dev_sources={arm:sum(r["passed"] for r in results if r["arm"]==arm) for arm in cfg["arms"]},
        required_dev_sources=q["dev_sources_required"],APG_not_run=True,no_guidance_search=True,default_not_promoted=True)

def run(cfg,study,out,replay=None):
    out=fresh(out);forbid_optimizers();root,pc,pop=inputs(cfg);pre=read(root/"data/protocol.json");native_environment(pre["assets"])
    from motion.geometry import fk_24_np
    scaler=np.load(checked(pre["assets"]["scaler"]));records=[];old=read(root/"native/result.json")["records"]
    bindings=read(Path(study)/"samples/result.json")["bindings"];by={r["case"]:r for r in pop["records"]}
    for b in bindings:
        row=by[b["case"]];z=np.load(checked(row["arrays"]));target=z["native"]
        tr=np.load(checked(b["trace"]) if b["reused_R22"] else Path(study)/"samples"/b["trace_file"])
        x=decode(tr["sample"],scaler,target);fk,physical=audit_native(x);gt=fk_24_np(target)
        name=f'{b["arm"]}_{b["case"]}_{b["variant"]}_cfg{b["cfg"]}';arrays(out,name+".npz",dict(native=x,fk=fk),replay)
        write(out/(name+"_physical.json"),physical)
        if replay:assert physical==read(Path(replay)/(name+"_physical.json"))
        r=score(x,target,fk,gt,np.arange(225));d=dynamics(x,fk);ds=dynamics(target,gt)
        r.update(arm=b["arm"],case=b["case"],variant=b["variant"],cfg=b["cfg"],role=b["role"],source_group=b["source_group"],
            native_ok=physical["absolute"]["ok"],native_reasons=physical["absolute"]["reasons"],native_audit=physical["absolute"]["audit"],
            source_activity_ratio=r["activity_mps"]/max(float(np.linalg.norm(np.diff(gt,axis=0)*30,axis=-1).mean()),1e-12),
            source_range_ratio=float(np.mean(d["joint_range_p95_rad"][1:22]))/max(float(np.mean(ds["joint_range_p95_rad"][1:22])),1e-12),
            domains=domains(fk,*pc["contexts"][b["variant"]]),array=name+".npz",reused_R22=b["reused_R22"])
        if b["cfg"]==2:
            previous=next(x for x in old if (x["arm"],x["case"],x["variant"],x["control"])==(b["arm"],b["case"],b["variant"],"correct"))
            for key in ("pose_error_rad","root_rmse_m","jerk_max_mps3","activity_mps","native_ok","native_reasons","native_audit"):assert r[key]==previous[key],key
        records.append(r)
    finish(out,dict(records=records,native_passes=sum(r["native_ok"] for r in records),native_count=64,
        comparisons=qualify(records,cfg),optimizer_steps=0,operational_qualified=False,whole_song_generations=0),cfg,replay)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--study",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.study,a.out,a.replay_from)
