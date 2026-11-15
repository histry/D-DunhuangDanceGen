"""Six frozen feasible windows; no fresh search, no motion prior or generator training."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.global_clock_r17.evaluate import score
from dunhuang.research.task_denoising_r18.evaluate import dynamics
from dunhuang.research.branch_r22.data import domains
from .common import inputs,finish,arrays,read,fresh,forbid_optimizers
from .representation import encode,decode

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();root,pc,pop=inputs(cfg)
    native_environment(read(root/"data/protocol.json")["assets"])
    audit=read(root/"data/audit.json");selected=audit["mechanism_selected"];assert len(selected)==6
    from motion.geometry import fk_24_np
    records=[];bindings=[];fits=0;settings=cfg["representation"]
    for case,item in enumerate(selected):
        source=item["source"];full=np.load(checked(source["source_native"]));a=source["start_frame"];native=full[a:a+225].copy()
        gt,source_report=audit_native(native);assert source_report["absolute"]["ok"]
        variants=[("source",None),("full_decode",None)]+[("boundary_"+str(l)+"_"+str(r),(l,r)) for l,r in settings["contexts"]]
        for mode,context in variants:
            stem=f"{case}_{mode}";folder=out/stem;folder.mkdir()
            if mode=="source":x=native.copy();dimensions=None;state=None
            else:
                state=encode(native,settings["coefficients"],context);x=decode(state);fits+=1
                arrays(out,stem+"_latent.npz",state,replay)
                dimensions=dict(latent_coefficients=int(state["coefficients"].size),observed_context_values=int(state["boundary"].size),
                    target_values=225*147,rank=int(state["rank"]),condition_number=float(state["condition_number"]))
                if context:
                    left,right=context;known=np.r_[0:left,225-right:225]
                    assert np.max(abs(x[known,4:]-native[known,4:]))<=1e-6
            fk,physical=audit_native(x)
            arrays(out,stem+".npz",dict(native=x,fk=fk),replay);write(folder/"physical.json",physical)
            if replay:assert physical==read(Path(replay)/stem/"physical.json")
            d=dynamics(x,fk);ds=dynamics(native,gt)
            value=score(x,native,fk,gt,np.arange(225))
            activity=value["activity_mps"]/max(float(np.linalg.norm(np.diff(gt,axis=0)*30,axis=-1).mean()),1e-12)
            movement=float(np.mean(d["joint_range_p95_rad"][1:22]))/max(float(np.mean(ds["joint_range_p95_rad"][1:22])),1e-12)
            preserved=value["pose_error_rad"]<=settings["max_pose_rad"] and value["root_rmse_m"]<=settings["max_root_m"] and settings["min_activity_ratio"]<=activity<=settings["max_activity_ratio"] and settings["min_range_ratio"]<=movement<=settings["max_range_ratio"]
            records.append(dict(case=case,mode=mode,role=item["role"],source_group=item["source_group"],recording_uid=item["recording_uid"],
                **value,native_ok=physical["absolute"]["ok"],native_reasons=physical["absolute"]["reasons"],native_audit=physical["absolute"]["audit"],
                source_activity_ratio=activity,source_range_ratio=movement,preservation_pass=bool(preserved),full_target_fit=mode!="source",
                context=list(context) if context else None,domains={str(pair):domains(fk,*pair) for pair in settings["contexts"]},
                dimensions=dimensions,array=stem+".npz",physical=stem+"/physical.json",contacts_observed=False,semantic_truth="UNKNOWN"))
            print("R23_REPRESENT",case,mode,physical["absolute"]["ok"],flush=True)
        bindings.append(dict(case=case,role=item["role"],source=source,original_source_index=item["recording_uid"],source_native_report_passed=True,
            duplicated_historical_case24=source["recording_uid"]==pop["records"][24]["left"]["recording_uid"] and a==pop["records"][24]["left"]["start_frame"]))
    train=[r for r in records if r["role"]=="train" and r["mode"]!="source"]
    gate=dict(five_train_all_representation_outputs_preserved=all(r["preservation_pass"] and r["native_ok"] for r in train),
        train_windows=5,train_music_groups=audit["mechanism_train_groups"],dev_exposed_windows=1,independent_generalization_proven=False,
        conditional_training_admitted=False,reason="Finite full-target representation upper bound only; population coverage and conditional-generation evidence insufficient")
    finish(out,dict(records=records,bindings=bindings,latent_least_squares_fits=fits,trainable_shared_decoder_parameters=0,
        shared_decoder="Fixed cubic B-spline basis; 147 root/6D channels, SO(3) via checked Gram-Schmidt; 24 rotations",
        boundary_contract="Observed context cubic splines + quintic C2 lift + endpoint-null cubic residual basis; no post-decoding hard overwrite",
        human_reviewed=False,operational_qualified=False,optimizer_steps=0,conditional_generation=False,gate=gate),cfg,replay)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
