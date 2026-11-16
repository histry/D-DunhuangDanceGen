"""Zero-training support qualification on all16 Train,4 heldout,8 FineDance and2 ChangE."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import sources,runtime
from .gauge import solve
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,forbid_optimizers,same_arrays,matrices

def derivatives(points,target,valid):
    result={}
    for k in (1,2,3):
        mask=np.array([valid[i:i+k+1].all() for i in range(len(valid)-k)])
        delta=(np.diff(points,n=k,axis=0)-np.diff(target,n=k,axis=0))*30**k
        values=delta[mask]
        result[str(k)]=dict(templates=int(mask.sum()),rmse=float(np.sqrt(np.mean(values**2))) if mask.any() else None)
    return result

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();rows,ledger,old=sources(cfg);d,n,s,f,e,info,a=runtime(cfg);a.close()
    scale=n.scaler.scale_.detach().cpu().numpy();offset=n.scaler.min_.detach().cpu().numpy()
    assert (scale>0).all() and n.scaler.clip and list(n.scaler.feature_range)==[-1,1]
    margin=cfg["normalized_box_margin"];lower=(-1+margin-offset[7:])/scale[7:];upper=(1-margin-offset[7:])/scale[7:]
    rawlower=(-1-offset[7:])/scale[7:];rawupper=(1-offset[7:])/scale[7:]
    np.savez_compressed(out/"bounds.npz",scale=scale,offset=offset,lower=lower,upper=upper,actual_lower=rawlower,actual_upper=rawupper)
    records=[]
    for row in rows:
        source=row["data"];raw=source["model_raw"].copy();T=len(raw);R=matrices(raw[:,7:].reshape(T,22,6),row=True)
        result=solve(R,lower,upper,cfg["minimum_gauge_norm"],cfg["sequence_iterations"],
                     cfg["coefficient_first_difference_weight"],cfg["coefficient_second_difference_weight"])
        feasible=result["feasible"];complete=feasible.all(1)
        weak=None
        if not feasible.all():weak=solve(R,rawlower,rawupper,1e-6,iterations=0)
        encoded=np.full_like(raw,np.nan);encoded[complete]=raw[complete];encoded[complete,7:]=result["code"][complete].reshape(-1,132)
        points=np.full((T,22,3),np.nan,np.float32);back=np.full_like(raw,np.nan);normalized=np.full_like(raw,np.nan)
        numerical=np.full_like(points,np.nan)
        # An affine-inverse diagnostic supplies the roundoff floor, never the actual sampler.
        with torch.no_grad():
            original=torch.tensor(raw[None],device="cuda")
            affine=original*n.scaler.scale_.to(original)+n.scaler.min_.to(original)
            algebra=(affine-n.scaler.min_.to(original))/n.scaler.scale_.to(original)
            numerical=f(algebra,s)[0,:,:22].cpu().numpy()
            if complete.any():
                value=torch.tensor(encoded[complete][None],device="cuda");z=n.normalize(value.clone());b=n.unnormalize(z.clone())
                back[complete]=b[0].cpu().numpy();normalized[complete]=z[0].cpu().numpy()
                points[complete]=f(b,s)[0,:,:22].cpu().numpy()
        target=source["model_points"]
        angles=None;position=None
        if complete.any():
            decoded=matrices(back[complete,7:].reshape(-1,22,6),row=True)
            diff=decoded.swapaxes(-1,-2)@R[complete]
            from scipy.spatial.transform import Rotation
            angles=float(Rotation.from_matrix(diff.reshape(-1,3,3)).magnitude().max())
            position=float(np.linalg.norm((points-target)[complete],axis=-1).max())
        actual_affine=raw*scale+offset
        other_support=bool((abs(actual_affine[:,:7])<=1+1e-6).all()) if row["Root_observed"] else None
        derivative=derivatives(points,target,complete)
        oracle=derivatives(numerical,target,np.ones(T,bool))
        tolerances={str(k):max(cfg["numerical_derivative_absolute_floors"][k-1],cfg["numerical_baseline_multiplier"]*oracle[str(k)]["rmse"]) for k in (1,2,3)}
        temporal=all(derivative[str(k)]["rmse"] is not None and derivative[str(k)]["rmse"]<=tolerances[str(k)] for k in (1,2,3))
        qualified=bool(feasible.all() and (other_support is not False) and angles is not None and angles<=cfg["angle_max_rad"] and position<=cfg["position_max_m"] and temporal)
        failure=np.argwhere(~feasible)
        path=out/(row["uid"]+".npz")
        np.savez_compressed(path,source_raw=raw,source22=target,encoded_raw=encoded,normalized_target=normalized,decoded22=points,numerical_oracle22=numerical,
            code=result["code"],feasible=feasible,coefficients=result["coefficients"],initial_coefficients=result["initial_coefficients"],
            alpha_lower=result["alpha_lower"],alpha_upper=result["alpha_upper"],failure_frame_joint=failure)
        record=dict(uid=row["uid"],domain=row["domain"],role=row["role"],case=row["case"],frames=T,Root_observed=row["Root_observed"],
            stable_joint_frames=int(feasible.sum()),total_joint_frames=T*22,full_frames=int(complete.sum()),
            failed_frame_joints=failure.tolist(),failed_alpha=int((~result["alpha_feasible"]).sum()),
            failed_beta_gamma=int((~result["beta_gamma_feasible"]).sum()),
            weak_actual_box_joint_frames=None if weak is None else int(weak["feasible"].sum()),
            minimum_alpha=float(np.nanmin(result["coefficients"][:,:,0])) if feasible.any() else None,
            minimum_orthogonal_beta_length=float(np.nanmin(result["coefficients"][:,:,2])) if feasible.any() else None,
            angle_max_rad=angles,position_max_m=position,derivatives=derivative,numerical_baseline=oracle,derivative_tolerances=tolerances,
            Root_C4_support=other_support,geometry_qualified=qualified,arrays=path.name,sha256=sha(path),
            scope="OBSERVED_COMMON22_GEOMETRY; NAN_PRESERVES_FAILURES; UNKNOWN_ROOT_CONTACT_NOT_TRAINING_TRUTH")
        records.append(record)
        if replay:same_arrays(path,Path(replay)/path.name)
        print("R31_GAUGE",row["uid"],int(feasible.sum()),"/",T*22,"QUALIFIED",qualified,flush=True)
    admitted=all(r["geometry_qualified"] for r in records)
    result=dict(records=records,source_records=len(records),train_recordings=16,train_music_groups=8,
        gauge_training_admitted=admitted,all_observed_geometry_qualified=admitted,
        required_branch="GAUGE_TARGET_ONLY" if admitted else "NEW_ACTION_CONTRACT_QUALIFICATION",
        failure_scope="REGISTERED_MINIMUM_NORMS_AND_INTERIOR_MARGIN; NOT_PROOF_NO_POSITIVE_6D_GAUGE_EXISTS",
        normalizer_unchanged=True,neural_optimizer_steps=0,operational_qualified=False,full_native_status="NOT_EVALUABLE",whole_song_generations=0)
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"bounds.npz",Path(replay)/"bounds.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,ledger_cases=old["recordings"]))
    print("R31_GAUGE_ADMISSION",admitted,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
