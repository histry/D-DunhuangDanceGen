"""Actual new-contract geometry and old-interface compatibility, before any optimizer."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import sources
from .interface import runtime,CanonicalNormalizer
from .audit import derivatives
from dunhuang.research.time_r29.common import packet
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,forbid_optimizers,same_arrays,restore,digest
from dunhuang.research.state_r27.trace import sample

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    audit=read(root/"audit/result.json");assert not audit["gauge_training_admitted"]
    data,ledger,original=sources(cfg)
    d,legacy,s,f,e,info,bundle=runtime(cfg,False);canonical=CanonicalNormalizer(legacy)
    records=[];arrays={}
    bounds=np.load(root/"audit/bounds.npz");lower=bounds["actual_lower"].reshape(22,6);upper=bounds["actual_upper"].reshape(22,6)
    for row in data:
        raw=row["data"]["model_raw"];target=row["data"]["model_points"];T=len(raw)
        with torch.no_grad():
            x=torch.tensor(raw[None],device="cuda")
            z=canonical.normalize(x);back=canonical.unnormalize(z.clone());p=f(back,s)[0,:,:22].cpu().numpy()
            legacy_back=legacy.unnormalize(legacy.normalize(x.clone()));lp=f(legacy_back,s)[0,:,:22].cpu().numpy()
        if not row["Root_observed"]:
            p=p-p[:,:1];target=target-target[:,:1];lp=lp-lp[:,:1]
        from dunhuang.contracts import matrices
        from scipy.spatial.transform import Rotation
        R=matrices(raw[:,7:].reshape(T,22,6),row=True);S=matrices(back[0,:,7:].cpu().numpy().reshape(T,22,6),row=True)
        angle=float(Rotation.from_matrix((R.swapaxes(-1,-2)@S).reshape(-1,3,3)).magnitude().max())
        position=float(np.linalg.norm(p-target,axis=-1).max())
        derivative=derivatives(p,target,np.ones(T,bool))
        physical=all(derivative[str(k)]["rmse"]<=cfg["numerical_derivative_absolute_floors"][k-1] for k in (1,2,3))
        if row["Root_observed"]:
            affine=x*legacy.scaler.scale_.to(x)+legacy.scaler.min_.to(x)
            other=bool((affine[:,:,:7].abs()<=1+1e-6).all())
        else:other=None
        arrays[row["uid"]+"_target"]=z[0].cpu().numpy();arrays[row["uid"]+"_decoded22"]=p
        if not row["Root_observed"]:arrays[row["uid"]+"_target"][:,:7]=np.nan
        first=R[:,:,0]
        sign_impossible=((first<-1e-7)&(lower[None,:,:3]>=0))|((first>1e-7)&(upper[None,:,:3]<=0))|((abs(first)<=1e-12)&((lower[None,:,:3]>1e-7)|(upper[None,:,:3]<-1e-7)))
        witnesses=np.argwhere(sign_impossible)
        displacement=np.linalg.norm(lp-target,axis=-1);where=np.unravel_index(np.argmax(displacement),displacement.shape)
        passed=angle<=cfg["angle_max_rad"] and position<=cfg["position_max_m"] and physical and other is not False
        records.append(dict(uid=row["uid"],domain=row["domain"],role=row["role"],case=row["case"],Root_observed=row["Root_observed"],
            angle_max_rad=angle,position_max_m=position,derivatives=derivative,Root_C4_support=other,qualified=bool(passed),
            actual_box_first_vector_sign_witnesses=witnesses.tolist(),sign_impossible_joint_frames=int(sign_impossible.any(-1).sum()),
            witness_scope="Any positive alpha violates an actual raw coordinate bound; frame/joint/component indices, tolerance1e-7",
            legacy_roundtrip_max_displacement_m=float(displacement[where]),legacy_worst_window_frame=int(where[0]),legacy_worst_joint=int(where[1])) )
    # Compatibility of added zero residual I/O is established with real COVER sampling.
    from dunhuang.research.continuity_r30.common import checked_parent,parent_config,values
    restore(bundle.reference,checked_parent(parent_config(),"models/COVER/adapter.pt"))
    rows,cases=values(parent_config(),"aist","train");music=torch.tensor(np.stack([r["music"] for r in rows]),device="cuda")
    old=np.load(checked_parent(parent_config(),"evaluation_coverage/arrays.npz"))
    noise=[]
    for mode in ("correct","none"):
        pred,nr,_=sample(d,bundle,packet(rows) if mode=="correct" else None,music,read("src/dunhuang/research/time_r29/config.json")["real_eval_seeds"][0])
        assert np.array_equal(pred.cpu().numpy(),old[f"train_COVER_{mode}_0_normalized"])
        arrays["LegacyIO_initial_"+mode]=pred.cpu().numpy();noise.append(dict(mode=mode,noise=nr))
    source_relative=[]
    from dunhuang.research.time_r29.dynamics import decomposition
    for role in ("train","holdout"):
        cases=cfg["train_cases"] if role=="train" else cfg["holdout_cases"]
        for i,case in enumerate(cases):
            src=next(r for r in data if r["domain"]=="aist" and r["case"]==case)["data"]["model_points"]
            sj=decomposition(src)["regions"]["core_interior"]["world"]
            for k in range(2):
                nj=decomposition(old[f"{role}_COVER_none_{k}_22"][i])["regions"]["core_interior"]["world"]
                source_relative.append(dict(role=role,case=case,seed_index=k,source_Core_jerk_p95=sj,NoRef_Core_jerk_p95=nj,source_passes_old_relative_jerk=bool(sj<=1.1*nj)))
    assert all(not p.requires_grad for p in d.model.parameters())
    result=dict(records=records,new_contract_training_admitted=all(r["qualified"] for r in records),
        LegacyIO_initial_COVER_exact=True,source_vs_NoRef_relative_jerk=source_relative,parameters=sum(p.numel() for p in bundle.parameters()),reference_parameters=sum(p.numel() for p in bundle.reference.parameters()),
        noise=noise,sampler_calls=2,generated_trajectories=8,optimizer_steps=0,frozen_backbone_digest=digest(d.model),
        new_qualification_not_inherited=True,operational_qualified=False,whole_song_generations=0,full_native_status="NOT_EVALUABLE")
    bundle.close();np.savez_compressed(out/"arrays.npz",**arrays)
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,arrays_sha256=sha(out/"arrays.npz")))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
    print("R31_ACTION_CONTRACT_ADMISSION",result["new_contract_training_admitted"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
