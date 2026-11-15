"""Pinned input conversion and original FineDance preprocessing, separate from frozen studies."""
import argparse,ast,contextlib,glob,os
from pathlib import Path
import numpy as np
import torch
from .input_contract import alignment_height,height_features,translate
from .representation import rotations
from dunhuang.research.state_r27.common import read,sha,checked,write,fresh,forbid_optimizers,matrices,metrics,commit,same_arrays
from dunhuang.research.state_r27.common import runtime as old_runtime

def originals(cfg):
    root=Path(cfg["r26"]);assert sha(root/"artifacts_manifest.json")==cfg["r26_manifest_sha256"]
    manifest=read(root/"artifacts_manifest.json")["files"];pop=read(root/"data/result.json")
    cases=cfg["train_cases"]+cfg["holdout_cases"];rows=[next(r for r in pop["records"] if r["case"]==c) for c in cases]
    assert all(r["role"]=="train" for r in rows)
    assert len({r["recording_uid"] for r in rows})==8
    assert not {r["source_group"] for r in rows[:4]}&{r["source_group"] for r in rows[4:]}
    data={}
    for r in rows:
        name="data/"+r["arrays"];assert sha(root/name)==manifest[name]["sha256"]
        data[r["case"]]=dict(np.load(root/name))
    return rows,data

def official_functions(s,f):
    base=Path("inputs/lodge_r22");manifest=read(base/"runtime_manifest.json")
    names=["data/code/preprocess.py","dld/data/FineDance_dataset.py","dld/models/modeltype/base.py","dld/data/pre/FineDance_normalizer.py","configs/data/assets.yaml"]
    sources={}
    for name in names:
        p=base/"official"/name;assert sha(p)==manifest["files"]["official/"+name];sources[name]=sha(p)
    tree=ast.parse((base/"official"/names[0]).read_text())
    functions=[x for x in tree.body if isinstance(x,ast.FunctionDef)]
    from dld.data.render_joints.smplfk import ax_from_6v,ax_to_6v
    env=dict(np=np,torch=torch,os=os,glob=glob,tqdm=lambda x:x,floor_height=0,
        SMPLX_Skeleton=lambda:s,do_smplxfk=f,ax_from_6v=ax_from_6v,ax_to_6v=ax_to_6v)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(base/"official"/names[0]),"exec"),env)
    return env,sources

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();rows,data=originals(cfg)
    d,n,s,f,e,info,a=old_runtime(read("src/dunhuang/research/state_r27/config.json"));a.close()
    env,source_hashes=official_functions(s,f);records=[]
    for row in rows:
        c=row["case"];old=data[c];native=old["native"]
        full=np.load(checked(row["source_native"]));first=full[:1]
        R=matrices(first[:,7:].reshape(1,24,6))[:,:22]
        first139=np.concatenate((np.zeros((1,4),np.float32),first[:,4:7],R[...,:2,:].reshape(1,132)),1).astype(np.float32)
        with torch.no_grad():p=f(torch.tensor(first139[None],device="cuda"),s)[0,:,:22].cpu().numpy()
        height=alignment_height(p[0,[10,11],1])
        expected=env["set_on_ground_139"](torch.tensor(first139.copy(),device="cuda"),s).cpu().numpy()
        shifted=translate(first139,np.array([0,-height,0],np.float32))
        parity=float(abs(expected-shifted).max());assert parity<2e-6
        R=matrices(native[:,7:].reshape(256,24,6))[:,:22]
        original=np.concatenate((np.zeros((256,4),np.float32),native[:,4:7],R[...,:2,:].reshape(256,132)),1).astype(np.float32)
        offset=np.array([-original[0,4],-height,-original[0,6]],np.float32)
        model=translate(original,offset)
        with torch.no_grad():points=f(torch.tensor(model[None],device="cuda"),s)[0,:,:22].cpu().numpy()
        model[:,:4]=height_features(points)
        variants={"legacy":old["raw139"],"model":model};statistics={}
        arrays=dict(music=old["music"],native=native,raw_original=original,model_offset=offset,
            legacy_offset=old["raw139"][0,4:7]-original[0,4:7])
        for mode,raw in variants.items():
            with torch.no_grad():
                value=torch.tensor(raw[None],device="cuda");target=n.normalize(value.clone());back=n.unnormalize(target.clone())
                xyz=f(value,s)[0,:,:22];q=f(back,s)[0,:,:22]
            arrays[mode+"_raw"]=raw;arrays[mode+"_target"]=target[0].cpu().numpy();arrays[mode+"_points"]=xyz.cpu().numpy()
            statistics[mode]=dict(root_roundtrip_rmse_m=float((back[:,:,4:7]-value[:,:,4:7]).square().mean().sqrt()),
                body_roundtrip_mse=float((back[:,:,7:]-value[:,:,7:]).square().mean()),
                clipping={k:float((target[:,:,lo:hi].abs()>1).float().mean()) for k,lo,hi in (("C4",0,4),("Root",4,7),("body",7,139))},
                normalized_C4_mean=float(target[:,:,:4].mean()),C4_feature_rate=raw[:,:4].mean(0).tolist(),
                original_target_vs_official_roundtrip=metrics(back[0].cpu().numpy(),q.cpu().numpy(),raw,xyz.cpu().numpy()))
        assert np.allclose(translate(model,-offset)[:,4:],original[:,4:],atol=2e-6)
        assert np.array_equal(rotations(old["raw139"][64:160,7:]),rotations(model[64:160,7:]))
        path=out/f"{c}.npz";np.savez_compressed(path,**arrays)
        records.append(dict(case=c,role="train" if c in cfg["train_cases"] else "adaptation_holdout_exposed_R26",source=row,arrays=path.name,
            source_first_frame_ground_height_m=height,model_to_native_translation=(-offset).tolist(),legacy_to_native_translation=(-arrays["legacy_offset"]).tolist(),
            official_first_frame_parity_max=parity,statistics=statistics,reference_equal_across_ground_C4_contracts=True,
            C4_scope="HEIGHT_DERIVED_C4_NOT_MEASURED_CONTACT",ground_scope="PUBLISHED_MODEL_INPUT_CONVENTION_NOT_NATIVE_GROUND_CALIBRATION"))
    assets=Path(cfg["asset_root"])/"finedance";asset_receipt=read(assets/"asset_receipt.json")
    for row in asset_receipt["records"]:assert sha(assets/row["path"])==row["sha256"]
    fd=out/"finedance";fd.mkdir()
    with (fd/"preprocess.log").open("w") as log,contextlib.redirect_stdout(log),torch.no_grad():
        env["motion_feats_extract"](str(assets/"motion"),str(fd/"motion319"),str(assets/"music_npy"),str(fd/"music35"))
    fine=[]
    for name in cfg["finedance_ids"]:
        motion=np.load(fd/"motion319"/(name+".npy"));music=np.load(fd/"music35"/(name+".npy"))
        assert motion.shape[1]==319 and music.shape[1]==35 and np.isfinite(motion).all() and np.isfinite(music).all()
        length=min(len(motion),len(music));assert abs(len(motion)-len(music))<=1 and length>=256
        with torch.no_grad():p=f(torch.tensor(motion[None,:,:139],device="cuda"),s)[0,:,:22].cpu().numpy()
        difference=float(abs(motion[:,:4]-height_features(p)).max());assert difference==0
        fine.append(dict(sequence_id=name,motion_frames=len(motion),music_frames=len(music),common_prefix_frames=length,
            official_joint_trim_frames=240 if name=="004" else 0,common_prefix_only_no_resampling=True,
            motion_sha256=sha(fd/"motion319"/(name+".npy")),music_sha256=sha(fd/"music35"/(name+".npy")),
            C4_FK_height_parity_max=difference,paired_identity_scope="OFFICIAL_ARCHIVE_MATCHING_SEQUENCE_ID_AND_COMMON_PREPROCESSING",
            feature_scope="ARCHIVE_35D_LODGE_INTERFACE_COMPATIBLE_PRODUCER_LIBRARY_VERSION_UNKNOWN",
            pretraining_exposure_unknown=True,training_admitted_this_round=False))
    result=dict(records=records,FineDance=fine,source_code_hashes=source_hashes,
        checkpoint=info,normalizer=dict(sha256=sha("inputs/lodge_r22/official/data/Normalizer.pth"),data_min=n.scaler.data_min_[:7].tolist(),data_max=n.scaler.data_max_[:7].tolist(),
            producer_training_archive_hash_unknown=True,released_config_feature_contract_verified=True),
        input_contract="FIRST_FULL_RECORDING_TOES_AVERAGE_IF_DIFFERENCE_LT_2CM_ELSE_MIN; FLOOR0; HEIGHT_C4 .12/.12/.05/.05; CROP_XZ_ORIGIN",
        original_and_roundtrip_targets_separate=True,fixed_source_transform_never_fit_to_generated_motion=True,
        fine_asset_receipt_sha256=sha(assets/"asset_receipt.json"),optimizer_steps=0,full_native_status="NOT_EVALUABLE",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,files={str(p.relative_to(out)):sha(p) for p in out.rglob("*.npy")} | {p.name:sha(p) for p in out.glob("*.npz")}))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in list(out.glob("*.npz"))+list(out.rglob("*.npy")):
            q=Path(replay)/p.relative_to(out)
            if p.suffix==".npz":same_arrays(p,q)
            else:assert np.array_equal(np.load(p),np.load(q))
    print("R28_INPUT_CONTRACT_READY",len(records),len(fine),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.out,args.replay_from)
