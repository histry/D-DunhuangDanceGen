"""Pinned official Local decoder and samplers, with explicit incomplete contract."""
import argparse,sys,pickle,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch,yaml
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.context_r21.data import same_audio_pcm
from .local_contract import common22,require_native24

def load(cfg):
    base=Path("inputs/lodge_r22");manifest=read(base/"runtime_manifest.json")
    assert sha(base/"runtime_manifest.json")==cfg["lodge_manifest_sha256"]
    assert manifest["source_commit"]==cfg["lodge_source_commit"]
    for name,digest in manifest["files"].items():assert sha(base/name)==digest,name
    official=(base/"official").resolve();sys.path.insert(0,str(official))
    # Python >=3.8 already has protocol 5; alias only the obsolete backport name.
    sys.modules["pickle5"]=pickle
    from dld.data.render_joints.smplfk import SMPLX_Skeleton,do_smplxfk
    from dld.models.architectures.model import Refine_DanceDecoder
    from dld.models.architectures.diffusion import GaussianDiffusion
    from dld.data.utils.audio import extract
    from dld.data.utils.preprocess import Normalizer,MinMaxScaler
    # Official public Normalizer contains only these reviewed classes and torch tensors.
    normalizer=torch.load(base/"official/data/Normalizer.pth",map_location="cpu")
    config=yaml.safe_load((base/manifest["config"]).read_text())
    model_args=dict(config["model"]["DanceDecoder"]["params"]);model_args.update(nfeats=139,seq_len=256)
    skeleton=SMPLX_Skeleton(Jpath=str(official/"data/smplx_neu_J_1.npy"))
    model=Refine_DanceDecoder(smplx_model=skeleton,normalizer=normalizer,genre_num=16,**model_args).cuda().eval()
    ckpt=torch.load(base/manifest["checkpoint"],map_location="cpu");states=ckpt["state_dict"]
    selected={k[len("DanceDecoder."):]:v for k,v in states.items() if k.startswith("DanceDecoder.")}
    assert all(torch.equal(v,states["diffusion.model."+k]) for k,v in selected.items())
    model.load_state_dict(selected,strict=True)
    diff_args=dict(config["model"]["diffusion"]["params"]);diff_args.update(horizon=256,repr_dim=139)
    dc=SimpleNamespace(Norm=True,FINEDANCE=SimpleNamespace(nfeats=139),soft=1.)
    diffusion=GaussianDiffusion(cfg=dc,model=model,normalizer=normalizer,smplx_model=skeleton,**diff_args).cuda().eval()
    for name in ("betas","alphas_cumprod"):assert torch.equal(getattr(diffusion,name).cpu(),states["diffusion."+name])
    for p in diffusion.parameters():p.requires_grad_(False)
    checkpoint_info=dict(epoch=ckpt["epoch"],global_step=ckpt["global_step"],decoder_tensors=len(selected),
        decoder_layers=len(model.seqTransDecoder.stack),refinement_attention_layers=len(model.refine_seqTransDecoder1.stack),
        sampling_weights="DanceDecoder == diffusion.model; EMA not selected",strict_load=True,
        checkpoint_sha256=sha(base/manifest["checkpoint"]),source_commit=manifest["source_commit"],
        compat_alias="pickle5 -> Python standard pickle; no model mathematics patch")
    del ckpt,states
    return diffusion,normalizer,skeleton,do_smplxfk,extract,checkpoint_info,manifest

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    diffusion,norm,skeleton,fk,extract,info,manifest=load(cfg);rows=read(root/"data/result.json")["records"]
    music=[];sources=[];points=[];bindings=[]
    for case in cfg["lodge_cases"]:
        row=next(r for r in rows if r["case"]==case);left=row["left"]
        full=np.load(checked(left["source_native"]));start=left["start_frame"];native=full[start:start+256];assert native.shape==(256,151)
        clock=read(checked(left["clock"]));assert clock["status"]=="VERIFIED_PAIR" and clock["calibration"]["verified"]
        cal=clock["calibration"];raw_audio=checked(left["audio"])
        audio,sr=sf.read(raw_audio,dtype="float32",always_2d=True);a=round((cal["audio_offset_seconds"]+start/30)*sr);b=a+round(256/30*sr)
        assert a>=0 and b<=len(audio)
        wav=out/(Path(left["raw_motion"]["path"]).stem+"_lodge256.wav");sf.write(wav,audio[a:b],sr,subtype="FLOAT")
        features,_=extract(str(wav),skip_completed=False);assert len(features)>=256
        features=features[:256].astype(np.float32);assert features.shape==(256,35) and np.isfinite(features).all();music.append(features)
        # Native stores matrix columns; PyTorch3D 6D stores the first two rows.
        R=matrices(native[:,7:].reshape(-1,24,6))[:,:22]
        rot=R[...,:2,:].reshape(256,132).astype(np.float32)
        x=np.concatenate((np.zeros((256,4),np.float32),native[:,4:7],rot),-1).copy()
        x[:,[4,6]]-=x[:1,[4,6]]
        with torch.no_grad():
            tensor=torch.tensor(x[None],device="cuda")
            before=fk(tensor,skeleton)[0,:,:22].cpu().numpy()
            shift=-1.2-float(np.quantile(before[:,[7,8,10,11],1],.02));x[:,5]+=shift
            source=fk(torch.tensor(x[None],device="cuda"),skeleton)[0,:,:22].cpu().numpy()
        sources.append(x);points.append(source)
        bindings.append(dict(case=case,role=row["role"],source=left,audio=ref(wav),sample_start=a,sample_end=b,sample_rate=sr,
            native_frames=256,music_frames=256,feature_dim=35,music_trim_policy="Discard centered STFT extra final frame; never stretch or pad",
            coordinates="Y-up world Root; horizontal origin relative; documented rigid vertical shift to official -1.2 ground",
            root_y_shift=shift,source_C4_observed=False,joint_validity=[True]*22+[False]*2,
            native24_admission="NOT_EVALUABLE_MISSING_TWO_JOINTS"))
        if replay:assert same_audio_pcm(wav,Path(replay)/wav.name)
    m=torch.tensor(np.stack(music),device="cuda");raw=torch.tensor(np.stack(sources),device="cuda")
    source_norm=norm.normalize(raw.clone());mask=torch.zeros_like(raw)
    # Source primitives stay soft cues in the official sampler, not fabricated hard truth.
    for a,b in ((0,4),(252,256),(28,36),(60,68),(156,164),(188,196)):mask[:,a:b,7:]=1
    arrays=dict(music=m.cpu().numpy(),source139=raw.cpu().numpy(),source22=np.stack(points),cue_mask=mask.cpu().numpy())
    results=[]
    with torch.no_grad():
        for index,mode in enumerate(("native_unconditioned","native_primitives","mismatched_primitives")):
            torch.manual_seed(cfg["seed"]+99000);torch.cuda.manual_seed_all(cfg["seed"]+99000)
            if mode=="native_unconditioned":sample=diffusion.ddim_sample((2,256,139),m,None)
            else:
                value=source_norm if mode=="native_primitives" else source_norm.flip(0)
                sample=diffusion.inpaint_soft_ddim((2,256,139),m,None,constraint=dict(value=value,mask=mask))
            normalized=sample.detach().clone();motion=norm.unnormalize(sample.detach().clone())
            xyz=fk(motion,skeleton)[:,:,:22].cpu().numpy();motion=motion.cpu().numpy()
            assert np.isfinite(motion).all() and np.isfinite(xyz).all()
            arrays[mode+"_normalized"]=normalized.cpu().numpy();arrays[mode+"_139"]=motion;arrays[mode+"_22"]=xyz
            for i,binding in enumerate(bindings):
                common=common22(xyz[i])
                RM=matrices(motion[i,:,7:].reshape(256,22,6),row=True);GT=matrices(sources[i][:,7:].reshape(256,22,6),row=True)
                angles=Rotation.from_matrix((GT.swapaxes(-1,-2)@RM).reshape(-1,3,3)).magnitude().reshape(256,22)
                editable=~mask[i,:,7:].bool().all(-1).cpu().numpy()
                common.update(mode=mode,case=binding["case"],pose_error_rad=float(angles[editable].mean()),soft_cue_pose_error_rad=float(angles[~editable].mean()),
                    root_rmse_m=float(np.sqrt(np.mean((motion[i,:,4:7]-sources[i][:,4:7])**2))))
                results.append(common)
            print("R22_LODGE_SAMPLE",mode,flush=True)
    try:require_native24(motion[0],[True]*22+[False]*2,30)
    except ValueError as e:admission=str(e)
    else:raise AssertionError("Incomplete Lodge representation entered Native")
    path=out/"arrays.npz";np.savez_compressed(path,**arrays)
    result=dict(checkpoint=info,bindings=bindings,common22=results,full_native_status="NOT_EVALUABLE",native24_rejection=admission,
        batched_sampler_calls=3,generated_trajectories=6,sampling_steps=50,fps=30,genre_condition=None,
        official_finedance_pretrained=True,input_domain="AIST real source-clock music; exploratory cross-dataset system baseline",
        root_observed=False,contacts_observed=False,endpoint_cues="4 frames, rotation only, official soft sampler; no hard-preservation claim",
        comparable_to_R22_native=False,checkpoint_training_data_and_budget_matched=False,
        asset_manifest=manifest,arrays=ref(path),human_reviewed=False,optimizer_steps=0,whole_song_generations=0,operational_qualified=False)
    if replay:
        old=read(Path(replay)/"result.json");same_arrays(path,checked(old["arrays"]))
        for name in ("checkpoint","common22","native24_rejection","batched_sampler_calls","generated_trajectories"):assert result[name]==old[name]
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info,asset_manifest=manifest))
    print("R22_LODGE_COMPLETE",info,"Native24",admission,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
