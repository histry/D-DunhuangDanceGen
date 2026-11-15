"""Read-only Lodge wiring/normalization and retargeted position-only diagnostics."""
import argparse,importlib.util,sys
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.contracts import matrices
from dunhuang.research.branch_r22.lodge import load as lodge_load
from dunhuang.research.branch_r22.local_contract import require_native24
from dunhuang.research.state_bridge.common import native_environment
from .common import inputs,finish,arrays,read,fresh,forbid_optimizers

def positions24(root,local22,parents,offsets):
    if local22.shape!=(len(root),22,3,3) or len(parents)!=24:raise ValueError("Mapped complete first22 required")
    if any(j in parents for j in (22,23)):raise ValueError("Missing rotations must be terminal leaves")
    world=[];xyz=[]
    for j,parent in enumerate(parents):
        if j==0:world.append(local22[:,0]);xyz.append(root)
        else:
            xyz.append(xyz[parent]+np.einsum("tij,j->ti",world[parent],offsets[j]))
            world.append(world[parent]@local22[:,j] if j<22 else None)
    return np.stack(xyz,1)

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();root,pc,pop=inputs(cfg);native_environment(read(root/"data/protocol.json")["assets"])
    from motion.geometry import fk_24_np
    from motion_geometry.smpl24 import PARENTS,OFFSETS
    diffusion,norm,skeleton,fk,extract,info,manifest=lodge_load(pc)
    assert info["decoder_layers"]==9 and info["refinement_attention_layers"]==0
    assert list(skeleton.parents[:22])==list(PARENTS[:22])
    prior=read(root/"lodge/result.json");z=dict(np.load(checked(prior["arrays"])))
    raw=torch.tensor(z["source139"],device="cuda")
    normalized=norm.normalize(raw.clone());roundtrip=norm.unnormalize(normalized.clone())
    raw_affine=raw*norm.scaler.scale_.to(raw.device)+norm.scaler.min_.to(raw.device)
    details=[]
    for i,b in enumerate(prior["bindings"]):
        errors=(roundtrip[i]-raw[i]).cpu().numpy()
        P=matrices(roundtrip[i,:,7:].cpu().numpy().reshape(256,22,6),row=True)
        G=matrices(raw[i,:,7:].cpu().numpy().reshape(256,22,6),row=True)
        angle=np.arccos(np.clip((np.trace(G.swapaxes(-1,-2)@P,axis1=-1,axis2=-2)-1)/2,-1,1))
        details.append(dict(case=b["case"],normalized_outside_fraction=float((abs(raw_affine[i])>1).float().mean()),
            root_roundtrip_rmse_m=float(np.sqrt(np.mean(errors[:,4:7]**2))),pose_roundtrip_mean_rad=float(angle.mean()),
            cue_pose_roundtrip_mean_rad=float(angle[z["cue_mask"][i,:,7:].all(-1)].mean()),
            C4_not_observed=True))
    model=diffusion.model
    torch.manual_seed(23041);x=torch.randn_like(raw);m=torch.tensor(z["music"],device="cuda");t=torch.tensor([519,519],device="cuda")
    with torch.no_grad():before=model(x,m,None,t,cond_drop_prob=0)
    handle=model.refine_cond_projection1.register_forward_hook(lambda mod,args,value:value+100.)
    try:
        with torch.no_grad():perturbed=model(x,m,None,t,cond_drop_prob=0)
    finally:handle.remove()
    assert torch.equal(before,perturbed)
    dead=model.refine_cond_projection1.weight;live=model.refine_final_layer1.weight
    assert not dead.requires_grad and not live.requires_grad
    dead.requires_grad_(True);live.requires_grad_(True)
    try:
        output=model(x,m,None,t,cond_drop_prob=0)
        g=torch.autograd.grad(output.square().mean(),dead,allow_unused=True)[0]
    finally:dead.requires_grad_(False);live.requires_grad_(False)
    values=dict(normalized=normalized.cpu().numpy(),roundtrip=roundtrip.cpu().numpy(),
        source139=z["source139"],source22=z["source22"],forward=before.cpu().numpy())
    position_rows=[]
    for i,b in enumerate(prior["bindings"]):
        source=b["source"];full=np.load(checked(source["source_native"]));a=source["start_frame"];native=full[a:a+256]
        sourceR=matrices(z["source139"][i,:,7:].reshape(256,22,6),row=True)
        assert np.max(abs(sourceR-matrices(native[:,7:].reshape(256,24,6))[:,:22]))<1e-6
        expected=fk_24_np(native)
        derived=positions24(native[:,4:7],sourceR,PARENTS,OFFSETS)
        assert np.max(abs(derived-expected))<1e-5
        for mode in ("native_unconditioned","native_primitives","mismatched_primitives"):
            motion=z[mode+"_139"][i];R=matrices(motion[:,7:].reshape(256,22,6),row=True)
            points=positions24(motion[:,4:7],R,PARENTS,OFFSETS);values[f"{i}_{mode}_retarget24"]=points
            jerk=np.linalg.norm(np.diff(points,n=3,axis=0)*30**3,axis=-1)
            position_rows.append(dict(case=b["case"],mode=mode,retarget24_jerk_p95=float(np.percentile(jerk,95)),retarget24_jerk_max=float(jerk.max()),
                terminal_hand_jerk_p95=float(np.percentile(jerk[:,22:],95)),source_fk_parity_max_m=float(np.max(abs(derived-expected))),
                complete_rotations24=False,full_native_status="NOT_EVALUABLE"))
    arrays(out,"external_arrays.npz",values,replay)
    finish(out,dict(checkpoint=info,normalizer_roundtrip=details,FK_condition_perturbation_output_exact=True,
        FK_projection_gradient_unused=g is None,missing_leaf_rotations_unknown=True,retargeted_positions=position_rows,
        strict_architecture_unmodified=True,new_external_sampler_calls=0,full_native_status="NOT_EVALUABLE",optimizer_steps=0),cfg,replay)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.out,a.replay_from)
