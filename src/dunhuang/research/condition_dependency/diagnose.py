"""Frozen B2 forward/canonical-origin/CFG audit. Never trains or samples a trajectory."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,sha,write
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.adapter import BoundaryContextAdapter
from dunhuang.research.boundary_context.common import load_model,digest,Data,masks,fixed_rows
from .codec import encode

def run(cfg,out,replay=None):
    out=fresh(out);parent=Path(cfg["b2"]);seal=read("reports/r9_summary.json")
    assert sha(parent/"train/result.json")==seal["evidence"]["b2/train/result.json"]
    training=read(parent/"train/result.json");payload=torch.load(checked(training["checkpoint"]),map_location="cpu",weights_only=True)
    pf=Path(cfg["main"])/"training_preflight";pp=read(pf/"protocol.json");assets=read(checked(pp["assets"]));export=np.load(checked(assets["scaler"]));data=Data(export)
    model,diffusion,norm=load_model(assets);before=digest(model);adapter=BoundaryContextAdapter().cuda().eval();adapter.load_state_dict(payload["state"],strict=True)
    rows=fixed_rows(read(pf/"rows.json"),"val",cfg["val_cases"],20261114);items=[];coordinates=[]
    for i,row in enumerate(rows):
        native=data.native(row);edge,o=encode(native);known=np.zeros((150,151),bool);known[52:60,4:]=True;known[90:98,4:]=True
        items.append(dict(name="val_"+str(i),target=edge,known=known,feature=np.load(checked(row["jukebox"])),source=row["source_native"]))
    cp=read(Path(cfg["r8"])/"boundary_audit/candidates.json")
    for c in cp:
        native=np.load(checked(c["native"]))[:150];edge,o=encode(native);old=MotionAdapter.to_edge(native)
        context=np.load(checked(c["context"]));mask=context["mask"][0];expected_mask=np.zeros_like(mask);expected_mask[52:60,4:]=True;expected_mask[90:98,4:]=True;old_norm=old*export["scale"]+export["bias"];new_norm=edge*export["scale"]+export["bias"]
        delta=old_norm-new_norm;expected=np.zeros_like(delta);expected[:,4:7]=o@np.array([[1,0,0],[0,0,-1],[0,1,0]],np.float32).T*export["scale"][4:7]
        assert np.max(abs(delta[mask]-expected[mask]))<2e-6
        coordinates.append(dict(case=c["name"],origin_native=o.tolist(),mask_equal=bool(np.array_equal(mask,expected_mask)),
            old_actual_context_roundtrip_max=float(abs(context["values"][0][mask]-old[mask]).max()),
            normalized_input_shift_max=float(abs(delta[mask]).max()),affine_shift_identity_error=float(abs(delta[mask]-expected[mask]).max()),
            root_d1_change_mps=float(abs(np.diff(edge[:,4:7],axis=0)-np.diff(old[:,4:7],axis=0)).max()*30),
            root_d2_change_mps2=float(abs(np.diff(edge[:,4:7],n=2,axis=0)-np.diff(old[:,4:7],n=2,axis=0)).max()*900),
            canonical_context_supported=bool((abs(new_norm[mask])<=1+32*np.finfo(np.float32).eps).all())))
        items.append(dict(name=c["name"],target=old,known=mask,feature=np.load(checked(c["features"]))[0],source=c["native"]))
    state={};residuals=[]
    def hook(module,args,value):
        if state["control"]=="zero":return value
        r=adapter(state["value"],state["known"],value)
        if state["control"].startswith("difference_"):r=r-adapter(state["value"],torch.zeros_like(state["known"]),value)
        residuals.append(r.detach());return value+r
    handle=model.cond_encoder.register_forward_hook(hook);records=[]
    for k in range(0,len(items),2):
        batch=items[k:k+2];target=torch.tensor(np.stack([x["target"] for x in batch])*export["scale"]+export["bias"],device="cuda")
        known=torch.tensor(np.stack([x["known"] for x in batch]),device="cuda");music=torch.tensor(np.stack([x["feature"] for x in batch]),device="cuda")
        for t in cfg["timesteps"]:
            torch.manual_seed(cfg["seed"]+k+t);torch.cuda.manual_seed_all(cfg["seed"]+k+t)
            ts=torch.full((len(batch),),t,dtype=torch.long,device="cuda");xt=diffusion.q_sample(target,ts);saved={}
            for control in ("zero","observed","removed","wrong","difference_observed","difference_removed","difference_wrong"):
                value=target.roll(1,0) if control.endswith("wrong") else target
                state.update(control=control,value=value,known=torch.zeros_like(known) if control.endswith("removed") else known)
                residuals.clear()
                with torch.no_grad():
                    unc=model(xt,music,ts,cond_drop_prob=1.);cond=model(xt,music,ts,cond_drop_prob=0.)
                    cfgpred=unc+2*(cond-unc);clipped=cfgpred.clamp(-1,1)
                residual_norm=[r.square().mean().sqrt().item() for r in residuals]
                saved[control]=(unc,cond,cfgpred,clipped)
                unknown=torch.zeros_like(known);unknown[:,60:90,4:]=True
                rec=dict(population=[x["name"] for x in batch],t=t,control=control,residual_rms_by_branch=residual_norm,
                    masked_cond_mse=float((cond-target)[unknown].square().mean()),masked_cfg_mse=float((clipped-target)[unknown].square().mean()))
                if control!="zero":
                    base=saved["zero"];rec.update(unconditional_effect=float(abs(unc-base[0]).max()),conditional_effect_rms=float((cond-base[1]).square().mean().sqrt()),
                        cfg_effect_rms=float((cfgpred-base[2]).square().mean().sqrt()),clip_fraction=float((abs(cfgpred)>1).float().mean()))
                records.append(rec)
            for c in ("removed","wrong"):
                a=saved["observed"];b=saved[c]
                from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle
                scale=torch.tensor(export["scale"],device="cuda");bias=torch.tensor(export["bias"],device="cuda")
                ra=rotation_6d_to_matrix(((a[3]-bias)/scale)[:,60:90,7:].reshape(-1,30,24,6))
                rb=rotation_6d_to_matrix(((b[3]-bias)/scale)[:,60:90,7:].reshape(-1,30,24,6))
                response=matrix_to_axis_angle(ra.transpose(-1,-2)@rb).norm(dim=-1)
                records.append(dict(population=[x["name"] for x in batch],t=t,root_rotation_response_rad=float(response[:,:,0].mean()),body_response_rad=float(response[:,:,1:].mean()),comparison="observed_vs_"+c,conditional_rms=float((a[1]-b[1]).square().mean().sqrt()),
                    cfg_rms=float((a[3]-b[3]).square().mean().sqrt()),root_delta_m_rms=float(((a[3][:,:,4:7]-b[3][:,:,4:7])/torch.tensor(export["scale"][4:7],device="cuda")).square().mean().sqrt())))
    handle.remove();assert digest(model)==before
    result=dict(coordinates=coordinates,forwards=records,model_digest=before,checkpoint=training["checkpoint"],new_optimizer_steps=0,new_neural_generations=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,checkpoint=training["checkpoint"],preflight=pp,items=[dict(name=x["name"],source=x["source"]) for x in items]))
    if replay:assert result==read(Path(replay)/"result.json")
    print("R10_P0_FROZEN_COMPLETE",len(items),len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
