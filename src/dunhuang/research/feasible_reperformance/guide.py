"""Reuse the immutable R8 pose guide; instrument / replace only explicit R9 conditions."""
import copy
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.contracts import Q
from dunhuang.research.source_capacity.common import read
from dunhuang.research.primitive_reperformance.reference import build
LOWER=(1,2,4,5,7,8,10,11)

def prepare(q,normalizer,out,device):
    from pytorch3d.transforms import rotation_6d_to_matrix as matrix,matrix_to_axis_angle as log,axis_angle_to_matrix as exp
    from dunhuang.research.primitive_reperformance.guide import prepare as r8
    relative=q.get("r9_relative_root",False);teacher=None;step=0
    original=copy.deepcopy(q);group=q["condition_group"]
    if relative:original["reperformance_config"]["groups"][group]=[j for j in original["reperformance_config"]["groups"][group] if j!=0]
    parent=r8(original,normalizer,out,device);T=q["frames"]
    plan=read(checked(q["event_plan"]));ref,_,weights,_=build(plan,q["reperformance_config"])
    from dunhuang.contracts import MotionAdapter
    target=matrix(torch.tensor(MotionAdapter.to_edge(ref)[:,7:13],device=device))
    scale=normalizer.scaler.scale_.to(device);bias=normalizer.scaler.min_.to(device)
    S=scale[7:13];B=bias[7:13];basis=torch.tensor(Q,device=device)
    channels=torch.tensor([7+6*j+k for j in LOWER for k in range(6)],device=device)
    if q.get("r9_lower_teacher"):
        teacher=torch.tensor(np.load(checked(q["r9_lower_teacher"]))["lower"],device=device)
        if teacher.shape!=(50,1,150,48):raise ValueError("Frozen matched T0 lower trajectory required")
    predictions=[];guided=[];lower=[];fractions=[]
    def guide(predicted,progress):
        nonlocal step
        before=matrix((predicted[0,:T,7:13]-B)/S)
        output=parent(predicted,progress)
        if relative:
            old=matrix((output[0,:T,7:13]-B)/S)
            # Gauge-free relative source rotation; no assumption about the first noisy heading.
            goal=old[0]@(target[0].T@target)
            w=torch.tensor(weights[:,0],device=device)*progress**2
            omega=log(old.transpose(-1,-2)@goal)*w[:,None]
            vectors=((output[0,:T,7:13]-B)/S).reshape(T,2,3)
            result=output[0,:T,7:13].clone();accepted=torch.zeros(T,dtype=torch.bool,device=device);dose=torch.zeros(T,device=device)
            for k in range(9):
                factor=2.**(-k);candidate=old@exp(omega*factor)
                encoded=(vectors@(old.transpose(-1,-2)@candidate)).reshape(T,6)*S+B
                good=(encoded.abs().amax(-1)<=1+32*torch.finfo(encoded.dtype).eps)&~accepted
                result=torch.where(good[:,None],encoded,result);dose=torch.where(good,torch.full_like(dose,factor),dose);accepted|=good
            output[0,:T,7:13]=result;fractions.append(float(dose.mean()))
        if teacher is not None:output[:,:,channels]=teacher[step]
        predictions.append((basis.T@before).detach().cpu().numpy())
        after=matrix((output[0,:T,7:13]-B)/S)
        guided.append((basis.T@after).detach().cpu().numpy())
        lower.append(output[:,:,channels].detach().cpu().numpy());step+=1
        if step==50:
            np.savez_compressed(out/"state_trace.npz",predicted_root=np.asarray(predictions),guided_root=np.asarray(guided),
                source_root=(basis.T@target).detach().cpu().numpy(),lower=np.asarray(lower))
            write(out/"r9_condition.json",dict(relative_root=relative,lower_frozen_to_T0=teacher is not None,relative_dose_mean=float(np.mean(fractions)) if fractions else None,
                steps=step,source_root_translation=False,source_contact=False,source_fingers=False,neural_weight_updates=0,trace_instrumentation_changes_predictions=False))
        return output
    return guide
