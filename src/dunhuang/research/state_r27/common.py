"""Pinned C1, matched complete targets, physical metrics and declared proxy scopes."""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from dunhuang.io import sha,write,checked
from dunhuang.contracts import matrices
from dunhuang.research.source_capacity.common import read,fresh,commit,forbid_optimizers,same_arrays
from dunhuang.research.reference_r26.common import packet
from dunhuang.research.reference_r26.geometry import fk22,relations
from dunhuang.research.lodge_adapter_r25.metrics import motion_metrics
from dunhuang.research.branch_r22.lodge import load
from .model import ReferenceModel,digest

def runtime(cfg,capacity=False,start=None):
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    d,n,s,f,e,info,_=load(read("src/dunhuang/research/branch_r22/config.json"))
    torch.manual_seed(cfg["seed"]);torch.cuda.manual_seed_all(cfg["seed"])
    adapter=ReferenceModel(capacity).cuda()
    if start is None:
        path=Path(cfg["r26"])/"models/C1/adapter.pt"
        assert sha(path)==read("reports/r26_release/r26_summary.json")["models"]["C1"]["checkpoint_hashes"]["adapter.pt"]
        value=torch.load(path,map_location="cpu",weights_only=True)["state"]
    else:
        value=torch.load(start,map_location="cpu",weights_only=True)["state"]
    result=adapter.load_state_dict(value,strict=not capacity)
    if capacity:assert set(result.missing_keys)<= {"lowrank.down.weight","lowrank.up.weight"} and not result.unexpected_keys
    adapter.attach(d.model)
    assert not any(p.requires_grad for p in d.model.parameters())
    return d,n,s,f,e,info,adapter

def checkpoint(adapter,path,cfg,arm,step):
    torch.save(dict(state={k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},config=cfg,arm=arm,step=step,capacity=adapter.capacity),path)

def restore(adapter,path):
    value=torch.load(path,map_location="cpu",weights_only=True);assert value["capacity"]==adapter.capacity
    adapter.load_state_dict(value["state"],strict=True)

def vector(a):return torch.cat([p.detach().flatten() for p in a.parameters()])
def gradient_norms(a):
    groups=dict(encoder=("parts",),input=("input",),sequence=("temporal",),projections=("out",),lowrank=("lowrank",))
    return {k:float(sum((p.grad.detach().square().sum() for name,p in a.named_parameters() if name.startswith(prefix) and p.grad is not None),torch.tensor(0.,device="cuda")).sqrt()) for k,prefix in groups.items()}

def readout_check(actual,expected):
    assert {k:v for k,v in actual.items() if k!="gradients"}=={k:v for k,v in expected.items() if k!="gradients"}
    result={}
    for k,x in actual["gradients"].items():
        y=expected["gradients"][k];assert math.isfinite(x) and math.isfinite(y) and min(x,y)>=0 and math.isclose(x,y,rel_tol=1e-5,abs_tol=0)
        result[k]=dict(actual=x,expected=y,relative_error=abs(x-y)/max(x,y) if x or y else 0.)
    return result

def metrics(raw,points,truth,target_points,mask=None):
    if mask is None:mask=np.r_[np.zeros(64,bool),np.ones(96,bool),np.zeros(96,bool)]
    pred=matrices(raw[:,7:].reshape(-1,22,6),row=True);target=matrices(truth[:,7:].reshape(-1,22,6),row=True)
    angles=Rotation.from_matrix((target.swapaxes(-1,-2)@pred).reshape(-1,3,3)).magnitude().reshape(-1,22)
    def geometry(p,R):
        vectors=np.stack((p[:,20]-p[:,16],p[:,21]-p[:,17],p[:,20]-p[:,21]),1)
        local=(R[:,0].swapaxes(-1,-2)[:,None]@vectors[:,:,:,None])[...,0]
        a=p[:,[16,17]]-p[:,[18,19]];b=p[:,[20,21]]-p[:,[18,19]]
        elbow=np.arccos(np.clip((a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-12),-1,1))
        return vectors,local,elbow
    world,local,elbow=geometry(points,pred);wtruth,ltruth,etruth=geometry(target_points,target)
    error=lambda x,y:float(np.sqrt(np.mean((x-y)[mask]**2)))
    source=target_points[mask]-target_points[mask,:1];generated=points[mask]-points[mask,:1]
    extent=lambda x:float(np.sqrt(np.mean(np.ptp(x[:,[16,17,18,19,20,21]],axis=0)**2)))
    activity=lambda x:float(np.linalg.norm(np.diff(x,axis=0)*30,axis=-1).mean())
    source_extent=extent(source);source_activity=activity(source)
    return dict(body22_rad=float(angles[mask].mean()),world_wrist_rmse_m=error(world[:,:2],wtruth[:,:2]),
        pelvis_wrist_rmse_m=error(local[:,:2],ltruth[:,:2]),double_wrist_rmse_m=error(world[:,2],wtruth[:,2]),
        elbow_angle_mae_rad=float(abs(elbow-etruth)[mask].mean()),arm_range_ratio=extent(generated)/source_extent if source_extent>1e-8 else None,
        body_activity_ratio=activity(generated)/source_activity if source_activity>1e-8 else None,
        layers={name:float(angles[mask][:,joints].mean()) for name,joints in dict(pelvis=[0],spine=[3,6,9],shoulder=[13,14,16,17],elbow=[18,19],wrist=[20,21]).items()})

def decoded_metrics(value,truth,n,s,f):
    with torch.no_grad():
        raw=n.unnormalize(value.clone());target=n.unnormalize(truth.clone())
        points=f(raw,s)[:,:,:22];target_points=f(target,s)[:,:,:22]
    return [metrics(x,p,y,q) for x,p,y,q in zip(raw.cpu().numpy(),points.cpu().numpy(),target.cpu().numpy(),target_points.cpu().numpy())]

def sources(cfg):
    root=Path(cfg["r26"]);assert sha(root/"artifacts_manifest.json")==cfg["r26_manifest_sha256"]
    manifest=read(root/"artifacts_manifest.json")["files"];pop=read(root/"data/result.json")
    for name in ["data/result.json","data/protocol.json"]:
        assert sha(root/name)==read("reports/r26_release/r26_receipt.json")["evidence"][name]
    cases=pop["train_probe"][:2];records=[next(r for r in pop["records"] if r["case"]==case) for case in cases];data={}
    for row in records:
        name="data/"+row["arrays"];assert sha(root/name)==manifest[name]["sha256"]
        data[row["case"]]=dict(np.load(root/name))
    return pop,records,data
