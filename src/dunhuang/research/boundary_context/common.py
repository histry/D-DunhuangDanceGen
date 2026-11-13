import hashlib,sys
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,pinned
from dunhuang.contracts import MotionAdapter
from dunhuang.research.source_capacity.common import read

def load_model(assets):
    sys.path.insert(0,str(pinned(assets["edge_root"],assets["edge_commit"])))
    from dataset.preprocess import Normalizer
    from dataset.scaler import MinMaxScaler
    from model.model import DanceDecoder
    from model.diffusion import GaussianDiffusion
    from vis import SMPLSkeleton
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    payload=torch.load(checked(assets["checkpoint"]),map_location="cpu",weights_only=False);norm=payload["normalizer"]
    assert isinstance(norm,Normalizer) and isinstance(norm.scaler,MinMaxScaler)
    exported=np.load(checked(assets["scaler"]))
    assert np.array_equal(exported["scale"],norm.scaler.scale_.numpy()) and np.array_equal(exported["bias"],norm.scaler.min_.numpy())
    model=DanceDecoder(nfeats=151,seq_len=150,latent_dim=512,ff_size=1024,num_layers=8,num_heads=8,dropout=.1,cond_feature_dim=4800,activation=torch.nn.functional.gelu).cuda().eval()
    model.load_state_dict(payload["ema_state_dict"],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    diffusion=GaussianDiffusion(model,150,151,SMPLSkeleton("cuda"),schedule="cosine",n_timestep=1000,predict_epsilon=False,loss_type="l2",use_p2=False,cond_drop_prob=.25,guidance_weight=2).cuda().eval()
    return model,diffusion,norm

def digest(model):
    h=hashlib.sha256()
    for n,p in sorted(model.named_parameters()):h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def fixed_rows(rows,role,count,seed):
    groups={}
    for r in rows:
        if r["role"]==role:groups.setdefault(r["source_group"],[]).append(r)
    for group in groups:
        groups[group].sort(key=lambda r:hashlib.sha256((str(seed)+r["raw_motion"]["sha256"]+str(r["start_frame"])).encode()).hexdigest())
    result=[]
    for i in range(max(map(len,groups.values()))):
        for g in sorted(groups):
            if i<len(groups[g]):result.append(groups[g][i])
            if len(result)==count:return result
    return result

class Data:
    def __init__(self,scaler):self.cache={};self.scale=scaler["scale"];self.bias=scaler["bias"]
    def native(self,row):
        ref=row["source_native"]
        if ref["sha256"] not in self.cache:self.cache[ref["sha256"]]=np.load(checked(ref))
        a=row["start_frame"];return self.cache[ref["sha256"]][a:a+150].copy()
    def batch(self,rows):
        target=[];music=[]
        for row in rows:
            x=self.native(row);o=x[0,4:7].copy();o[1]=0.;target.append(MotionAdapter.to_edge(x,o)*self.scale+self.bias)
            music.append(np.load(checked(row["jukebox"])))
        return torch.tensor(np.asarray(target),device="cuda",dtype=torch.float32),torch.tensor(np.asarray(music),device="cuda",dtype=torch.float32)

def masks(target,a,b,joints=None):
    known=torch.zeros_like(target,dtype=torch.bool);known[:,a-8:a,4:]=True;known[:,b:b+8,4:]=True
    if joints is not None:
        for j,allowed in enumerate(joints):
            if not allowed:known[:,:,7+6*j:13+6*j]=False
    unknown=torch.zeros_like(known);unknown[:,a:b,4:]=True
    return known,unknown
