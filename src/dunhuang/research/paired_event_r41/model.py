"""Ordered global representation and a separate, translation-equivariant local content path."""
import torch
from torch import nn
from torch.nn import functional as F
from dunhuang.research.bridge_r38.content import OrderedRetrieval

RADIUS=6

class EventContent(OrderedRetrieval):
    def __init__(self,width=96):
        super().__init__(width)
        def branch():return nn.Sequential(nn.Conv1d(width,width,5),nn.GELU(),nn.Conv1d(width,width,5,dilation=2),nn.GELU())
        self.local_music=branch();self.local_motion=branch();self.local_music_norm=nn.LayerNorm(width);self.local_motion_norm=nn.LayerNorm(width)
    def content(self,music,x,mask,valid):
        if x.shape!=mask.shape or x.shape[-1]!=274 or music.shape[:2]!=x.shape[:2] or len(x.shape)!=3 or x.shape[1]<=2*RADIUS:raise ValueError('Real common-time features with valid-convolution support')
        mask=mask.clone();mask[:,:,-2:]=False;x=torch.where(mask,x,torch.zeros_like(x));m=self.music(music);d=self.motion(torch.cat([x,mask.to(x)],-1))
        m=self.local_music_norm(self.local_music(m.transpose(1,2)).transpose(1,2));d=self.local_motion_norm(self.local_motion(d.transpose(1,2)).transpose(1,2))
        mv=valid.unfold(1,13,1).all(-1);dv=mask.any(-1).unfold(1,13,1).all(-1)
        return F.normalize(m,dim=-1),F.normalize(d,dim=-1),mv,dv


def scores_for_offsets(mt,dt,offsets=range(-15,16),radius=RADIUS):
    q=torch.arange(30,60,device=mt.device)-radius
    if q.min()<0 or q.max()>=mt.shape[1]:raise ValueError('Shared genuine query support')
    rows=[]
    for delta in offsets:
        d=q+int(delta)
        if d.min()<0 or d.max()>=dt.shape[1]:raise ValueError('Offset has no common valid-convolution support')
        rows.append((mt[:,q]*dt[:,d]).sum(-1).mean(-1))
    return torch.stack(rows,-1)


def local_objective(mt,dt,origins,temperature=.1):
    origins=torch.as_tensor(origins,device=mt.device,dtype=torch.long);delta=origins[:,0]-origins[:,1]
    if (abs(delta)>15).any():raise ValueError('Registered offset support')
    q=torch.arange(30,60,device=mt.device)
    logits=mt[:,q-RADIUS]@dt.transpose(-1,-2)/temperature;truth=q[None]+delta[:,None]-RADIUS
    token=F.cross_entropy(logits.reshape(-1,dt.shape[1]),truth.reshape(-1));correlation=scores_for_offsets(mt,dt);clock=F.cross_entropy(correlation/temperature,delta+15)
    expected=(correlation.div(temperature).softmax(-1)*torch.arange(-15,16,device=mt.device)).sum(-1)
    return clock,token,expected
