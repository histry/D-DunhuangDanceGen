"""Pose-aware SDMR research interface; no weights or routing qualification implied."""
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import relative_fk,rotation_sample
from .clock import source_positions

def observations(rotations,J,parents,duration=None):
    R=np.asarray(rotations,float)
    if len(R)<2 or R.shape[1:]!=(22,3,3):raise ValueError('Real variable-length body observations')
    # Geometry is pelvis-heading invariant while explicit local orientations retain body configuration.
    P=relative_fk(R,J,parents);P=(R[:,:1].swapaxes(-1,-2)@(P-P[:,:1])[...,None])[...,0]
    v=np.empty((len(R),22,3));v[1:]=Rotation.from_matrix((R[:-1].swapaxes(-1,-2)@R[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30;v[0]=v[1]
    speed=np.linalg.norm(v,axis=-1);agg=np.zeros((len(R),8));agg[:,0]=speed.mean(-1);agg[:,1]=np.percentile(speed,90,axis=1);agg[:,4]=(speed**2).mean(-1)
    agg[:,6]=(speed[:,[3,6,9,12,13,14,15,16,17,18,19,20,21]]**2).mean(-1);agg[:,7]=(speed[:,[0,1,2,4,5,7,8,10,11]]**2).mean(-1)
    pose=R[...,:2,:].reshape(len(R),132);geometry=P.reshape(len(R),66);dynamic=np.c_[v.reshape(len(R),66),agg]
    values=np.c_[pose,geometry,dynamic,np.linspace(0,1,len(R)),np.full(len(R),(duration or len(R))/30)].astype(np.float32)
    mask=np.ones_like(values,bool);mask[:,198+np.array([68,69,71])]=False
    return values,mask

def mapped_observations(event,rotations,J,parents):
    ticks=np.arange(*event['music_interval']);q=source_positions(event,ticks)
    return observations(rotation_sample(rotations,q),J,parents,len(ticks)),q

def relations(rows,task='local',tolerance_frames=2):
    """Trusted audio+physical source clock only. Unknown/overlap ambiguity excluded, not converted to positives."""
    n=len(rows);positive=np.zeros((n,n),bool);negative=positive.copy()
    if task not in ('local','song'):raise ValueError('Separate retrieval tasks')
    for i,a in enumerate(rows):
        for j,b in enumerate(rows):
            if not a.get('paired_clock_verified') or not b.get('paired_clock_verified'):continue
            same=a['audio_sha256']==b['audio_sha256'];A=a['audio_frame_interval'];B=b['audio_frame_interval']
            if same and (task=='song' or max(abs(A[k]-B[k]) for k in (0,1))<=tolerance_frames):positive[i,j]=True
            elif not same or max(A[0],B[0])>=min(A[1],B[1]):negative[i,j]=True
    return positive,negative

def build_model(music_dim=50,width=96):
    import torch
    from torch import nn
    class ContentRetrieval(nn.Module):
        def __init__(self):
            super().__init__();self.motion=nn.Sequential(nn.Linear(548,width),nn.GELU(),nn.Linear(width,width));self.music=nn.Sequential(nn.Linear(music_dim,width),nn.GELU(),nn.Linear(width,width))
            self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,batch_first=True),2)
            self.reconstruction=nn.Linear(width,198)
        def motion_tokens(self,x,mask):
            if x.shape!=mask.shape or x.shape[-1]!=274:raise ValueError('274D values and channel mask')
            valid=mask.any(-1)
            if (~valid.any(-1)).any():raise ValueError('Cannot pool an unobserved sequence')
            tokens=self.temporal(self.motion(torch.cat([torch.where(mask,x,torch.zeros_like(x)),mask.to(x)],-1)),src_key_padding_mask=~valid)
            embedding=(tokens*valid[...,None]).sum(1)/valid.sum(1)[:,None]
            return torch.nn.functional.normalize(embedding,dim=-1),self.reconstruction(tokens)
        def forward(self,music,x,mask,music_valid):
            motion,reconstructed=self.motion_tokens(x,mask);m=self.music(music);m=(m*music_valid[...,None]).sum(1)/music_valid.sum(1).clamp_min(1)[:,None]
            return torch.nn.functional.normalize(m,dim=-1)@motion.T,reconstructed
    return ContentRetrieval()

def multi_positive_loss(scores,positive,negative,temperature=.1):
    import torch
    p=torch.as_tensor(positive,device=scores.device,dtype=torch.bool);n=torch.as_tensor(negative,device=scores.device,dtype=torch.bool)
    if scores.shape!=p.shape or (p&n).any():raise ValueError('Disjoint explicit relations')
    good=p.any(1)&n.any(1)
    if not good.any():raise ValueError('No trusted positive/negative competition; do not train')
    s=scores/temperature
    return (torch.logsumexp(s.masked_fill(~(p|n),-torch.inf),1)[good]-torch.logsumexp(s.masked_fill(~p,-torch.inf),1)[good]).mean()
