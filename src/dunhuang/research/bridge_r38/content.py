"""Ordered music/motion tokens. Clock/duration channels are excluded from content."""
import math
import torch
from torch import nn
from dunhuang.research.transition_r37.content import relations,multi_positive_loss

def position(n,width,device):
    t=torch.arange(n,device=device,dtype=torch.float32)[:,None];w=torch.exp(torch.arange(0,width,2,device=device)*(-math.log(10000.)/width));p=torch.zeros(n,width,device=device);p[:,0::2]=torch.sin(t*w);p[:,1::2]=torch.cos(t*w);return p

class OrderedRetrieval(nn.Module):
    def __init__(self,width=96):
        super().__init__();self.motion=nn.Sequential(nn.Linear(548,width),nn.GELU(),nn.Linear(width,width));self.music=nn.Sequential(nn.Linear(50,width),nn.GELU(),nn.Linear(width,width))
        self.motion_time=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,batch_first=True),2)
        self.music_time=nn.ModuleList([nn.Conv1d(width,width,5,padding=2),nn.Conv1d(width,width,5,padding=4,dilation=2)])
        self.pool_m=nn.Linear(width,1);self.pool_d=nn.Linear(width,1);self.reconstruction=nn.Linear(width,198)
    def encode(self,music,x,mask,valid):
        mask=mask.clone();mask[:,:,-2:]=False;x=torch.where(mask,x,torch.zeros_like(x));support=mask.any(-1)
        if (~support.any(-1)).any() or (~valid.any(-1)).any():raise ValueError('No observed tokens')
        p=position(x.shape[1],self.pool_m.in_features,x.device)[None]
        d=self.motion_time(self.motion(torch.cat([x,mask.to(x)],-1))+p,src_key_padding_mask=~support)
        m=(self.music(music)+p)*valid[...,None]
        for conv in self.music_time:m=(m+torch.nn.functional.gelu(conv(m.transpose(1,2)).transpose(1,2)))*valid[...,None]
        def pool(v,head,ok):
            w=head(v).squeeze(-1).masked_fill(~ok,-torch.inf).softmax(1);return torch.nn.functional.normalize((v*w[...,None]).sum(1),dim=-1)
        return pool(m,self.pool_m,valid),pool(d,self.pool_d,support),torch.nn.functional.normalize(m,dim=-1),torch.nn.functional.normalize(d,dim=-1),self.reconstruction(d)

def training_groups(rows):
    from collections import defaultdict
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if r['role']=='train' and r['paired_clock_verified']:groups[(r['audio_sha256'],r['frames'],tuple(r['audio_frame_interval']))].append(i)
    groups={k:v for k,v in groups.items() if len({rows[i]['raw_motion_sha256'] for i in v})>=2};pairs={}
    for k,v in groups.items():
        other=[q for q in groups if q[0]==k[0] and q[1]==k[1] and max(q[2][0],k[2][0])>=min(q[2][1],k[2][1])]
        if other:pairs[k]=other
    if len({k[0] for k in pairs})<2:raise ValueError('Insufficient genuine cross-recording positives plus same-song disjoint times')
    return groups,pairs

def schedule(rows,seed,steps=80):
    import numpy as np
    groups,pairs=training_groups(rows);rng=np.random.default_rng(seed);audios=sorted({k[0] for k in pairs});result=[]
    for step in range(steps):
        ids=[]
        for audio in rng.choice(audios,2,replace=False):
            opts=[k for k in pairs if k[0]==audio];k=opts[int(rng.integers(len(opts)))];q=pairs[k][int(rng.integers(len(pairs[k])))]
            for key in (k,q):
                pool=groups[key];i=int(rng.choice(pool));others=[j for j in pool if rows[j]['raw_motion_sha256']!=rows[i]['raw_motion_sha256']];ids.extend([i,int(rng.choice(others))])
        p,n=relations([rows[i] for i in ids])
        if p.sum(1).min()<2 or not n.any(1).all():raise ValueError('Actual multiple positives required per query')
        result.append(ids)
    return result
