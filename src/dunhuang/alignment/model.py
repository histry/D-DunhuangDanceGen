import math
import torch
from torch import nn
from torch.nn import functional as F

class TemporalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.conv=nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.output=nn.Linear(width,width)
    def forward(self,x,valid):
        update=self.conv(self.norm(x).transpose(1,2)).transpose(1,2)
        return (x+self.output(F.gelu(update)))*valid[...,None]

class Encoder(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.input=nn.Linear(channels*2,128)
        self.tcn=nn.ModuleList(TemporalBlock(128,d) for d in (1,2,4,8))
        layer=nn.TransformerEncoderLayer(128,4,512,.1,activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,2)
        self.pool=nn.Linear(128,1)
        self.output=nn.Linear(128,256)
    def forward(self,x,mask):
        valid=mask.any(-1)
        if not valid.any(-1).all():raise ValueError('Empty observation')
        x=self.input(torch.cat((x*mask,mask.float()),-1))*valid[...,None]
        for block in self.tcn:x=block(x,valid)
        x=x[:,::2];valid=valid[:,::2]
        position=torch.arange(x.shape[1],device=x.device)[:,None]/15
        frequency=torch.exp(torch.arange(0,128,2,device=x.device)*(-math.log(10000.)/128))
        phase=position*frequency
        encoding=torch.zeros_like(x[0]);encoding[:,::2]=phase.sin();encoding[:,1::2]=phase.cos()
        x=self.transformer(x+encoding,src_key_padding_mask=~valid)
        logits=self.pool(x).squeeze(-1).masked_fill(~valid,-1e4)
        pooled=(logits.softmax(-1)[...,None]*x).sum(1)
        return F.normalize(self.output(pooled),dim=-1),x,valid

class RetrievalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.music=Encoder(50);self.motion=Encoder(74)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.07)))
        self.sync=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Linear(128,1))
        self.semantic=nn.Linear(256,8)
        self.preference=nn.Sequential(nn.Linear(512,64),nn.GELU(),nn.Linear(64,1))
    def sync_score(self,music,motion,valid):
        features=torch.cat((music,motion,music-motion,music*motion),-1)
        return (self.sync(features).squeeze(-1)*valid).sum(-1)/valid.sum(-1).clamp_min(1)

def contrastive(m,d,groups):
    same_group=torch.tensor([[a==b for b in groups] for a in groups],device=m.device)
    positive=torch.eye(len(groups),device=m.device,dtype=torch.bool)
    score=(m@d.T).masked_fill(same_group & ~positive,-1e4)
    def loss(s):
        return (torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(~positive,-1e4),1)).mean()
    return .5*(loss(score)+loss(score.T))
