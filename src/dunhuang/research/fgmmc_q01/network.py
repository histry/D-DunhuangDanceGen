"""Existing strict partial DP, restricted to a finite globally executable clock bank."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import CFG,array_hash
from dunhuang.alignment.model import TemporalBlock
from dunhuang.alignment.partial_path import soft_path_cost
from dunhuang.planning.anchor_reachability import multiple

def clocks(T):
 result=[];proof=[]
 for shift in CFG['clock_amplitudes']:
  knots=np.array([[0,0],[3,3],[(T-1)/2,(T-1)/2+shift],[T-4,T-4],[T-1,T-1]],float);rates=np.diff(knots[:,1])/np.diff(knots[:,0]);assert rates.min()>=.75 and rates.max()<=1.33
  cert=multiple([0,T],T,[],fixed_knots=knots.tolist());assert cert['rate_status']=='REACHABLE_BY_RATE';values=np.interp(np.arange(T),knots[:,0],knots[:,1]);result.append(values);proof.append(dict(amplitude=shift,clock_sha256=array_hash(values),certificate=cert))
 assert len({array_hash(x) for x in result})==len(result);return np.stack(result),proof

def canonical_bank(values):
 unique={array_hash(x):x for x in values};return np.stack([unique[k] for k in sorted(unique)])

class Branch(nn.Module):
 def __init__(self,channels):
  super().__init__();self.input=nn.Linear(channels*2,64);self.blocks=nn.ModuleList([TemporalBlock(64,d) for d in [1,2]]);self.out=nn.Linear(64,64)
 def forward(self,x,mask):
  valid=mask.any(-1);x=self.input(torch.cat([torch.where(mask,x,0),mask.to(x)],-1))*valid[...,None]
  for block in self.blocks:x=block(x,valid)
  # Equal fixed nonoverlapping bins, mask-weighted: same time coverage/count for all arms.
  bounds=np.linspace(0,x.shape[1],CFG['tokens']+1).round().astype(int);rows=[]
  for a,b in zip(bounds[:-1],bounds[1:]):
   support=valid[:,a:b];assert support.any(-1).all();rows.append((x[:,a:b]*support[...,None]).sum(1)/support.sum(1)[:,None])
  return F.normalize(self.out(torch.stack(rows,1)),dim=-1)

class Correspondence(nn.Module):
 def __init__(self):super().__init__();self.music=Branch(50);self.motion=Branch(274);self.priors={}
 def tokens(self,M,D,DM,MV):
  DM=DM.clone();DM[:,:,-2:]=False
  return self.music(M,MV[...,None].expand_as(M)),self.motion(D,DM)
 def score_tokens(self,m,d,T,constrained=True,uniform_only=False):
  values,_=clocks(T);values=values[:1] if uniform_only else canonical_bank(values);K=len(values) if constrained else 1;B=len(m);N=CFG['tokens'];centers=(np.linspace(0,T,N+1)[:-1]+np.linspace(0,T,N+1)[1:])/2-.5
  if constrained:
   mapped=np.stack([np.interp(centers,np.arange(T),v) for v in values]);allowed=np.abs(centers[None,:,None]-mapped[:,None,:])<=CFG['match_tolerance_frames']
  else:allowed=np.ones((1,N,N),bool)
  A=torch.tensor(allowed,device=m.device).repeat(B,1,1);cost=(1-m@d.transpose(-1,-2))[:,None].expand(B,K,N,N).reshape(B*K,N,N);valid=torch.ones((B*K,N),device=m.device,dtype=torch.bool);drops=torch.full((B*K,N),CFG['drop_cost'],device=m.device)
  total=soft_path_cost(cost,drops,drops,A,valid,valid,CFG['path_epsilon']);key=(T,constrained,uniform_only,str(m.device))
  if key not in self.priors:
   with torch.no_grad():self.priors[key]=soft_path_cost(torch.ones((K,N,N),device=m.device),drops[:K],drops[:K],A[:K],valid[:K],valid[:K],CFG['path_epsilon'])
  scores=(self.priors[key].repeat(B)-total).reshape(B,K)/(2*N)
  # Equal-prior mean over unique globally legal clocks; path multiplicity corrected above.
  return .1*(torch.logsumexp(scores/.1,1)-np.log(K))
 def forward(self,M,D,DM,MV,constrained=True,uniform_only=False):return self.score_tokens(*self.tokens(M,D,DM,MV),M.shape[1],constrained,uniform_only)

def initial():
 torch.manual_seed(CFG['seed']);return Correspondence()
def digest(model):return __import__('hashlib').sha256(''.join(k+array_hash(v.detach().cpu().numpy()) for k,v in sorted(model.state_dict().items())).encode()).hexdigest()
def tensors(batch,device='cuda'):return [torch.as_tensor(np.stack(v),device=device) for v in zip(*batch)]
