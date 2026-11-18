"""Frozen R55 contracts and the shared synthetic/real objective."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import torch
from torch.nn import functional as F
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash
from dunhuang.research.paired_event_r41.train import seeded,digest,save
ROOT=Path('outputs/clock_residual_r55_01')
CFG=read(Path(__file__).with_name('protocol.json'))

def fresh(p):
 p=Path(p)
 if p.exists():raise ValueError('Fresh R55 output required: '+str(p))
 p.mkdir(parents=True);return p

def binding():return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in ('.py','.json') and p.name not in ('view.py','__main__.py','release.py')}

def residual_scores(raw,base,h):return torch.as_tensor(h,device=raw.device,dtype=raw.dtype)+CFG['residual_weight']*(raw-base.detach())

def ranking_loss(raw,base,h,positive):
 p=torch.as_tensor(positive,device=raw.device,dtype=torch.bool)
 if raw.ndim!=1 or p.shape!=raw.shape or not p.any() or p.all():raise ValueError('Observed positive and negative candidates required')
 score=residual_scores(raw,base,h);margin=(score[p,None]-score[None,~p])/CFG['temperature']
 return F.softplus(-margin).mean()+CFG['residual_penalty']*(raw-base.detach()).square().mean()

def diagonal(mt,dt,mv,dv,stride=3):
 valid=mv[0,::stride][None]&dv[:,::stride]
 if not valid.any(-1).all():raise ValueError('No genuine joint local support')
 return (((mt[0,::stride][None]*dt[:,::stride]).sum(-1)*valid).sum(-1)/valid.sum(-1))

def clock_scores(mt,dt,mv,dv,offsets=None):
 offsets=CFG['offsets'] if offsets is None else offsets
 margin=max(map(abs,offsets))
 if mt.shape[1]<=2*margin:raise ValueError('No shared non-circular clock support')
 q=torch.arange(margin,mt.shape[1]-margin,device=mt.device)
 if not len(q):raise ValueError('No shared non-circular clock support')
 rows=[]
 for delta in offsets:
  valid=mv[:,q]&dv[:,q+int(delta)]
  if not valid.any(-1).all():raise ValueError('Clock observation absent')
  rows.append((((mt[:,q]*dt[:,q+int(delta)]).sum(-1)*valid).sum(-1)/valid.sum(-1)))
 return torch.stack(rows,-1)

def shifted_inputs(inputs,shift):
 m,d,dm,mv=inputs;n=m.shape[1];pad=max(map(abs,CFG['offsets']))
 if n<=2*pad+2*pad+12:raise ValueError('Continuous source context too short')
 return m[:,pad:n-pad],d[:,pad+shift:n-pad+shift],dm[:,pad+shift:n-pad+shift],mv[:,pad:n-pad]

def hit(scores,truth):
 x=np.asarray(scores);p=np.asarray(truth,bool);best=np.isclose(x,x.max(),atol=1e-7,rtol=0);return float(p[best].mean())

def partition(meta):
 groups=sorted(meta['groups']['train'],key=lambda g:hashlib.sha256((str(CFG['seed'])+g).encode()).hexdigest());cal=groups[:CFG['calibration_groups']];train=groups[CFG['calibration_groups']:]
 assert not set(train)&set(cal);return dict(optimization=train,calibration=cal,val=meta['groups']['val'])

def fingerprint():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
