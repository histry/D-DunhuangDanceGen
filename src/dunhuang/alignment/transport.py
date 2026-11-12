"""Clock-banded partial matching; correspondence is never a replacement time warp."""
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA='clock_null_transport_v1'


def sinkhorn(log_kernel,mu,nu,iterations):
    if iterations<1 or torch.any(mu<0) or torch.any(nu<0):
        raise ValueError('Positive solver budget and nonnegative marginals required')
    if not torch.allclose(mu.sum(-1),nu.sum(-1),atol=1e-6):
        raise ValueError('Transport mass differs')
    lm=mu.clamp_min(1e-12).log();ln=nu.clamp_min(1e-12).log()
    v=torch.zeros_like(nu)
    for _ in range(iterations):
        u=lm-torch.logsumexp(log_kernel+v[:,None,:],dim=-1)
        v=ln-torch.logsumexp(log_kernel+u[:,:,None],dim=-2)
    return (log_kernel+u[:,:,None]+v[:,None,:]).exp()


class ClockNullTransport(nn.Module):
    def __init__(self,width=128,band_seconds=.5,epsilon=.15,iterations=80):
        super().__init__()
        if band_seconds<=0 or epsilon<=0:raise ValueError('Physical bandwidth and entropy required')
        self.music=nn.Linear(width,width,bias=False);self.motion=nn.Linear(width,width,bias=False)
        nn.init.eye_(self.music.weight);nn.init.eye_(self.motion.weight)
        self.null_logit=nn.Parameter(torch.tensor(0.))
        self.band_seconds=band_seconds;self.epsilon=epsilon;self.iterations=iterations

    def forward(self,music,motion,mv,dv,mt,dt):
        b,n,width=music.shape;k=motion.shape[1]
        if motion.shape!=(b,k,width) or mv.shape!=(b,n) or dv.shape!=(b,k):
            raise ValueError('Token and mask layouts differ')
        if mv.dtype!=torch.bool or dv.dtype!=torch.bool or not mv.any(-1).all() or not dv.any(-1).all():
            raise ValueError('Nonempty observed token masks required')
        if mt.ndim==1:mt=mt.expand(b,-1)
        if dt.ndim==1:dt=dt.expand(b,-1)
        if mt.shape!=mv.shape or dt.shape!=dv.shape or not torch.isfinite(mt).all() or not torch.isfinite(dt).all():
            raise ValueError('Explicit finite physical clocks required')
        if torch.any(torch.diff(mt,dim=-1)<=0) or torch.any(torch.diff(dt,dim=-1)<=0):
            raise ValueError('Candidate clocks must remain strictly increasing')
        if not torch.isfinite(music[mv]).all() or not torch.isfinite(motion[dv]).all():
            raise ValueError('Nonfinite observed tokens')
        m=F.normalize(self.music(torch.where(mv[...,None],music,0)),dim=-1)
        d=F.normalize(self.motion(torch.where(dv[...,None],motion,0)),dim=-1)
        similarity=m@d.transpose(-1,-2)
        distance=(mt[:,:,None]-dt[:,None,:]).abs()
        allowed=(distance<=self.band_seconds+1e-6)&mv[:,:,None]&dv[:,None,:]
        cost=1-similarity+.2*distance/self.band_seconds
        null_cost=.05+.9*self.null_logit.sigmoid()
        real=(-cost/self.epsilon).masked_fill(~allowed,-1e4)
        top=torch.cat((real,(-null_cost/self.epsilon).expand(b,n,1)),dim=-1)
        bottom=torch.cat(((-null_cost/self.epsilon).expand(b,1,k),torch.zeros(b,1,1,device=m.device)),dim=-1)
        kernel=torch.cat((top,bottom),dim=-2)
        nm=mv.sum(-1);nd=dv.sum(-1);total=nm+nd
        mu=torch.cat((mv.to(m.dtype),nd[:,None]),dim=-1)/total[:,None]
        nu=torch.cat((dv.to(m.dtype),nm[:,None]),dim=-1)/total[:,None]
        plan=sinkhorn(kernel,mu,nu,self.iterations)
        p=plan[:,:n,:k]*allowed
        capacity=torch.minimum(nm,nd)/total
        mass=p.sum((-1,-2));coverage=(mass/capacity).clamp(0,1)
        row_mass=p.sum(-1)
        expected=(p*dt[:,None,:]).sum(-1)/row_mass.clamp_min(1e-12)
        reliable=mv&(row_mass>1e-5)
        adjacent=reliable[:,1:]&reliable[:,:-1]
        reversed_step=F.relu(expected[:,:-1]-expected[:,1:])
        monotonic=(reversed_step*adjacent).sum(-1)/adjacent.sum(-1).clamp_min(1)
        # Missing matches are penalized, never called perfect alignment.
        score=coverage*(p*(similarity-.2*distance/self.band_seconds)).sum((-1,-2))/mass.clamp_min(1e-12)-(1-coverage)
        residual=torch.maximum((plan.sum(-1)-mu).abs().amax(-1),(plan.sum(-2)-nu).abs().amax(-1))
        return dict(score=score,plan=p,music=m,motion=d,coverage=coverage,
            music_null_fraction=(1-mass/(nm/total)).clamp(0,1),motion_null_fraction=(1-mass/(nd/total)).clamp(0,1),
            expected_motion_seconds=expected,row_matched_mass=row_mass,
            monotonic_penalty=monotonic,monotonic_violations=((reversed_step>1e-4)&adjacent).sum(-1),
            marginal_residual=residual,band_leak_mass=(p*~allowed).sum((-1,-2)),
            solver_plan=plan,mu=mu,nu=nu,allowed=allowed,music_valid=mv,motion_valid=dv)


def position_contrastive(result,groups,temperature=.07):
    """Stop-gradient soft positions; only unrelated known source groups are negatives."""
    m,d=result['music'],result['motion'];b,n,width=m.shape;k=d.shape[1]
    p=result['plan'].detach();positive=p/p.sum(-1,keepdim=True).clamp_min(1e-12)
    if len(groups)!=b:raise ValueError('Source associations required for negatives')
    different=torch.tensor([[a!=z for z in groups] for a in groups],device=m.device)
    logits=torch.einsum('bid,cjd->bicj',m,d)/temperature
    own=torch.eye(b,device=m.device,dtype=torch.bool)[:,None,:,None]
    allowed=different[:,None,:,None].expand(b,n,b,k)|(
        own&result['allowed'][:,:,None,:])
    allowed=allowed&result['motion_valid'][None,None,:,:]
    flat=logits.masked_fill(~allowed,-1e4).reshape(b,n,b*k)
    log_probability=F.log_softmax(flat,dim=-1).reshape(b,n,b,k)
    same=torch.stack([log_probability[i,:,i,:] for i in range(b)])
    valid=(p.sum(-1)>1e-5)&different.any(-1)[:,None]
    per_row=-(positive*same).sum(-1)
    return (per_row*valid).sum()/valid.sum().clamp_min(1)
