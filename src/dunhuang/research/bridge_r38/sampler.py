"""Shared actual reverse process and differentiable projection used during training."""
import torch
import numpy as np
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle,axis_angle_to_matrix
from dunhuang.research.state_r27.trace import row_noise
from dunhuang.research.transition_r37.geometry import c2_envelope

def project(rt,proposal,anchor,budget):
    R=rotation_6d_to_matrix(anchor[:,:,7:].reshape(len(anchor),256,22,6));P=rotation_6d_to_matrix(proposal[:,:,7:].reshape(len(anchor),256,22,6))
    v=matrix_to_axis_angle(R.transpose(-1,-2)@P);limit=budget[None,:,None]
    v=v*torch.minimum(torch.ones_like(v[...,0]),limit/v.norm(dim=-1).clamp_min(1e-7))[...,None]
    corrected=R@axis_angle_to_matrix(v);out=proposal.clone();out[:,:,7:]=corrected[...,:2,:].reshape(len(anchor),256,132)
    # Source bodies are protected; Root outside the edit comes from its own anchor,
    # which is observed only for paired recordings and model-generated otherwise.
    editable=budget>0;out[:,~editable]=anchor[:,~editable]
    return out

def condition(rt,source,ctx,gap):
    rt.c.clear_all();rt.c.set_source(source);rt.c.set_context(ctx);old=rt.c.change.inject
    g=torch.tensor(1-c2_envelope(256,gap),device='cuda',dtype=torch.float32)[None,:,None]
    rt.c.change.inject=lambda i,h:h+(old(i,h)-h)*g
    return old

@torch.no_grad()
def sample_bridge(rt,music,source,ctx,anchor,cfg,seeds,cache=False):
    old=condition(rt,source,ctx,cfg['gap']);torch.manual_seed(27041);torch.cuda.manual_seed_all(27041)
    times=list(reversed(torch.linspace(-1,cfg['start_t'],51).int().tolist()));weights=np.clip(np.linspace(0,rt.d.guidance_weight*2,50),None,rt.d.guidance_weight)
    budget=torch.tensor(cfg['budget_rad']*c2_envelope(256,cfg['gap']),device='cuda',dtype=anchor.dtype);states=[]
    try:
        with row_noise(seeds,device='cuda') as noise:
            x=rt.d.q_sample(anchor,torch.full((len(anchor),),cfg['start_t'],device='cuda',dtype=torch.long),torch.randn_like(anchor))
            for i,(t,nxt,w) in enumerate(zip(times[:-1],times[1:],weights)):
                if cache and i in (0,10,25,40,49):states.append((t,x.cpu().numpy()))
                ts=torch.full((len(anchor),),t,device='cuda',dtype=torch.long)
                _,proposal,*_=rt.d.model_predictions(x,music,None,ts,weight=float(w),clip_x_start=rt.d.clip_denoised)
                x0=project(rt,proposal,anchor,budget)
                if nxt<0:x=x0;continue
                alpha=rt.d.alphas_cumprod[t];an=rt.d.alphas_cumprod[nxt];epsilon=(x-alpha.sqrt()*x0)/(1-alpha).sqrt()
                sigma=((1-alpha/an)*(1-an)/(1-alpha)).clamp_min(0).sqrt();x=x0*an.sqrt()+(1-an-sigma.square()).clamp_min(0).sqrt()*epsilon+sigma*torch.randn_like(x)
        assert noise['draws']==50
    finally:rt.c.change.inject=old;rt.c.clear_all()
    return x,noise,states
