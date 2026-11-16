"""Explicit source-initialized SO(3)-bounded diffusion editing, not a learned residual claim."""
import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix,matrix_to_axis_angle,axis_angle_to_matrix
from dunhuang.research.state_r27.trace import row_noise

@torch.no_grad()
def sample_edit(rt,music,reference,base,seeds,budget,start_t=119,source_packet=None,context_packet=None):
    if reference.shape!=(len(music),256,132) or base.shape!=(len(music),256,139):raise ValueError('Actual source body and model-generated full state required')
    if not torch.isfinite(reference).all() or not torch.isfinite(base).all():raise ValueError('Finite observed body/model proposal')
    if len(seeds)<2:raise ValueError('Match the official batch contract')
    rt.c.clear_all()
    torch.manual_seed(27041);torch.cuda.manual_seed_all(27041)
    if source_packet is not None:rt.c.set_source(source_packet)
    if context_packet is not None:rt.c.set_context(context_packet)
    reference=reference.to(base);R=rotation_6d_to_matrix(reference.reshape(len(base),256,22,6))
    limits=torch.as_tensor(budget,device=base.device,dtype=base.dtype)
    if limits.shape!=(256,):raise ValueError('Per-frame region budget required')
    initial=base.clone();initial[:,:,7:]=reference
    times=list(reversed(torch.linspace(-1,start_t,51).int().tolist()))
    if len(set(times))!=51:raise ValueError('Diffusion schedule must contain distinct steps')
    weights=np.clip(np.linspace(0,rt.d.guidance_weight*2,50),None,rt.d.guidance_weight);ledger=[]
    with row_noise(seeds,device=str(base.device)) as nr:
        eps=torch.randn_like(initial);ts=torch.full((len(base),),start_t,device=base.device,dtype=torch.long)
        x=rt.d.q_sample(initial,ts,eps)
        for time,nxt,weight in zip(times[:-1],times[1:],weights):
            t=torch.full((len(x),),time,device=x.device,dtype=torch.long)
            _,x0,*_=rt.d.model_predictions(x,music,None,t,weight=float(weight),clip_x_start=rt.d.clip_denoised)
            P=rotation_6d_to_matrix(x0[:,:,7:].reshape(len(x),256,22,6))
            rv=matrix_to_axis_angle(R.transpose(-1,-2)@P);size=rv.norm(dim=-1)
            scale=torch.minimum(torch.ones_like(size),limits[None,:,None]/size.clamp_min(1e-8))
            corrected=R@axis_angle_to_matrix(rv*scale[...,None])
            x0=x0.clone();x0[:,:,7:]=corrected[...,:2,:].reshape(len(x),256,132)
            ledger.append(dict(t=time,max_correction_rad=float((rv*scale[...,None]).norm(dim=-1).max())))
            if nxt<0:x=x0;continue
            alpha=rt.d.alphas_cumprod[time];an=rt.d.alphas_cumprod[nxt]
            pred_noise=(x-alpha.sqrt()*x0)/(1-alpha).sqrt()
            sigma=((1-alpha/an)*(1-an)/(1-alpha)).clamp_min(0).sqrt()
            coefficient=(1-an-sigma.square()).clamp_min(0).sqrt()
            x=x0*an.sqrt()+coefficient*pred_noise+sigma*torch.randn_like(x)
    rt.c.clear_all();assert nr['draws']==50
    return x,nr,dict(start_t=start_t,steps=50,eta=1,body_budget_rad=limits.cpu().tolist(),steps_observed=ledger,
        Root_C4_initialization='MODEL_PROPOSAL_NOT_SOURCE_TRUTH',Root_C4_updated_jointly_each_step=True,
        source_body_used_in_initial_state=True,projection='EACH_X0_ON_SO3; NOT_6D_ADDITION; NOT_POST_RENDER_SMOOTHING',new_training=False)
