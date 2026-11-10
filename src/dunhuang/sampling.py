"""Joint overlap/anchor DDIM. The frozen model and upstream code are unchanged.
Conditioning operates at EVERY reverse step; there is no post-hoc pose writer.
Raw denoiser error is retained so exact constraints cannot imply model fidelity.
"""
import numpy as np

def consensus(x):
    # With 50% overlap every global frame appears in at most two windows.
    if len(x)>1:
        shared = (x[:-1,75:].clone()+x[1:,:75].clone())/2
        x[:-1,75:] = shared
        x[1:,:75] = shared
    return x

def constrained_ddim(diffusion, cond, values, masks):
    import torch
    shape = tuple(values.shape)
    if tuple(cond.shape) != (shape[0],150,4800) or shape[1:] != (150,151) or masks.shape != values.shape or masks.dtype != torch.bool:
        raise ValueError('Sampler contract mismatch')
    if len(values)>1 and (not torch.equal(masks[:-1,75:],masks[1:,:75]) or not torch.equal(values[:-1,75:][masks[:-1,75:]],values[1:,:75][masks[1:,:75]])):
        raise ValueError('Inconsistent global anchor mapping')
    device = diffusion.betas.device
    values, masks, cond = values.to(device), masks.to(device), cond.to(device)
    x = consensus(torch.randn(shape, device=device))
    times = list(reversed(torch.linspace(-1,diffusion.n_timestep-1,51).int().tolist()))
    weights = np.clip(np.linspace(0,diffusion.guidance_weight*2,50),None,diffusion.guidance_weight)
    residuals=[]
    with torch.no_grad():
        for step,(t,next_t) in enumerate(zip(times[:-1],times[1:])):
            ts = torch.full((len(x),),t,device=device,dtype=torch.long)
            weight = diffusion.guidance_weight if len(x)==1 else float(weights[step])
            _, predicted, *_ = diffusion.model_predictions(x,cond,ts,weight=weight,clip_x_start=diffusion.clip_denoised)
            residuals.append(float((predicted[masks]-values[masks]).abs().max()) if masks.any() else 0.)
            clean = torch.where(masks,values,consensus(predicted))
            if next_t < 0:
                x = clean
                continue
            alpha = diffusion.alphas_cumprod[t]; next_alpha = diffusion.alphas_cumprod[next_t]
            predicted_noise = (x-alpha.sqrt()*clean)/(1-alpha).sqrt()
            sigma = ((1-alpha/next_alpha)*(1-next_alpha)/(1-alpha)).sqrt()
            noise = consensus(torch.randn_like(x))
            x = consensus(next_alpha.sqrt()*clean+(1-next_alpha-sigma**2).clamp(min=0).sqrt()*predicted_noise+sigma*noise)
            noisy_known = next_alpha.sqrt()*values+(1-next_alpha).sqrt()*noise
            x = torch.where(masks,noisy_known,x)
    return x, dict(steps=50,raw_denoiser_anchor_max_abs_by_step=residuals,
        final_normalized_anchor_max_abs=float((x[masks]-values[masks]).abs().max()) if masks.any() else None,
        overlap_max_abs=float((x[:-1,75:]-x[1:,:75]).abs().max()) if len(x)>1 else 0.,
        post_generation_pose_paste=False)
