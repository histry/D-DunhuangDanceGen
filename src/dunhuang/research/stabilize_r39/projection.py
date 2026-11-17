"""Physical-meter Root residual projection at each x0, separate from body radians."""
def continuous_root(proposal,anchor,envelope,scale,budget_m):
    import torch
    if proposal.shape!=anchor.shape or proposal.shape[-1]!=3 or envelope.shape!=(proposal.shape[1],) or scale.shape!=(3,):raise ValueError('Root and physical affine-scale contract')
    if budget_m<=0 or not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Positive meters and scale')
    delta=(proposal-anchor)/scale;factor=(float(budget_m)/delta.norm(dim=-1).clamp_min(1e-12)).clamp(max=1)
    result=anchor+envelope[None,:,None]*delta*factor[...,None]*scale
    return torch.where(envelope[None,:,None]>0,result,anchor)

class Projector:
    def __init__(self,original,mode,budget_m):self.original=original;self.mode=mode;self.budget_m=budget_m;self.trace=[]
    def __call__(self,rt,proposal,anchor,budget):
        import torch
        from dunhuang.research.transition_r37.geometry import c2_envelope
        out=self.original(rt,proposal,anchor,budget)
        if self.mode=='RootContinuous':
            scale=rt.n.legacy.scaler.scale_[4:7].to(out);envelope=torch.tensor(c2_envelope(256,[104,152]),device=out.device,dtype=out.dtype)
            out[:,:,4:7]=continuous_root(proposal[:,:,4:7],anchor[:,:,4:7],envelope,scale,self.budget_m)
        elif self.mode!='Raw':raise ValueError('Explicit projection arm')
        before=rt.n.unnormalize(proposal.clone())[:,:,4:7];after=rt.n.unnormalize(out.clone())[:,:,4:7]
        self.trace.append(torch.stack((before,after),1).detach().cpu().numpy());return out
