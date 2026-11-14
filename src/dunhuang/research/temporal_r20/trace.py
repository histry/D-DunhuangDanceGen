"""Observe unchanged CFG/clamp/decoder operations; a frozen blend intervention only."""
import torch
from dunhuang.research.global_clock_r17.model import merge

def blend(windows,policy="mean"):
    if windows.shape==(1,225,151):
        if policy!="mean":raise ValueError("No fusion intervention in joint layout")
        return windows
    if policy=="mean":return merge(windows)
    if policy!="quintic":raise ValueError("Unknown blend")
    u=torch.linspace(0,1,75,device=windows.device,dtype=windows.dtype)
    weight=(u**3*(10-15*u+6*u**2)).clamp(0,1)[:,None]
    middle=(1-weight)*windows[0,75:]+weight*windows[1,:75]
    return torch.cat((windows[0,:75],middle,windows[1,75:]),0)[None]

def stats(x):
    x=x.detach();finite=torch.isfinite(x);bad=float((~finite).float().mean())
    return dict(rms=float(x.square().mean().sqrt()) if finite.all() else None,nonfinite_fraction=bad,elements=x.numel())

class Telemetry:
    def __init__(self,w,head):
        self.enabled=False;self.grad_enabled=False;self.forward={};self.backward={};self.raw=[];self.handles=[]
        net=w.net
        modules=dict(direct=net.input_projection.base,observed=net.input_projection.observed,
            combined=net.input_projection,state_residual=head.out,
            decoder_first=net.seqTransDecoder.stack[0],decoder_last=net.seqTransDecoder.stack[-1],
            output=net.final_layer)
        for name,module in modules.items():
            self.handles.append(module.register_forward_hook(self.hook(name)))
    def hook(self,name):
        def call(module,args,out):
            if not self.enabled:return
            if not isinstance(out,torch.Tensor):raise TypeError("Expected tensor layer output")
            self.forward.setdefault(name,[]).append(stats(out))
            if name=="output":self.raw.append(out.detach())
            if self.grad_enabled and out.requires_grad:
                out.register_hook(lambda g:self.backward.setdefault(name,[]).append(stats(g)))
        return call
    def begin(self):
        self.forward={};self.backward={};self.raw=[]
    def close(self):
        for h in self.handles:h.remove()

class TracedPredictor:
    def __init__(self,base):
        self.base=base;self.wrapper=base.wrapper;self.adapter=base.adapter
        self.telemetry=Telemetry(self.wrapper,self.adapter);self.policy="mean";self.last={};self.calls=0
    def __call__(self,x,music,value,known,meta,t,control="correct"):
        self.calls+=1
        q=self.telemetry;q.begin()
        p,windows=self.base(x,music,value,known,meta,t,control)
        p=blend(windows,self.policy)
        if q.enabled:
            assert len(q.raw)==2,"Both CFG branches must be observed"
            unc,cond=q.raw;pre=unc+2*(cond-unc)
            assert torch.equal(pre.clamp(-1,1),windows.detach())
            sat=(abs(pre)>=1).float()
            self.last=dict(unc=unc,cond=cond,preclip=pre,clipped=windows.detach(),
                layer_forward=dict(q.forward),saturation=dict(all=float(sat.mean()),
                actually_changed=float((pre!=pre.clamp(-1,1)).float().mean()),
                root=float(sat[:,:,4:7].mean()),per_joint=sat[:,:,7:].reshape(*sat.shape[:2],24,6).mean((0,1,3)).cpu().tolist()))
        return p,windows
    def close(self):
        self.telemetry.close();self.base.close()
