"""Common random numbers only at official sampler-state draws; math stays native."""
import hashlib
from contextlib import contextmanager
import torch

@contextmanager
def common_noise(seed,batch=6,frames=256,dim=139,device="cpu"):
    original_randn=torch.randn;original_like=torch.randn_like;receipt=dict(draws=0,sha256=None,all_rows_equal=True)
    generator=torch.Generator(device=device);generator.manual_seed(seed);digest=hashlib.sha256()
    shape=(batch,frames,dim)
    def draw(device,dtype):
        base=original_randn((1,frames,dim),device=device,dtype=dtype,generator=generator)
        digest.update(base.detach().cpu().numpy().tobytes());receipt["draws"]+=1
        return base.expand(batch,-1,-1).clone()
    def randn(*args,**kwargs):
        requested=tuple(args[0]) if len(args)==1 and isinstance(args[0],(tuple,list,torch.Size)) else tuple(args)
        if requested==shape:return draw(kwargs.get("device",generator.device),kwargs.get("dtype",torch.float32))
        return original_randn(*args,**kwargs)
    def like(x,*args,**kwargs):
        if tuple(x.shape)==shape:return draw(kwargs.get("device",x.device),kwargs.get("dtype",x.dtype))
        return original_like(x,*args,**kwargs)
    torch.randn=randn;torch.randn_like=like
    try:yield receipt
    finally:
        torch.randn=original_randn;torch.randn_like=original_like;receipt["sha256"]=digest.hexdigest()
