"""Train-only RMS units; no new trainable parameters and no target/output clipping."""
import torch
from .adapter import ContentContextAdapter

class BalancedContextAdapter(ContentContextAdapter):
    def __init__(self,scale,bias,inv_rms):
        super().__init__(scale,bias)
        inv=torch.as_tensor(inv_rms,dtype=torch.float32)
        if inv.shape!=(606,) or not torch.isfinite(inv).all() or not (inv>0).all():raise ValueError("Finite positive frozen feature units required")
        self.register_buffer("feature_inv_rms",inv)
    def features(self,value,known,t):
        f,support=super().features(value,known,t)
        return f*self.feature_inv_rms,support

def make(payload,scale,bias):
    if payload["schema"]=="r10_content_context_v1":return ContentContextAdapter(scale,bias)
    if payload["schema"]=="r10_balanced_context_v1":return BalancedContextAdapter(scale,bias,payload["state"]["feature_inv_rms"])
    raise ValueError("Unknown R10 condition schema")
