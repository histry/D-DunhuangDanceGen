"""Frozen C2 diagnostic: shared hard gauge, genuinely independent soft-side masks."""
import torch
from dunhuang.research.condition_dependency.balanced import BalancedContextAdapter

def soft_control(value,known,wrong,arm):
    if arm not in ("A0","A1","A2","A3","A4","A5"):raise ValueError("Unknown side intervention")
    v=value.clone();k=known.clone()
    if arm in ("A1","A5"):k[:,90:98]=False
    if arm in ("A2","A5"):k[:,52:60]=False
    if arm=="A3":v[:,90:98]=wrong[:,90:98]
    if arm=="A4":v[:,52:60]=wrong[:,52:60]
    return v,k

class SideDiagnosticAdapter(BalancedContextAdapter):
    hard_anchor=None
    def features(self,value,known,t):
        if self.hard_anchor is None or self.hard_anchor.shape!=(len(value),1,2):raise ValueError("Explicit common hard-frame59 horizontal gauge required")
        if not torch.isfinite(self.hard_anchor).all():raise ValueError("Nonfinite coordinate gauge")
        # Sentinel admits legacy preprocessing only; every affected Root content/mask/derivative
        # column is replaced below using the actual soft mask. No sentinel is fed to the model.
        sentinel=known.clone();sentinel[:,59,4:7]=True
        f,_=super().features(value,sentinel,t)
        k=known[:,:,4:7];v=torch.where(k,value[:,:,4:7],torch.zeros_like(value[:,:,4:7]))
        v=v.clone();v[:,:,:2]-=self.hard_anchor;v=torch.where(k,v,torch.zeros_like(v))
        d1=torch.zeros_like(v);d2=torch.zeros_like(v)
        pair=k[:,1:]&k[:,:-1];triple=pair[:,1:]&pair[:,:-1]
        d1[:,1:]=((value[:,1:,4:7]-value[:,:-1,4:7])/self.scale[4:7])*30*pair
        d2[:,2:]=(d1[:,2:]-d1[:,1:-1])*30*triple
        for start,block in ((4,v),(155,k.to(value.dtype)),(306,d1),(457,d2)):
            f[:,:,start:start+3]=block*self.feature_inv_rms[start:start+3]
        return f,known.any((1,2))
