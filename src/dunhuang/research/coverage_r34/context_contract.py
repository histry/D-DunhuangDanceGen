"""Window-local context plus explicit global origin; observations never enter the requested gap."""
from dataclasses import dataclass
import torch

@dataclass
class WindowContext:
    values:torch.Tensor
    valid:torch.Tensor
    frame_index:torch.Tensor
    side:torch.Tensor
    provenance:torch.Tensor
    gap:torch.Tensor
    window_origin:torch.Tensor
    global_frame_index:torch.Tensor
    coordinate_uid:str
    value_contract:str="R31_CANONICAL_NORMALIZED_139"
    def validate(self):
        x=self.values;b,n=x.shape[:2]
        if x.ndim!=3 or x.shape[-1]!=139 or self.valid.shape!=x.shape or self.valid.dtype!=torch.bool:raise ValueError("BNC139 values and channel mask required")
        if any(v.shape!=(b,n) or v.dtype!=torch.long for v in (self.frame_index,self.side,self.provenance,self.global_frame_index)):raise ValueError("Context token metadata")
        if self.gap.shape!=(b,2) or self.gap.dtype!=torch.long or self.window_origin.shape!=(b,) or self.window_origin.dtype!=torch.long:raise ValueError("Explicit gap and global origin")
        if (self.gap[:,0]<8).any() or (self.gap[:,1]>248).any() or (self.gap[:,0]>=self.gap[:,1]).any():raise ValueError("Gap must leave context on both sides")
        if not torch.equal(self.global_frame_index,self.frame_index+self.window_origin[:,None]):raise ValueError("Local/global clock mismatch")
        if not self.coordinate_uid or self.value_contract!="R31_CANONICAL_NORMALIZED_139":raise ValueError("Explicit shared coordinate contract")
        if ((self.frame_index<0)|(self.frame_index>255)).any() or ((self.side<0)|(self.side>1)).any() or ((self.provenance<0)|(self.provenance>2)).any():raise ValueError("Context metadata range")
        if ((self.side==0)&(self.frame_index>=self.gap[:,0,None])).any() or ((self.side==1)&(self.frame_index<self.gap[:,1,None])).any():raise ValueError("Target gap leaked into observed context")
        if not torch.isfinite(x[self.valid]).all() or (x[self.valid].abs()>1.00001).any():raise ValueError("Finite observable normalized values required")
        return self
    def encoded_input(self):
        self.validate();x=torch.where(self.valid,self.values,torch.zeros_like(self.values));t=self.frame_index.to(x)/255
        return torch.cat((x,self.valid.to(x),torch.stack((t,1-t),-1),torch.nn.functional.one_hot(self.side,2).to(x),torch.nn.functional.one_hot(self.provenance,3).to(x)),-1)

def context(values,gaps,mode="both",seed=0,partial_body=False,window_origin=None):
    if values.ndim!=3 or values.shape[1:]!=(256,139):raise ValueError("Full target coordinate clock, not a partial fake x0")
    b=len(values);device=values.device;gaps=torch.as_tensor(gaps,dtype=torch.long,device=device)
    if gaps.shape==(2,):gaps=gaps[None].repeat(b,1)
    ids=torch.stack([torch.cat((torch.arange(int(a)-8,int(a),device=device),torch.arange(int(z),int(z)+8,device=device))) for a,z in gaps])
    side=torch.tensor([0]*8+[1]*8,device=device)[None].repeat(b,1);row=torch.arange(b,device=device)[:,None]
    x=values[row,ids].clone();valid=torch.isfinite(x)
    if partial_body:valid[:,:,:7]=False
    if mode=="no_left":valid[:,:8]=False
    elif mode=="no_right":valid[:,8:]=False
    elif mode=="partial":
        valid[:,8:,:7]=False
        generator=torch.Generator(device=device).manual_seed(seed)
        valid &= torch.rand(valid.shape,device=device,generator=generator)>.25
    elif mode=="wrong":
        # Valid alternate pose observations; keep missing world Root/C4 masked, no invented coordinate fit.
        x=x.roll(1,0);valid=valid.roll(1,0);valid[:,:,:7]=False
    elif mode!="both":raise ValueError("Context intervention")
    x=torch.where(valid,x,torch.full_like(x,float("nan")))
    origin=torch.zeros(b,dtype=torch.long,device=device) if window_origin is None else torch.as_tensor(window_origin,dtype=torch.long,device=device)
    packet=WindowContext(x,valid,ids,side,torch.zeros_like(side),gaps,origin,ids+origin[:,None],"SHARED_MODEL0_WINDOW")
    return packet.validate()
