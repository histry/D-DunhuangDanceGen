"""Source/context readability, supervised truth, and editing permission are different masks."""
from dataclasses import dataclass
import torch

@dataclass
class SourcePacket:
    domain:str
    values:dict
    source_uids:tuple
    usage:str="inference"
    def branch(self):
        if self.domain not in ("aist","change"):raise ValueError("Explicit catalog domain required; no filename/classifier inference")
        if self.usage not in ("inference","change_replay"):raise ValueError("Source usage")
        if self.usage=="change_replay":
            if self.domain!="aist":raise ValueError("Replay override is only registered AIST-to-change training")
            return "change"
        return "general" if self.domain=="aist" else "change"
    def validate(self):
        self.branch();x=self.values["features"];v=self.values["valid"]
        if x.ndim!=4 or x.shape[1:]!=(96,22,9) or v.shape!=x.shape[:-1] or v.dtype!=torch.bool:raise ValueError("Source96x22x9 and bool validity")
        if len(self.source_uids)!=len(x) or not all(self.source_uids):raise ValueError("Explicit source identities required")
        if not torch.isfinite(x[v]).all():raise ValueError("Observed source values finite")
        return self

@dataclass
class ContextPacket:
    values:torch.Tensor
    valid:torch.Tensor
    frame_index:torch.Tensor
    side:torch.Tensor
    provenance:torch.Tensor
    coordinate_uid:str
    value_contract:str="R31_CANONICAL_NORMALIZED_139"
    def validate(self):
        x=self.values
        if x.ndim!=3 or x.shape[-1]!=139 or self.valid.shape!=x.shape or self.valid.dtype!=torch.bool:raise ValueError("Context value/mask BxNcx139")
        shape=x.shape[:2]
        if any(y.shape!=shape or y.dtype!=torch.long for y in (self.frame_index,self.side,self.provenance)):raise ValueError("Explicit integer context metadata")
        if not self.coordinate_uid or self.value_contract!="R31_CANONICAL_NORMALIZED_139":raise ValueError("Shared window coordinates and canonical contract required")
        if ((self.frame_index<0)|(self.frame_index>255)).any() or ((self.side<0)|(self.side>1)).any() or ((self.provenance<0)|(self.provenance>2)).any():raise ValueError("Context time/side/provenance range")
        if ((self.side==0)&(self.frame_index>=64)).any() or ((self.side==1)&(self.frame_index<160)).any():raise ValueError("Left/right context must lie outside Core64:160")
        if not torch.isfinite(x[self.valid]).all() or (x[self.valid].abs()>1.00001).any():raise ValueError("Finite normalized observable context required")
        return self
    def encoded_input(self):
        self.validate();safe=torch.where(self.valid,self.values,torch.zeros_like(self.values))
        t=self.frame_index.to(safe.dtype)/255.;clock=torch.stack((t,1-t),-1)
        side=torch.nn.functional.one_hot(self.side,2).to(safe);provenance=torch.nn.functional.one_hot(self.provenance,3).to(safe)
        return torch.cat((safe,self.valid.to(safe),clock,side,provenance),-1)

@dataclass
class TargetSupport:
    values:torch.Tensor
    valid:torch.Tensor
    editable:torch.Tensor
    def validate(self):
        if self.values.ndim!=3 or self.values.shape[1:]!=(256,139) or self.valid.shape!=self.values.shape or self.editable.shape!=self.values.shape:raise ValueError("Full target/support/edit shapes")
        if self.valid.dtype!=torch.bool or self.editable.dtype!=torch.bool:raise ValueError("Truth and edit masks are separate booleans")
        if not torch.isfinite(self.values[self.valid]).all():raise ValueError("Observed targets finite")
        return self
    def complete_for_q(self):
        self.validate()
        if not self.valid.all():raise ValueError("Partial target cannot become full diffusion x0; unknowns must not be zero-filled")
        return self.values
    def teacher_preservation_mask(self,requested_change):
        self.validate()
        if requested_change.shape!=self.values.shape or requested_change.dtype!=torch.bool:raise ValueError("Explicit requested-change mask")
        return self.valid & self.editable & ~requested_change
