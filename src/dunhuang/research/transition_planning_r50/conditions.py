"""Explicit phase propagation for the existing variable-reference attention."""
from dataclasses import dataclass
import torch
from dunhuang.research.coverage_r34.model import WindowReference
from dunhuang.research.lodge_adapter_r25.model import PARTS


@dataclass
class MusicReperformanceCondition:
    music_features: torch.Tensor
    reference_features: torch.Tensor
    source_positions: torch.Tensor
    reference_phase: torch.Tensor
    query_phase: torch.Tensor
    clock: torch.Tensor
    observed_mask: torch.Tensor
    edit_permissions: torch.Tensor
    left_context: torch.Tensor
    right_context: torch.Tensor

    def validate(self):
        batch,n,joints,features = self.reference_features.shape
        if joints!=22 or features!=9 or self.observed_mask.shape!=(batch,n,22):
            raise ValueError("Observed reference feature contract")
        if self.observed_mask.dtype!=torch.bool or self.edit_permissions.dtype!=torch.bool:
            raise ValueError("Explicit boolean validity and edit masks required")
        if self.music_features.shape[:2]!=(batch,256) or self.source_positions.shape!=(batch,256):
            raise ValueError("Music and target clock must share 256 frames")
        if self.reference_phase.shape!=(batch,n,8) or self.query_phase.shape!=(batch,256,8):
            raise ValueError("Phase support shape differs")
        if self.edit_permissions.shape!=(batch,256):
            raise ValueError("Explicit edit permission per query frame")
        if self.clock.shape!=(batch,n,2):
            raise ValueError("Source clock shape differs")
        if self.left_context.shape[0]!=batch or self.right_context.shape[0]!=batch:
            raise ValueError("Bilateral context batch differs")
        for tensor in (self.music_features,self.source_positions,self.clock,self.reference_phase,
                       self.query_phase,self.left_context,self.right_context):
            if not torch.isfinite(tensor).all():
                raise ValueError("Nonfinite observed condition")
        if not torch.isfinite(self.reference_features[self.observed_mask]).all():
            raise ValueError("Nonfinite valid reference")
        if (self.source_positions.diff(dim=1)<0).any() or self.source_positions.min()<0 or self.source_positions.max()>n-1:
            raise ValueError("Source time escapes continuous reference memory")


class PhaseReference(WindowReference):
    def inject(self,index,hidden):
        proposal = super().inject(index,hidden)
        if self.context is None:
            return hidden
        return torch.where(self.control_mask[:,:,None],proposal,hidden)

    def set_condition(self, condition):
        condition.validate()
        self.set_reference(condition.reference_features,condition.observed_mask,condition.clock,
                           condition.reference_phase,condition.query_phase,
                           condition.source_positions,condition.edit_permissions)

    def set_reference(self,features,valid,clock,ref_phase,query_phase,mapped_source=None,control_mask=None):
        super().set_reference(features,valid,clock,ref_phase,query_phase,mapped_source,control_mask)
        if self.legacy:
            raise ValueError("Use explicit variable-memory condition contract")
        safe = torch.where(valid[:,:,:,None],features,torch.zeros_like(features))
        inp = torch.cat((safe,valid[:,:,:,None].to(safe.dtype)),-1)
        parts = [layer(inp[:,:,p].flatten(2)) for layer,p in zip(self.parts,PARTS)]
        frame = valid.any(-1)
        present = frame.any(-1)
        phase = torch.where(frame[:,:,None],ref_phase,torch.zeros_like(ref_phase))
        memory = self.input(torch.cat(parts+[clock,phase],-1))*frame[:,:,None]
        for conv in self.temporal:
            memory = memory+torch.nn.functional.silu(conv(memory.transpose(1,2)).transpose(1,2))*frame[:,:,None]
        memory = self.norm(memory)
        global_token = (memory*frame[:,:,None]).sum(1)/frame.sum(1).clamp_min(1)[:,None]
        self.context = (torch.cat((global_token[:,None],memory),1),
                        torch.cat((torch.ones_like(present[:,None]),frame),1),
                        present,query_phase)


def diagnose():
    torch.manual_seed(50050)
    model = PhaseReference().eval()
    # Fixed nonzero probe readout reveals structural paths before training.
    with torch.no_grad():
        for layer in model.out:
            layer.weight.normal_(std=.01)
    batch,n = 1,128
    features = torch.randn(batch,n,22,9)
    valid = torch.ones(batch,n,22,dtype=torch.bool)
    rp = torch.randn(batch,n,8,requires_grad=True)
    qp = torch.randn(batch,256,8,requires_grad=True)
    packet = MusicReperformanceCondition(torch.randn(batch,256,35),features,
             torch.linspace(0,n-1,256)[None],rp,qp,torch.zeros(batch,n,2),valid,
             torch.ones(batch,256,dtype=torch.bool),torch.randn(batch,8,139),torch.randn(batch,8,139))
    hidden = torch.randn(batch,256,512)
    model.set_condition(packet)
    output = model.inject(0,hidden)
    gradient = torch.autograd.grad(output.square().mean(),(rp,qp))
    norms = [float(g.norm()) for g in gradient]
    with torch.no_grad():
        packet.reference_phase = torch.zeros_like(rp)
        packet.query_phase = torch.zeros_like(qp)
        model.set_condition(packet)
        zero = model.inject(0,hidden)
        delta = float((output-zero).square().mean().sqrt())
        valid[:,:,64:] = False
        model.set_condition(packet)
        masked = model.inject(0,hidden)
        features[:,:,64:] = float("nan")
        model.set_condition(packet)
        poisoned = model.inject(0,hidden)
        assert torch.equal(masked,poisoned)
        packet.edit_permissions[:,:8] = False
        packet.edit_permissions[:,-8:] = False
        model.set_condition(packet)
        protected = model.inject(0,hidden)
        assert torch.equal(protected[:,:8],hidden[:,:8])
        assert torch.equal(protected[:,-8:],hidden[:,-8:])
    assert min(norms)>0 and delta>0
    return dict(reference_and_query_phase_gradient_norm=norms,phase_zero_output_rms=delta,
                hidden_poison_invariant=True,probe="FIXED_RANDOM_WEIGHTS_NONZERO_READOUT",
                scope="STRUCTURAL_PROPAGATION",music_context_routing="EXISTING_BACKBONE_CONTROLLER",
                actual_frozen_model_quality_evaluated=False)
