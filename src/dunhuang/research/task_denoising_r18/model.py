"""Matched parameter counts; explicit exclusive-key mask; disentangled interventions."""
import math
import torch
from dunhuang.research.global_clock_r17.model import SharedState, Predictor as BasePredictor

def memory_support(clock, mode):
    if clock.shape != (2,150) or mode not in ("self","full","unique"):
        raise ValueError("Two local views and a registered memory mode required")
    index = torch.tensor([0,1] if mode=="self" else [1,0], device=clock.device)
    positions=clock[index]
    allowed=torch.ones_like(positions,dtype=torch.bool)
    if mode=="unique":
        allowed=(positions<clock.min(-1).values[:,None]) | (positions>clock.max(-1).values[:,None])
        if not (allowed.sum(-1)==75).all():raise ValueError("Exactly75 exclusive keys per view")
    return index, positions, allowed

def intervene(state, positions, control):
    """State slices: motion147, mask147, sin/cos2, event/phase2."""
    if control not in ("correct","content","clock","phase","key_order","removed"):
        raise ValueError("Unknown state control")
    memory=state.clone()
    if control=="content":memory[:,:,:147]=state[:,:,:147].flip(1)
    if control=="clock":memory[:,:,294:296]=state[:,:,294:296].flip(1)
    if control=="phase":memory[:,:,296:298]=state[:,:,296:298].flip(1)
    if control=="key_order":memory=memory.flip(1);positions=positions.flip(1)
    return memory,positions

class State(SharedState):
    def __init__(self,mode,width=64):
        super().__init__("self" if mode=="self" else "cross",width)
        if mode not in ("self","full","unique"):raise ValueError("State mode")
        self.memory_mode=mode;self.last={}
    def forward(self,hidden):
        if self.context is None:raise RuntimeError("State context absent")
        if self.control=="removed":self.last={};return hidden
        state,clock=self.context
        index,positions,allowed=memory_support(clock,self.memory_mode)
        state=state[index]
        # Content perturbations stay INSIDE the admitted support: never pull overlap
        # motion into a unique-only key through a sequence-wide permutation.
        admitted=state.clone()
        if self.control in ("content","clock","phase"):
            a,b={"content":(0,147),"clock":(294,296),"phase":(296,298)}[self.control]
            for view in range(2):
                ids=allowed[view].nonzero().flatten()
                admitted[view,ids,a:b]=state[view,ids.flip(0),a:b]
            state=admitted;control="correct"
        else:control=self.control
        state,positions=intervene(state,positions,control)
        if control=="key_order":allowed=allowed.flip(1)
        memory=self.encoder(state)
        score=self.q(hidden)@self.k(memory).transpose(-1,-2)/math.sqrt(memory.shape[-1])
        score=score-abs(clock[:,:,None]-positions[:,None,:])/75.
        attention=score.masked_fill(~allowed[:,None,:],-torch.inf).softmax(-1)
        overlap=(positions>=75)&(positions<150)
        self.last=dict(overlap_mass=float((attention*overlap[:,None,:]).sum(-1).mean().detach()),
            unique_mass=float((attention*(~overlap)[:,None,:]).sum(-1).mean().detach()),
            admitted_keys=allowed.sum(-1).detach().cpu().tolist(),
            entropy=float(-(attention*attention.clamp_min(1e-20).log()).sum(-1).mean().detach()))
        return hidden+self.out(attention@self.v(memory))

def observation_mismatch(value,known):
    """Move complete observed Root/rotation blocks; preserve masks and SO(3) blocks."""
    out=value.clone()
    for begin,end in [(4,7)]+[(7+6*j,13+6*j) for j in range(24)]:
        ids=known[0,:,begin:end].all(-1).nonzero().flatten()
        if len(ids)>1:out[0,ids,begin:end]=value[0,ids.flip(0),begin:end]
    return out

class Predictor(BasePredictor):
    def __call__(self,x,music,value,known,meta,t,control="correct"):
        if control=="observation":
            value=observation_mismatch(value,known);control="correct"
        return super().__call__(x,music,value,known,meta,t,control)
