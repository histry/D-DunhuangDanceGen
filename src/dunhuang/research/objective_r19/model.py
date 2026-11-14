"""Add a type-preserving diagnostic without adding a trainable parameter."""
import torch
from dunhuang.research.task_denoising_r18.model import State as ParentState,memory_support

def typed_permutation(state,allowed):
    if state.shape!=(2,150,298) or allowed.shape!=(2,150):raise ValueError("Two-view support required")
    out=state.clone()
    for view in range(2):
        ids=allowed[view].nonzero().flatten()
        kinds=state[view,ids,147:294]
        _,inverse=torch.unique(kinds,dim=0,return_inverse=True)
        for group in inverse.unique():
            group_ids=ids[inverse==group]
            out[view,group_ids,:147]=state[view,group_ids.flip(0),:147]
    return out

class State(ParentState):
    def forward(self,hidden):
        if self.control!="typed_content":return super().forward(hidden)
        if self.context is None:raise RuntimeError("Context absent")
        state,clock=self.context
        index,positions,allowed=memory_support(clock,self.memory_mode)
        # Parent forward subsequently chooses the source view. Undo that choice
        # after permuting each receiving view's admitted same-type keys.
        selected=typed_permutation(state[index],allowed)
        inverse=torch.argsort(index);changed=selected[inverse]
        previous=self.context;control=self.control
        self.context=(changed,clock);self.control="correct"
        try:return super().forward(hidden)
        finally:self.context=previous;self.control=control
