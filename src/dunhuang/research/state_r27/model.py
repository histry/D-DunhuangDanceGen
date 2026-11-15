"""C1 is preserved. Conditional capacity control adds a zero-head gated rank8 output."""
import torch
from torch import nn
from dunhuang.research.reference_r26.model import Adapter,digest

class ReferenceModel(Adapter):
    def __init__(self,capacity=False):
        super().__init__(True,width=64,part_width=16,layers=(1,4,7),hidden=512)
        self.capacity=capacity
        if capacity:
            self.lowrank=nn.ModuleDict(dict(down=nn.Linear(512,8,bias=False),up=nn.Linear(8,139,bias=False)))
            nn.init.zeros_(self.lowrank["up"].weight)

    def attach(self,model):
        super().attach(model)
        if self.capacity:
            self.handles.append(model.final_layer.register_forward_hook(self.output_control))

    def output_control(self,module,args,output):
        if self.context is None:return output
        present=self.context[2]
        if not present.any():return output
        hidden=args[0];assert hidden.shape[-1]==512 and output.shape[1:]==(256,139)
        residual=self.lowrank["up"](self.lowrank["down"](hidden))/8.
        region=torch.zeros_like(present[:,None,None]).expand(-1,256,1).clone()
        region[:,64:160]=present[:,None,None]
        return torch.where(region,output+residual,output)
