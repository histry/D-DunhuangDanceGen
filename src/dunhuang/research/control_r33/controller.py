"""One hook per layer; source and context both read the original hidden."""
from torch import nn

class Controller(nn.Module):
    layers=(1,4,7)
    def __init__(self,general,change,context):
        super().__init__();self.general=general;self.change=change;self.context_branch=context
        self.route=None;self.handles=[]
    def set_source(self,packet):
        packet.validate();self.clear_source();self.route=packet.branch()
        getattr(self,self.route).set_reference(**packet.values)
    def set_context(self,packet):self.context_branch.set_context(packet)
    def clear_source(self):
        self.general.clear();self.change.clear();self.route=None
    def clear_context(self):self.context_branch.clear_context()
    def clear_all(self):self.clear_source();self.clear_context()
    def inject(self,index,hidden):
        with_source=hidden if self.route is None else getattr(self,self.route).inject(index,hidden)
        if self.context_branch.packet is None:return with_source
        return with_source+self.context_branch.delta(index,hidden)
    def attach(self,model):
        if self.handles or len(model.seqTransDecoder.stack)!=9:raise ValueError("One pinned9-layer Lodge, attach once")
        for i,layer in enumerate(self.layers):
            self.handles.append(model.seqTransDecoder.stack[layer].register_forward_hook(lambda m,a,o,i=i:self.inject(i,o)))
    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.clear_all()
