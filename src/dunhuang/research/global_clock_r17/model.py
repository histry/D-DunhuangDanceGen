"""Two150-frame views + learned Root/Pose/mask/global-clock state exchange."""
import math
import torch
from torch import nn
from dunhuang.research.motion_bridge.model import rotations
from dunhuang.contracts import Q

def views(x):
    if x.shape[1:]!=(225,151) or len(x)!=1:raise ValueError("One225-frame pair required")
    return torch.cat((x[:,:150],x[:,75:]),0)

def merge(x):
    if x.shape!=(2,150,151):raise ValueError("Two150-frame predictions required")
    return torch.cat((x[0,:75],(x[0,75:]+x[1,:75])/2,x[1,75:]),0)[None]

class SharedState(nn.Module):
    def __init__(self,mode,width=64):
        super().__init__()
        if mode not in ("cross","self"):raise ValueError("Matched state mode")
        self.mode=mode;self.encoder=nn.Sequential(nn.Linear(298,width),nn.SiLU(),nn.Linear(width,width))
        self.q=nn.Linear(512,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width)
        self.out=nn.Linear(width,512);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.context=None;self.control="correct"
    def forward(self,hidden):
        if self.context is None:raise RuntimeError("Global state context missing")
        if self.control=="removed":return hidden
        state,clock=self.context
        memory=self.encoder(state)
        index=torch.tensor([1,0] if self.mode=="cross" else [0,1],device=hidden.device)
        memory=memory[index];positions=clock[index]
        if self.control=="reversed":memory=memory.flip(1)
        elif self.control!="correct":raise ValueError("Unknown state intervention")
        score=self.q(hidden)@self.k(memory).transpose(-1,-2)/math.sqrt(memory.shape[-1])
        score=score-abs(clock[:,:,None]-positions[:,None,:])/75.
        attention=score.softmax(-1)
        return hidden+self.out(attention@self.v(memory))

class Predictor:
    def __init__(self,wrapper,adapter=None):
        self.wrapper=wrapper;self.adapter=adapter;self.handle=None
        if adapter is not None:self.handle=wrapper.net.input_projection.register_forward_hook(lambda m,i,o:adapter(o))
    def __call__(self,x,music,value,known,meta,t,control="correct"):
        if x.shape!=(1,225,151) or music.shape!=(2,150,4800):raise ValueError("Global225 clock or local music shape differs")
        safe=torch.where(known,value,torch.zeros_like(value));effective=torch.where(known,safe,x)
        if self.adapter is not None:
            state=torch.cat((effective[:,:,4:],known[:,:,4:].to(x.dtype),meta),-1)
            state=torch.cat((state[:,:150],state[:,75:]),0)
            clock=torch.stack((torch.arange(150,device=x.device),torch.arange(75,225,device=x.device))).to(x.dtype)
            self.adapter.context=(state,clock);self.adapter.control=control
        try:
            p=self.wrapper(views(x),music,t.expand(2),views(safe),views(known))
            return merge(p),p
        finally:
            if self.adapter is not None:self.adapter.context=None
    def close(self):
        if self.handle is not None:self.handle.remove()

def fk(normalized,scale,bias,origin,parents,offsets):
    x=(normalized-bias)/scale;n=x.shape[1];R=rotations(x[:,:,7:].reshape(-1,n,24,6))
    q=x.new_tensor(Q);local=[q.T@R[:,:,0]]+[R[:,:,j] for j in range(1,24)]
    world=[];positions=[]
    for j,parent in enumerate(parents):
        if j==0:world.append(local[0]);positions.append(x[:,:,4:7]@q+origin[:,None])
        else:
            world.append(world[parent]@local[j])
            positions.append(positions[parent]+(world[parent]@offsets[j,:,None]).squeeze(-1))
    return torch.stack(positions,2)
