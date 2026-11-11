import torch
from torch import nn

class PrimitiveAdapter(nn.Module):
    """Zero-init residual on EDGE music tokens, before pooling and cross-attention."""
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(303,128)
        block=nn.TransformerEncoderLayer(128,4,256,0.,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(block,1)
        self.output=nn.Linear(128,512)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,value,known):
        if value.shape!=known.shape or value.shape[1:]!=(150,151):raise ValueError('Adapter condition contract differs')
        phase=torch.linspace(0,1,150,device=value.device)[None,:,None].expand(len(value),-1,-1)
        clean=torch.where(known,value,torch.zeros_like(value))
        return self.output(self.encoder(self.input(torch.cat((clean,known.float(),phase),-1))))

def masked_reference(target):
    known=torch.zeros_like(target,dtype=torch.bool)
    for a,b in ((0,8),(40,44),(106,110),(142,150)):known[:,a:b]=True
    return torch.where(known,target,torch.zeros_like(target)),known
