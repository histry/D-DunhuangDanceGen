import torch
from torch import nn

class BoundaryContextAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(606,128)
        layer=nn.TransformerEncoderLayer(128,4,256,dropout=0.,activation="gelu",batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,1)
        self.music=nn.Linear(512,128)
        self.output=nn.Linear(128,512)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,value,known,music):
        if value.shape!=known.shape or value.shape[-2:]!=(150,151) or known.dtype!=torch.bool or known[:,:,:4].any():
            raise ValueError("Actual Root/rotation context mask required; no contact observations")
        v=torch.where(known,value,torch.zeros_like(value));d1=torch.zeros_like(v);d2=torch.zeros_like(v)
        pair=known[:,1:]&known[:,:-1];d1[:,1:]=(v[:,1:]-v[:,:-1])*pair
        triple=pair[:,1:]&pair[:,:-1];d2[:,2:]=(d1[:,2:]-d1[:,1:-1])*triple
        u=torch.linspace(0,1,150,device=v.device,dtype=v.dtype)[None,:,None].expand(len(v),-1,-1)
        x=torch.cat([v,known.to(v.dtype),d1,d2,u,1-u],dim=-1)
        return self.output(self.encoder(self.input(x)+self.music(music)))
