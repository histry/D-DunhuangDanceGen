"""Parameter-matched event/music fusion. Research-only, single frozen EDGE."""
import torch
from torch import nn


class EventMusicAdapter(nn.Module):
    def __init__(self, mode='cross'):
        super().__init__()
        if mode not in ('cross','additive'):
            raise ValueError('Unknown event/music fusion')
        self.mode=mode
        self.input=nn.Linear(153,128)
        layer=nn.TransformerEncoderLayer(128,4,256,0.,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,1)
        self.music=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,128))
        self.attention=nn.MultiheadAttention(128,4,dropout=0.,batch_first=True)
        self.norm=nn.LayerNorm(128)
        self.reconstruct=nn.Linear(128,48)
        self.output=nn.Linear(128,512)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,geometry,music,return_aux=False):
        if geometry.ndim!=3 or geometry.shape[1:]!=(150,153) or music.shape!=(len(geometry),150,512):
            raise ValueError('Native event/music token contract differs')
        if not torch.isfinite(geometry).all() or not torch.isfinite(music).all():
            raise ValueError('Nonfinite event/music tokens')
        event=self.encoder(self.input(geometry))
        musical=self.music(music)
        # Both variants use all trainable modules and exactly the same parameters.
        query=musical if self.mode=='cross' else event
        attended=self.attention(query,event,event,need_weights=False)[0]
        fused=self.norm(event+musical+attended)
        residual=self.output(fused)
        return (residual,self.reconstruct(event)) if return_aux else residual


def authorize_event_adapter(payload,base_sha256,research=False):
    if payload.get('base_sha256')!=base_sha256 or payload.get('condition_schema')!='native_event_music_v1':
        raise ValueError('Event/music base or geometry schema differs')
    if payload.get('mode') not in ('cross','additive'):
        raise ValueError('Unknown saved fusion mode')
    if research is not True:
        raise ValueError('Event/music Adapter is research-only, lacks generation qualification')
    return dict(research_unqualified=True,qualified=False)
