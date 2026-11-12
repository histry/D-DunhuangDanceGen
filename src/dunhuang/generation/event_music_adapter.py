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

    def forward(self,geometry,music,return_aux=False,visibility_policy="legacy",return_trace=False):
        from .event_visibility import forward_event
        return forward_event(self,geometry,music,return_aux,visibility_policy,return_trace)



def authorize_event_adapter(payload,base_sha256,research=False,visibility_policy="legacy"):
    if payload.get('base_sha256')!=base_sha256 or payload.get('condition_schema') not in ('native_event_music_v1','root_contact_event_music_v1'):
        raise ValueError('Event/music base or geometry schema differs')
    if payload.get('mode') not in ('cross','additive'):
        raise ValueError('Unknown saved fusion mode')
    saved_visibility=payload.get('visibility_policy','legacy')
    if saved_visibility not in ('legacy','observed') or visibility_policy not in ('legacy','observed'):
        raise ValueError('Unknown saved/runtime event visibility')
    if saved_visibility!=visibility_policy:
        raise ValueError('Checkpoint visibility requires a matching research runtime')
    if research is not True:
        raise ValueError('Event/music Adapter is research-only, lacks generation qualification')
    return dict(research_unqualified=True,qualified=False)
