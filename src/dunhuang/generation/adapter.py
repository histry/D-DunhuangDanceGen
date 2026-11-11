"""Learnable zero-init residual on EDGE music tokens.

Reference is observed source motion, never the hidden output target.
An optional phase is measured from the Event Plan's source-to-music map.
"""
import torch
from torch import nn


class PrimitiveAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(303, 128)
        block = nn.TransformerEncoderLayer(128, 4, 256, 0., activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(block, 1)
        self.output = nn.Linear(128, 512)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value, known, phase=None):
        if (value.ndim != 3 or value.shape[1:] != (150, 151) or
                value.shape != known.shape or known.dtype != torch.bool):
            raise ValueError('Adapter condition contract differs')
        if phase is None:
            phase = torch.linspace(0, 1, 150, device=value.device, dtype=value.dtype)[None, :, None].expand(len(value), -1, -1)
        if (phase.shape != (len(value), 150, 1) or
                not torch.isfinite(phase).all() or
                torch.any((phase < 0) | (phase > 1))):
            raise ValueError('Observed source-phase contract differs')
        if not torch.isfinite(value[known]).all():
            raise ValueError('Nonfinite observed reference')
        clean = torch.where(known, value, torch.zeros_like(value))
        return self.output(self.encoder(self.input(torch.cat((clean, known.float(), phase), -1))))


def masked_reference(target, scope='full'):
    if scope not in ('full', 'event_soft', 'upperbody_probe'):
        raise ValueError('Unknown reference supervision scope')
    known = torch.zeros_like(target, dtype=torch.bool)
    if scope == 'full':
        for a, b in ((0, 8), (40, 44), (106, 110), (142, 150)):
            known[:, a:b] = True
    else:
        # Dense source prompt with truly hidden completion spans.
        known[:, :, 79:127] = True
        known[:, 25:40, 79:127] = False
        known[:, 90:115, 79:127] = False
    return torch.where(known, target, torch.zeros_like(target)), known
