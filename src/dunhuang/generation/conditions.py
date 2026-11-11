"""Explicit source reference, observed condition, and target supervision objects."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class ReferenceCondition:
    value: torch.Tensor
    known: torch.Tensor
    phase: torch.Tensor
    policy: str

def source_condition(reference,available,phase,policy='dense_upper8',window_offsets=None):
    if reference.ndim!=3 or reference.shape[1:]!=(150,151) or available.shape!=reference.shape or available.dtype!=torch.bool:
        raise ValueError('Source condition [B,150,151] contract differs')
    if phase.shape!=(len(reference),150,1) or not torch.isfinite(phase).all() or torch.any((phase<0)|(phase>1)):
        raise ValueError('Source phase contract differs')
    if policy not in ('dense_upper8','completion_upper8'):raise ValueError('Unknown source condition policy')
    known=torch.zeros_like(available);known[:,:,79:127]=available[:,:,79:127]
    if policy=='completion_upper8':
        offsets=torch.zeros(len(reference),device=reference.device,dtype=torch.long) if window_offsets is None else torch.as_tensor(window_offsets,device=reference.device)
        if offsets.shape!=(len(reference),) or offsets.dtype not in (torch.int32,torch.int64):raise ValueError('Integer window origins required')
        time=(torch.arange(150,device=reference.device)[None]+offsets[:,None])%150
        holes=((time>=25)&(time<40))|((time>=90)&(time<115))
        known=known&~holes[:,:,None]
    if not torch.isfinite(reference[known]).all():raise ValueError('Nonfinite observed source')
    return ReferenceCondition(torch.where(known,reference,torch.zeros_like(reference)),known,phase,policy)

def target_supervision(target,available,condition,scope='upper8_hidden'):
    if target.shape!=condition.value.shape or available.shape!=target.shape or available.dtype!=torch.bool:
        raise ValueError('Independent target supervision contract differs')
    if scope not in ('upper8_hidden','full_hidden','upper8_all','full_all'):raise ValueError('Unknown supervision scope')
    supervised=available.clone() if scope.endswith('_all') else available&~condition.known
    if scope.startswith('upper8'):
        joints=torch.zeros_like(supervised);joints[:,:,79:127]=True;supervised&=joints
    if not supervised.any() or not torch.isfinite(target[supervised]).all():
        raise ValueError('Missing finite supervised target observations')
    return supervised
