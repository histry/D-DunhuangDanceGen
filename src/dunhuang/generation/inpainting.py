import torch

def boundary_context(draft,valid,left=4,right=4):
    """Soft context only where measured valid; future errors are never fixed truth."""
    if draft.shape!=valid.shape or valid.dtype!=torch.bool:raise ValueError('Context validity mask required')
    mask=torch.zeros_like(valid)
    mask[:,:left]=valid[:,:left];mask[:,-right:]=valid[:,-right:]
    return torch.where(mask,draft,torch.zeros_like(draft)),mask
