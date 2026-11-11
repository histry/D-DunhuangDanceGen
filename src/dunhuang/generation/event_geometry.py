"""Native SO(3) reference features, independent of EDGE target normalization."""
import torch


def _matrix(six):
    a,b=six[...,:3],six[...,3:]
    an=a.norm(dim=-1,keepdim=True)
    a=a/an.clamp_min(1e-8)
    b=b-(a*b).sum(-1,keepdim=True)*a
    bn=b.norm(dim=-1,keepdim=True)
    if torch.any(an<1e-6) or torch.any(bn<1e-6):
        raise ValueError('Degenerate observed native rotation')
    b=b/bn
    return torch.stack((a,b,torch.cross(a,b,dim=-1)),dim=-1)


def _six(matrix):
    return torch.cat((matrix[...,0],matrix[...,1]),dim=-1)


def event_geometry(native,available,phase):
    """153D: absolute/first-observed-relative/adjacent SO(3) columns, mask8, phase.

    Only local upper8 rotations enter. Root/contact/unobserved channels never enter;
    target Scaler support is NOT a reference admission requirement.
    """
    if native.ndim!=3 or native.shape[1:]!=(150,151) or available.shape!=native.shape or available.dtype!=torch.bool:
        raise ValueError('Native reference and observation contract differs')
    if phase.shape!=(len(native),150,1) or not torch.isfinite(phase).all() or torch.any((phase<0)|(phase>1)):
        raise ValueError('Source phase contract differs')
    observed=available[:,:,79:127].reshape(-1,150,8,6).all(-1)
    value=native[:,:,79:127].reshape(-1,150,8,6)
    if not torch.isfinite(value[observed]).all():
        raise ValueError('Nonfinite observed source rotation')
    identity=torch.tensor([1,0,0,0,1,0],device=native.device,dtype=native.dtype)
    clean=torch.where(observed[...,None],value,identity)
    rotation=_matrix(clean)
    # A separate first-observed origin for every batch and joint.
    first=observed.to(torch.int64).argmax(dim=1)
    gather=first[:,None,:,None,None].expand(-1,1,-1,3,3)
    origin=rotation.gather(1,gather)
    relative=origin.transpose(-1,-2)@rotation
    previous=torch.cat((rotation[:,:1],rotation[:,:-1]),dim=1)
    adjacent=previous.transpose(-1,-2)@rotation
    adjacent_valid=observed & torch.cat((observed[:,:1],observed[:,:-1]),dim=1)
    parts=[torch.where(observed[...,None],_six(rotation),0).flatten(2),
           torch.where(observed[...,None],_six(relative),0).flatten(2),
           torch.where(adjacent_valid[...,None],_six(adjacent),0).flatten(2),
           observed.to(native.dtype),phase]
    return torch.cat(parts,dim=-1)
