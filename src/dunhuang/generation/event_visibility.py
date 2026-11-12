"""Explicit visibility for upper8 event adapters; legacy behavior remains the default.
Unknown event keys are excluded. All-missing references bypass the event adapter.
"""
import torch

def clean_geometry(geometry):
    if geometry.ndim!=3 or geometry.shape[1:]!=(150,153):raise ValueError("Event geometry shape differs")
    bits=geometry[...,144:152]
    if not torch.isfinite(bits).all() or not ((bits==0)|(bits==1)).all():raise ValueError("Binary event observations required")
    known=bits.bool();valid=known.any(-1)
    values=geometry[...,:144].reshape(-1,150,3,8,6)
    adjacent=known&torch.cat((known[:,:1],known[:,:-1]),1)
    support=torch.stack((known,known,adjacent),2)[...,None]
    clean=torch.where(support,values,0).flatten(2)
    phase=torch.where(valid[...,None],geometry[...,152:],0)
    clean=torch.cat((clean,bits,phase),-1)
    if not torch.isfinite(clean).all():raise ValueError("Nonfinite observed event feature")
    return clean,valid

def forward_event(adapter,geometry,music,return_aux=False,visibility_policy="legacy",return_trace=False):
    if geometry.ndim!=3 or geometry.shape[1:]!=(150,153) or music.shape!=(len(geometry),150,512):raise ValueError("Native event/music token contract differs")
    if visibility_policy not in ("legacy","observed"):raise ValueError("Unknown event visibility policy")
    if not torch.isfinite(music).all():raise ValueError("Nonfinite music")
    valid=None;empty=None
    if visibility_policy=="observed":
        geometry,valid=clean_geometry(geometry);empty=~valid.any(-1)
        padding=~valid.clone();padding[empty,0]=False # dummy only avoids all-masked softmax
    else:
        if not torch.isfinite(geometry).all():raise ValueError("Nonfinite event/music tokens")
        padding=None
    encoded_input=adapter.input(geometry)
    event=adapter.encoder(encoded_input) if padding is None else adapter.encoder(encoded_input,src_key_padding_mask=padding)
    if valid is not None:event=torch.where(valid[...,None],event,0)
    musical=adapter.music(music);query=musical if adapter.mode=="cross" else event
    attended=adapter.attention(query,event,event,need_weights=False,key_padding_mask=padding)[0]
    if empty is not None:attended=torch.where(empty[:,None,None],0,attended)
    fused=adapter.norm(event+musical+attended);residual=adapter.output(fused)
    reconstruction=adapter.reconstruct(event) if (return_aux or return_trace) else None
    if empty is not None:
        residual=torch.where(empty[:,None,None],0,residual)
        if reconstruction is not None:reconstruction=torch.where(valid[...,None],reconstruction,0)
    if return_trace:
        return residual,reconstruction,dict(event_input=geometry,event_128_input=encoded_input,event_encoded=event,
            music_128=musical,attention=attended,fused=fused,residual_512=residual)
    return (residual,reconstruction) if return_aux else residual

def intervene(geometry,control,seed=0):
    if control not in ("original","removed","mask_only","unknown_corrupted","phase_reversed"):raise ValueError("Unknown event intervention")
    result=geometry.clone()
    if control=="removed":result[...,:152]=0
    elif control=="mask_only":result[...,:144]=0
    elif control=="phase_reversed":result[...,152:]=1-result[...,152:]
    elif control=="unknown_corrupted":
        known=(result[...,144:152]>0)[...,None].expand(-1,-1,-1,6)
        generator=torch.Generator(device=geometry.device);generator.manual_seed(seed)
        for start in (0,48,96):
            part=result[...,start:start+48].reshape(-1,150,8,6)
            noise=torch.randn(part.shape,generator=generator,device=part.device,dtype=part.dtype)*10
            result[...,start:start+48]=torch.where(known,part,noise).flatten(2)
    return result
