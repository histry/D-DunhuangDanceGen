"""Source-only Root/contact observations; no RootNet and no output replacement."""
import numpy as np
import torch
from .event_geometry import _matrix
from .event_music_adapter import EventMusicAdapter
SCHEMA='root_contact_event_music_v1'
CHANNELS=37


def root_contact_features(native,observed,contact_observed=None):
    b,n,c=native.shape
    if c!=151 or n<2 or observed.shape!=native.shape or observed.dtype!=torch.bool:
        raise ValueError('Observed native Root reference required')
    pv=observed[:,:,4:7].all(-1);rv=observed[:,:,7:13].all(-1)
    cv=torch.zeros(b,n,4,device=native.device,dtype=torch.bool) if contact_observed is None else contact_observed
    if cv.shape!=(b,n,4) or cv.dtype!=torch.bool:raise ValueError('Contact observation mask differs')
    if not torch.isfinite(native[:,:,4:7][pv]).all() or not torch.isfinite(native[:,:,7:13][rv]).all():
        raise ValueError('Nonfinite observed Root')
    if not torch.isfinite(native[:,:,:4][cv]).all():raise ValueError('Nonfinite observed contact proxy')
    position=torch.where(pv[...,None],native[:,:,4:7],0)
    identity=torch.tensor([1,0,0,0,1,0],device=native.device,dtype=native.dtype)
    matrix=_matrix(torch.where(rv[...,None],native[:,:,7:13],identity))
    yaw=torch.atan2(matrix[:,:,0,2],matrix[:,:,2,2])
    first=pv.long().argmax(1);first_r=rv.long().argmax(1)
    ix=torch.arange(b,device=native.device)
    origin=position[ix,first];heading=yaw[ix,first_r]
    relative=position-origin[:,None];co=heading.cos()[:,None];si=heading.sin()[:,None]
    local=torch.stack((co*relative[:,:,0]-si*relative[:,:,2],relative[:,:,1],
        si*relative[:,:,0]+co*relative[:,:,2]),dim=-1)
    local=torch.where(pv[...,None],local,0)
    velocity=torch.cat((torch.zeros_like(local[:,:1]),torch.diff(local,dim=1)*30),dim=1)
    vv=pv&torch.cat((torch.zeros_like(pv[:,:1]),pv[:,:-1]),dim=1)
    velocity=torch.where(vv[...,None],velocity,0)
    angle=yaw-heading[:,None]
    orient=torch.where(rv[...,None],torch.stack((angle.sin(),angle.cos()),-1),0)
    dy=torch.diff(angle,dim=1);omega=torch.cat((torch.zeros_like(angle[:,:1]),torch.atan2(dy.sin(),dy.cos())*30),dim=1)
    ov=rv&torch.cat((torch.zeros_like(rv[:,:1]),rv[:,:-1]),dim=1)
    omega=torch.where(ov,omega,0)
    speed=velocity.norm(dim=-1)
    saliency=torch.cat((torch.zeros_like(speed[:,:1]),torch.diff(speed,dim=1).abs()),dim=1)
    saliency=torch.where(vv&torch.cat((torch.zeros_like(vv[:,:1]),vv[:,:-1]),1),saliency,0)
    def boundaries(values,valid):
        first=valid.long().argmax(1);last=n-1-valid.flip(1).long().argmax(1);exists=valid.any(1)
        entry=values[ix,first]*exists[:,None];exit=values[ix,last]*exists[:,None]
        return torch.cat((entry,exit),-1)[:,None].expand(-1,n,-1),exists[:,None,None].expand(-1,n,2).to(native.dtype)
    bv,bvm=boundaries(velocity,vv);bh,bhm=boundaries(orient,rv)
    parts=[local,velocity,orient,omega[...,None],torch.where(cv,native[:,:,:4],0),
        cv.to(native.dtype),pv[...,None],rv[...,None],vv[...,None],ov[...,None],
        saliency[...,None],omega.abs()[...,None],bv,bh,bvm,bhm]
    result=torch.cat(parts,-1).to(native.dtype)
    if result.shape!=(b,n,CHANNELS) or not torch.isfinite(result).all():raise ValueError('Root/contact feature layout differs')
    return result


class RootContactAdapter(EventMusicAdapter):
    def __init__(self,mode='cross'):
        super().__init__(mode)
        self.root_input=torch.nn.Linear(CHANNELS,128,bias=False)
        torch.nn.init.zeros_(self.root_input.weight)
    def forward(self,geometry,music,return_aux=False,root=None):
        if root is None:root=torch.zeros(len(geometry),150,CHANNELS,device=geometry.device,dtype=geometry.dtype)
        if root.shape!=(len(geometry),150,CHANNELS) or not torch.isfinite(root).all():
            raise ValueError('Root/contact condition differs')
        event=self.encoder(self.input(geometry)+self.root_input(root))
        musical=self.music(music);query=musical if self.mode=='cross' else event
        attended=self.attention(query,event,event,need_weights=False)[0]
        fused=self.norm(event+musical+attended);residual=self.output(fused)
        return (residual,self.reconstruct(event)) if return_aux else residual


def build_reference(plan,frames):
    from dunhuang.io import checked
    from dunhuang.planning.time_map import source_positions
    from .soft_reference import resample_native
    if plan['frames']!=frames:raise ValueError('Root reference clock differs')
    windows=max(1,(max(0,frames-150)+74)//75+1);covered=150+75*(windows-1)
    features=np.zeros((covered,CHANNELS),np.float32);observed_frames=0;cursor=0
    for event in plan['events']:
        a,b=event['music_interval']
        if a!=cursor or not a<b<=frames:raise ValueError('Contiguous Root reference clock required')
        cursor=b
        evidence=event.get('root_observation')
        if evidence is None:continue
        import json
        e=json.loads(checked(evidence).read_text())
        if (e.get('schema')!='source_root_observation_v1' or e.get('source_sha256')!=event['reference']['sha256']
            or e.get('fps')!=30 or e.get('units')!='m' or e.get('up_axis')!='y'
            or not e.get('provenance') or e.get('contact_semantics')!='UNVERIFIED_GEOMETRIC_PROXY'):
            raise ValueError('Root source observation provenance differs')
        raw=np.load(checked(event['reference']));mask=np.load(checked(e['observed']))
        proxy=np.load(checked(e['contact_observed'])) if e.get('contact_observed') else np.zeros((len(raw),4),bool)
        if mask.shape!=raw.shape or mask.dtype!=bool or proxy.shape!=(len(raw),4) or proxy.dtype!=bool:
            raise ValueError('Observed source Root masks differ')
        positions=source_positions(event,np.arange(a,b));sampled=resample_native(raw,positions)
        left=np.floor(positions).astype(int);right=np.ceil(positions).astype(int)
        available=mask[left]&mask[right];contacts=proxy[left]&proxy[right]
        with torch.no_grad():v=root_contact_features(torch.from_numpy(sampled)[None],torch.from_numpy(available)[None],torch.from_numpy(contacts)[None])[0]
        features[a:b]=v.numpy();observed_frames+=int(available[:,4:7].all(-1).sum())
    if cursor!=frames:raise ValueError('Incomplete Root reference')
    return np.stack([features[75*i:75*i+150] for i in range(windows)]),dict(
        schema=SCHEMA,observed_root_frames=observed_frames,total_frames=frames,
        contact_semantics='UNVERIFIED_GEOMETRIC_PROXY',independent_static_contact_confidence=0,
        output_root_replaced=False,source_only=True)


def make_adapter(payload):
    return RootContactAdapter(payload['mode']) if payload.get('condition_schema')==SCHEMA else EventMusicAdapter(payload['mode'])
