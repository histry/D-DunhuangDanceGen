"""Frozen Source: separate pose/derivative masks, segmented temporal convolution.

Unknown derivatives are neutralized and retained as explicit packet metadata.
No new learned derivative-uncertainty channel is claimed.
"""
import numpy as np
import torch
from dunhuang.research.coverage_r34.model import WindowReference,time_bias
from dunhuang.research.coverage_r34.common import WindowSource
from dunhuang.research.lodge_adapter_r25.model import PARTS
def feature_arrays(R,gap,event_ids=None):
    from scipy.spatial.transform import Rotation
    R=np.asarray(R,float);n=len(R);a,b=gap;ids=np.zeros(n,dtype=np.int64) if event_ids is None else np.asarray(event_ids,np.int64)
    if ids.shape!=(n,):raise ValueError('Event identity per frame')
    pose=np.ones((n,22),bool);pose[a:b]=False;velocity=pose.copy();velocity[0]=False
    velocity[1:] &= pose[:-1] & (ids[1:]==ids[:-1])[:,None]
    f=np.zeros((n,22,9),np.float32);f[:,:,:6]=np.where(pose[...,None],R[...,:2,:].reshape(n,22,6),0)
    safe=R.copy();safe[~pose]=np.eye(3)
    v=Rotation.from_matrix((safe[:-1].swapaxes(-1,-2)@safe[1:]).reshape(-1,3,3)).as_rotvec().reshape(n-1,22,3)*30
    f[1:,:,6:]=np.where(velocity[1:,:,None],v,0);segments=ids.copy();segments[a:b]=-1;segments[b:]+=int(ids.max()+2)
    return f,pose,velocity,segments
def packet(rotations,gap,uids,event_ids=None,device='cuda'):
    pieces=[feature_arrays(r,gap,None if event_ids is None else event_ids[i]) for i,r in enumerate(rotations)];b=len(pieces);n=len(rotations[0]);t=np.arange(n)/(n-1)
    values=dict(features=torch.tensor(np.stack([p[0] for p in pieces]),device=device),valid=torch.tensor(np.stack([p[1] for p in pieces]),device=device),velocity_valid=torch.tensor(np.stack([p[2] for p in pieces]),device=device),segments=torch.tensor(np.stack([p[3] for p in pieces]),device=device),clock=torch.tensor(np.broadcast_to(np.stack([np.sin(2*np.pi*t),np.cos(2*np.pi*t)],-1),(b,n,2)).copy(),device=device,dtype=torch.float32),ref_phase=torch.zeros(b,n,8,device=device),query_phase=torch.zeros(b,256,8,device=device),mapped_source=torch.arange(n,device=device,dtype=torch.float32)[None].repeat(b,1),control_mask=torch.ones(b,n,device=device,dtype=torch.bool))
    return WindowSource(values,tuple(uids)).validate()
class SegmentedReference(WindowReference):
    def set_reference(self,features,valid,clock,ref_phase,query_phase,mapped_source,control_mask,velocity_valid,segments):
        if velocity_valid.shape!=valid.shape or segments.shape!=features.shape[:2] or (velocity_valid & ~valid).any():raise ValueError('Separate derivative/event support')
        time_bias(mapped_source,features.shape[1],control_mask,self.beta);self.legacy=False
        safe=torch.where(valid[...,None],features,torch.zeros_like(features)).clone();safe[:,:,:,6:]=torch.where(velocity_valid[...,None],safe[:,:,:,6:],torch.zeros_like(safe[:,:,:,6:]))
        inp=torch.cat((safe,valid[...,None].to(safe)),-1);parts=[layer(inp[:,:,p].flatten(2)) for layer,p in zip(self.parts,PARTS)]
        frame=valid.any(-1);present=frame.any(-1);memory=self.input(torch.cat(parts+[clock,torch.zeros_like(ref_phase)],-1))*frame[...,None]
        for conv in self.temporal:
            delta=torch.zeros_like(memory)
            for i in range(len(memory)):
                for segment in torch.unique_consecutive(segments[i]).tolist():
                    if segment<0:continue
                    indices=torch.where((segments[i]==segment)&frame[i])[0]
                    if not len(indices):continue
                    if len(indices)>1 and not torch.all(indices.diff()==1):raise ValueError('Segment must be contiguous')
                    delta[i,indices]=torch.nn.functional.silu(conv(memory[i,indices].T[None])[0].T)
            memory=memory+delta
        memory=self.norm(memory);glob=(memory*frame[...,None]).sum(1)/frame.sum(1).clamp_min(1)[:,None]
        self.context=(torch.cat((glob[:,None],memory),1),torch.cat((torch.ones_like(present[:,None]),frame),1),present,torch.zeros_like(query_phase));self.mapped_source=mapped_source;self.control_mask=control_mask;self.readouts={};self.attention_readouts={}
def install(rt):
    branch=SegmentedReference(4.).cuda();branch.load_state_dict(rt.c.change.state_dict(),strict=True);branch.requires_grad_(False);rt.c.change=branch;rt.frozen=rt.fingerprint()
