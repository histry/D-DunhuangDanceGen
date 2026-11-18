"""Phase-enabled segmented source memory, state-compatible with frozen Lodge."""
import torch
import numpy as np
from dunhuang.research.bridge_r38.source import SegmentedReference
from dunhuang.research.lodge_adapter_r25.model import PARTS


def observed_phases(values):
    f=values['features'].detach().cpu().numpy();pose=values['valid'].any(-1).cpu().numpy();dv=values['velocity_valid'].all(-1).cpu().numpy();batch,n=f.shape[:2];phases=[]
    for i in range(batch):
        t=np.arange(n,dtype=float)/(n-1);base=np.stack((t,np.sin(2*np.pi*t),np.cos(2*np.pi*t),np.ones(n)),-1)
        speed=np.linalg.norm(f[i,:,12:,6:],axis=-1).mean(-1)
        accel=np.r_[0,np.diff(speed)];acc_valid=dv[i]&np.r_[False,dv[i,:-1]]
        jerk=np.r_[0,np.diff(accel)];jerk_valid=acc_valid&np.r_[False,acc_valid[:-1]]
        bumps=[]
        for x,valid in ((abs(accel),acc_valid),(speed,dv[i]),(abs(jerk),jerk_valid),(-speed,dv[i])):
            ids=np.flatnonzero(valid)
            if not len(ids):bumps.append(np.zeros(n));continue
            k=int(ids[np.argmax(x[ids])]);bumps.append(np.exp(-.5*((np.arange(n)-k)/4.)**2))
        phases.append(np.concatenate((base,np.stack(bumps,-1)),-1).astype(np.float32))
    query=torch.tensor(np.stack(phases),device=values['features'].device)
    ref=torch.where(values['valid'].any(-1)[...,None],query,torch.zeros_like(query))
    return ref,query


class PhaseSegmentedReference(SegmentedReference):
    def set_reference(self,features,valid,clock,ref_phase,query_phase,mapped_source,control_mask,velocity_valid,segments):
        if ref_phase.shape!=(*features.shape[:2],8) or query_phase.shape!=(len(features),256,8):raise ValueError('Phase shape')
        if not torch.isfinite(ref_phase).all() or not torch.isfinite(query_phase).all():raise ValueError('Observed phases must be finite')
        super().set_reference(features,valid,clock,ref_phase,query_phase,mapped_source,control_mask,velocity_valid,segments)
        safe=torch.where(valid[...,None],features,torch.zeros_like(features)).clone()
        safe[:,:,:,6:]=torch.where(velocity_valid[...,None],safe[:,:,:,6:],torch.zeros_like(safe[:,:,:,6:]))
        inp=torch.cat((safe,valid[...,None].to(safe)),-1);parts=[layer(inp[:,:,p].flatten(2)) for layer,p in zip(self.parts,PARTS)]
        frame=valid.any(-1);present=frame.any(-1);rp=torch.where(frame[...,None],ref_phase,torch.zeros_like(ref_phase))
        memory=self.input(torch.cat(parts+[clock,rp],-1))*frame[...,None]
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
        self.context=(torch.cat((glob[:,None],memory),1),torch.cat((torch.ones_like(present[:,None]),frame),1),present,query_phase)

    def inject(self,index,hidden):
        result=super().inject(index,hidden)
        return hidden if self.context is None else torch.where(self.control_mask[...,None],result,hidden)


def install(rt):
    model=PhaseSegmentedReference(4.).cuda();model.load_state_dict(rt.c.change.state_dict(),strict=True);model.requires_grad_(False);rt.c.change=model;rt.frozen=rt.fingerprint()
