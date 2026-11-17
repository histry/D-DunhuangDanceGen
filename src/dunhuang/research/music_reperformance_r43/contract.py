"""Per-region permissions derived only from the visible source, never a hidden target."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.transition_r37.geometry import c2_envelope

UPPER=(3,6,9,12,13,14,15,16,17,18,19,20,21)


@dataclass
class ReperformanceCondition:
    body_budget:np.ndarray
    root_envelope:np.ndarray
    phase_proxy:np.ndarray
    protected_frames:np.ndarray
    anchors:list
    provenance:dict

    def validate(self):
        b=self.body_budget
        if b.shape!=(256,22) or not np.isfinite(b).all() or (b<0).any():raise ValueError('256x22 finite radians')
        if self.root_envelope.shape!=(256,) or not np.isfinite(self.root_envelope).all() or (self.root_envelope<0).any() or (self.root_envelope>1).any():raise ValueError('Root envelope')
        if self.phase_proxy.shape!=(256,8) or not np.isfinite(self.phase_proxy).all():raise ValueError('Observed phase proxy')
        if not np.array_equal(self.protected_frames,(b==0).all(-1)&(self.root_envelope==0)):raise ValueError('Protection must follow actual permissions')
        return self


def permissions(rotations,cfg,provenance=None):
    R=np.asarray(rotations,float);a,b=cfg['gap'];n=len(R)
    if R.shape!=(256,22,3,3) or not np.isfinite(R[np.r_[0:a,b:n]]).all():raise ValueError('Visible source geometry')
    budget=np.zeros((n,22),np.float32);phase=np.zeros((n,8),np.float32);anchors=[]
    for side,(lo,hi) in enumerate(((0,a),(b,n))):
        # Derivatives are recomputed only within the real observed Core, never across its gap.
        source=R[lo:hi];v=Rotation.from_matrix((source[:-1].swapaxes(-1,-2)@source[1:]).reshape(-1,3,3)).as_rotvec().reshape(len(source)-1,22,3)
        speed=np.r_[0,np.linalg.norm(v[:,UPPER],axis=-1).mean(-1)*30]
        positions=np.arange(lo,hi);u=(positions-lo)/max(hi-lo-1,1)
        phase[lo:hi,side]=1;phase[lo:hi,2]=u;phase[lo:hi,3]=np.sin(np.pi*u);phase[lo:hi,4]=speed/(1+speed)
        left=lo+cfg['outer_protected_frames'] if lo==0 else lo+3
        right=hi-cfg['outer_protected_frames'] if hi==n else hi-3
        envelope=c2_envelope(n,[left,right])
        available=np.arange(lo+16,hi-16);peak=int(available[np.argmax(speed[available-lo])]);anchors.append(dict(frame=peak,kind='VISIBLE_UPPER_SPEED_PEAK_PROXY',semantic_truth='UNKNOWN'))
        d=abs(np.arange(n)-peak);w=cfg['phase_protection_half_width'];x=np.clip((d-2)/max(w-2,1),0,1);envelope*=x**3*(10-15*x+6*x*x)
        budget[:,UPPER]+=cfg['core_upper_radians']*envelope[:,None]
    bridge=c2_envelope(n,[a+3,b-3]);budget+=cfg['bridge_radians']*bridge[:,None]
    phase[a:b,5]=1;phase[a:b,2]=np.linspace(0,1,b-a);phase[:,6]=np.max(budget,axis=-1);phase[:,7]=bridge
    result=ReperformanceCondition(budget,bridge,phase,(budget==0).all(-1)&(bridge==0),anchors,provenance or {})
    return result.validate()
