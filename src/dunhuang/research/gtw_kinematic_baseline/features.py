"""GTW-inspired local reference kNN; not the original discrete-path algorithm."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import relative_fk

METHODS=('GTWLocal','GTWExtended','GTWContext')

def descriptors(R,J,parents):
    R=np.asarray(R,float);P=relative_fk(R,J,parents)
    torso=R[:,0]@R[:,3]@R[:,6]@R[:,9]
    P=np.einsum('tji,tkj->tki',torso,P-P[:,9:10])
    P=uniform_filter1d(P,size=3,axis=0,mode='nearest')
    V=np.gradient(P,1/30,axis=0,edge_order=2);A=np.gradient(V,1/30,axis=0,edge_order=2)
    W=np.zeros((len(R),22,3));W[1:]=Rotation.from_matrix((R[:-1].swapaxes(-1,-2)@R[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30;W[0]=W[1]
    angular=np.gradient(W,1/30,axis=0,edge_order=2);jerk=np.gradient(A,1/30,axis=0,edge_order=2)
    blocks=[x.reshape(len(R),-1) for x in [P,V,A,W,angular,jerk]]
    return blocks

class ReferenceCost:
    def __init__(self,reference,J,parents):
        self.J,self.parents=J,parents
        self.reference=descriptors(reference,J,parents)
        # Scale per block from natural-speed reference only, never candidate statistics.
        self.scales=[max(float(np.sqrt(np.mean(x[18:-18]**2))),1e-3)*np.sqrt(x.shape[1]) for x in self.reference]
        self.refs={m:self.vector(self.reference,m)[18:-18] for m in METHODS}
        if min(len(x) for x in self.refs.values())<3:raise ValueError('Natural reference support too short')

    def vector(self,blocks,method):
        n=3 if method=='GTWLocal' else 6
        parts=[x/s for x,s in zip(blocks[:n],self.scales[:n])]
        if method=='GTWContext':
            # Two fixed 0.5s contexts; no phase labels and no future hidden truth.
            for shift in [-15,15]:
                index=np.clip(np.arange(len(blocks[0]))+shift,0,len(blocks[0])-1)
                parts.extend(x[index]/s for x,s in zip(blocks[:3],self.scales[:3]))
        return np.concatenate(parts,axis=1)/np.sqrt(len(parts))

    def score(self,R,coarse_only=False,only=None):
        b=descriptors(R,self.J,self.parents);scores={};coarse={}
        for m in ([only] if only else METHODS):
            v=self.vector(b,m)[18:-18]
            if coarse_only:v=v[::4]
            D=cdist(v,self.refs[m],metric='euclidean')
            local=np.partition(D,2,axis=1)[:,:3].mean(axis=1)
            scores[m]=float(local.mean());coarse[m]=float(local.mean() if coarse_only else local[::4].mean())
        return scores,coarse

def select(rows,method,budget=None):
    usable=[r for r in rows if r.get('eligible')]
    if budget is not None:usable=sorted(usable,key=lambda r:(r['coarse'][method],r['id']))[:budget]
    return min(usable,key=lambda r:(r['scores'][method],r['id'])) if usable else None
