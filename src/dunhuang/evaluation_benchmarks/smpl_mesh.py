"""Canonical beta=0 SMPL mesh from the user's explicit local model asset.

Independent standard LBS equations; no licensed model data is bundled. Legacy
Chumpy leaf arrays are decoded as data only, without importing/executing Chumpy.
"""
import pickle
import numpy as np
from dunhuang.io import checked
from dunhuang.contracts import matrices,motion151

class _ChLeaf:
    def __setstate__(self,state):self.__dict__.update(state)

class _ModelUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        import scipy.sparse
        if (module,name)==('chumpy.ch','Ch'):return _ChLeaf
        if (module,name)==('__builtin__','set'):return set
        if module=='numpy' and name in ('dtype','ndarray'):return getattr(np,name)
        if (module,name)==('numpy.core.multiarray','_reconstruct'):return np.core.multiarray._reconstruct
        if module in ('scipy.sparse.csc','scipy.sparse._csc') and name=='csc_matrix':return scipy.sparse.csc_matrix
        raise ValueError('Unexpected SMPL pickle global '+module+'.'+name)

def load_model(ref):
    with checked(ref).open('rb') as f:result=_ModelUnpickler(f,encoding='latin1').load()
    for k,v in list(result.items()):
        if isinstance(v,_ChLeaf):
            if not isinstance(getattr(v,'x',None),np.ndarray):raise ValueError('Non-leaf Chumpy model value')
            result[k]=v.x
    return result

class SMPLMesh:
    def __init__(self,ref,device='cpu'):
        import torch
        data=load_model(ref);self.ref=ref;self.device=device
        self.template=torch.as_tensor(np.asarray(data['v_template']),dtype=torch.float32,device=device)
        regressor=data['J_regressor'].toarray() if hasattr(data['J_regressor'],'toarray') else np.asarray(data['J_regressor'])
        self.rest=torch.as_tensor(regressor,dtype=torch.float32,device=device)@self.template
        self.weights=torch.as_tensor(data['weights'],dtype=torch.float32,device=device)
        self.posedirs=torch.as_tensor(data['posedirs'].reshape(-1,207),dtype=torch.float32,device=device)
        table=data['kintree_table'];ids={int(v):i for i,v in enumerate(table[1])}
        self.parents=[-1]+[ids[int(p)] for p in table[0,1:]]
        if len(self.rest)!=24 or self.template.shape[1]!=3 or self.weights.shape!=(len(self.template),24):
            raise ValueError('SMPL24 model layout differs')
        if any(p>=i or p<0 for i,p in enumerate(self.parents[1:],1)):raise ValueError('SMPL tree is not topologically ordered')
        if not torch.allclose(self.weights.sum(-1),torch.ones(len(self.template),device=device),atol=1e-4):
            raise ValueError('Invalid LBS weights')
        self.description=dict(model=ref,shape='canonical neutral beta0, shared for all methods; source betas unknown',
            vertices=len(self.template),up_axis='y',root='Native root-joint world position, not raw SMPL transl',
            backend='independent standard SMPL pose-blend + LBS; separately validated when official smplx backend is available')
    def forward(self,motion):
        import torch
        x=motion151(motion);batch=len(x);device=self.device
        rotations=torch.as_tensor(matrices(x[:,7:].reshape(-1,24,6)),dtype=torch.float32,device=device)
        root=torch.as_tensor(x[:,4:7],dtype=torch.float32,device=device)
        identity=torch.eye(3,device=device)
        feature=(rotations[:,1:]-identity).reshape(batch,207)
        posed=self.template[None]+(feature@self.posedirs.T).reshape(batch,-1,3)
        global_transforms=[]
        for j,parent in enumerate(self.parents):
            t=torch.zeros(batch,4,4,device=device);t[:,:3,:3]=rotations[:,j];t[:,3,3]=1.
            t[:,:3,3]=self.rest[j] if parent<0 else self.rest[j]-self.rest[parent]
            global_transforms.append(t if parent<0 else global_transforms[parent]@t)
        g=torch.stack(global_transforms,1)
        joints=g[:,:,:3,3]+root[:,None]-self.rest[0]
        correction=g.clone()
        correction[:,:,:3,3]-=torch.einsum('bjkl,jl->bjk',g[:,:,:3,:3],self.rest)
        transforms=torch.einsum('vj,bjkl->bvkl',self.weights,correction)
        vertices=(transforms[:,:,:3,:3]@posed[:,:,:,None]).squeeze(-1)+transforms[:,:,:3,3]
        vertices+=root[:,None]-self.rest[0]
        return vertices,joints
    def sequence(self,motion,chunk=64):
        import torch
        minimum=[];joints=[]
        with torch.no_grad():
            for start in range(0,len(motion),chunk):
                v,j=self.forward(motion[start:start+chunk])
                minimum.append(v[:,:,1].min(1).values.cpu().numpy());joints.append(j.cpu().numpy())
        return np.concatenate(minimum),np.concatenate(joints)
