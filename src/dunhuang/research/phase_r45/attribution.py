"""Integrated chain-rule diagnostics with explicit finite-difference/quadrature residual."""
import argparse,math
from pathlib import Path
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation,RotationSpline
from numpy.polynomial.legendre import leggauss
from dunhuang.research.complete_r36.core import read,write,relative_fk,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.reperformance_r44.motion import decode

def stencil(order,h):
    x=np.arange(-3,4,dtype=float);V=x[None]**np.arange(7)[:,None];b=np.zeros(7);b[order]=math.factorial(order);return np.linalg.solve(V,b)/h**order

class Curve:
    def __init__(self,raw,J,parents,h):
        R,_=decode(raw);self.splines=[RotationSpline(np.arange(len(raw)),Rotation.from_matrix(R[:,j])) for j in range(22)];self.J=J;self.parents=parents;self.h=h
    def positions(self,q):return relative_fk(np.stack([p(np.asarray(q)).as_matrix() for p in self.splines],axis=1),self.J,self.parents)
    def derivatives(self,q):
        q=np.asarray(q);out=[[],[],[]]
        for start in range(0,len(q),512):
            times=(q[start:start+512,None]+np.arange(-3,4)[None]*self.h).ravel();R=np.stack([p(times).as_matrix() for p in self.splines],axis=1);P=relative_fk(R,self.J,self.parents).reshape(-1,7,22,3)
            for n in (1,2,3):out[n-1].append(np.einsum('s,nsjk->njk',stencil(n,self.h),P))
        return [np.concatenate(v) for v in out]

def integrated_terms(starts,poly,derivatives,order):
    nodes,w=leggauss(order);nodes=(nodes+1)/2;w=w/2;grid=np.array(np.meshgrid(nodes,nodes,nodes,indexing='ij')).reshape(3,-1).sum(0);weight=np.prod(np.array(np.meshgrid(w,w,w,indexing='ij')),axis=0).ravel();t=np.asarray(starts)[:,None]+grid[None];q=poly(t).ravel();d1,d2,d3=derivatives(q);shape=(len(starts),len(grid))+d1.shape[1:];q1=poly(t,1).ravel()[:,None,None];q2=poly(t,2).ravel()[:,None,None];q3=poly(t,3).ravel()[:,None,None]
    terms=[d3*q1**3,3*d2*q1*q2,d1*q3];return [np.einsum('s,nsjk->njk',weight,v.reshape(shape))*27000 for v in terms]

def diagnose(root,out):
    root=Path(root);meta=read(root/'result.json');cfg=meta['config'];z=Arrays(root/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];records=[];arrays={}
    for row in meta['records']:
        if row['source']['recording_uid'] not in cfg['detailed_attribution_sources']:continue
        key=row['key'];base=z[key+'_Geometry_139'][0];P0=relative_fk(decode(base)[0],J,parents);curve=Curve(base,J,parents,cfg['attribution_difference_step_source_frames']);P0hi=curve.positions(np.arange(256));methods={}
        for method,knots in [('R44C2',row['matched_anchor_maps']),('Phase',row['phase_maps'])]:
            actualP=relative_fk(decode(z[key+'_'+method+'_139'][0])[0],J,parents);J0=np.concatenate([np.diff(P0[a:b],n=3,axis=0)*27000 for a,b in cfg['core_intervals']]);actual=np.concatenate([np.diff(actualP[a:b],n=3,axis=0)*27000 for a,b in cfg['core_intervals']]);J0hi=np.concatenate([np.diff(P0hi[a:b],n=3,axis=0)*27000 for a,b in cfg['core_intervals']]);Phi=curve.positions(z[key+'_'+method+'_clock']);Jhi=np.concatenate([np.diff(Phi[a:b],n=3,axis=0)*27000 for a,b in cfg['core_intervals']]);roundtrip_delta=(actual-Jhi)-(J0-J0hi);byorder={}
            for order in cfg['attribution_quadrature_orders']:
                parts=[[],[],[]]
                for r in knots:
                    a,b=r['interval'];poly=BPoly.from_derivatives(r['target'],[[s,1.,0.] for s in r['source']],extrapolate=False);v=integrated_terms(np.arange(a,b-3),poly,curve.derivatives,order)
                    for i,x in enumerate(v):parts[i].append(x)
                A,B,C=[np.concatenate(v) for v in parts];residual=Jhi-A-B-C;terms=dict(source_speed=A-J0hi,clock_acceleration=B,clock_jerk=C,quadrature_source_derivative_residual=residual,representation_roundtrip_increment=roundtrip_delta);energy_delta=float(np.mean(actual**2)-np.mean(J0**2));contribution={n:float(np.mean((actual+J0)*x)) for n,x in terms.items()};denom=max(float(np.sqrt(np.mean(Jhi**2))),1e-10);byorder[str(order)]=dict(actual_D3_rms=float(np.sqrt(np.mean(actual**2))),baseline_D3_rms=float(np.sqrt(np.mean(J0**2))),continuous_path_sampled_D3_rms=float(np.sqrt(np.mean(Jhi**2))),baseline_continuous_path_sampled_D3_rms=float(np.sqrt(np.mean(J0hi**2))),energy_delta=energy_delta,signed_energy_contributions=contribution,contribution_sum_error=abs(sum(contribution.values())-energy_delta),relative_closure_rms=float(np.sqrt(np.mean(residual**2))/denom),warning='SIGNED_DIAGNOSTIC_NOT_CAUSAL_EFFECT; ROUNDTRIP_INCLUDES_6D_QUANTIZATION_AND_ROTATION_REPROJECTION; NO_NEW_PRECISION_CANDIDATE_SELECTED')
                arrays[f'{key}_{method}_{order}_terms']=np.stack([A,B,C,residual,roundtrip_delta]);arrays[f'{key}_{method}_{order}_actual']=actual
            low,high=cfg['attribution_quadrature_orders'];pred_low=arrays[f'{key}_{method}_{low}_terms'][:3].sum(0);pred_high=arrays[f'{key}_{method}_{high}_terms'][:3].sum(0);byorder['quadrature_relative_difference']=float(np.sqrt(np.mean((pred_low-pred_high)**2))/max(float(np.sqrt(np.mean(actual**2))),1e-10));arrays[key+'_'+method+'_joint_delta_energy']=np.mean(actual**2-J0**2,axis=(0,2));methods[method]=byorder;print('R45_ATTRIBUTION',key,method,byorder[str(cfg['attribution_quadrature_orders'][-1])]['relative_closure_rms'],flush=True)
        records.append(dict(key=key,methods=methods,scope='ROOT_RELATIVE_COMMON22_CORE; NOT_WORLD_ROOT_OR_CONTACT_CAUSALITY'))
    out=Path(out)
    if out.exists():raise ValueError('Fresh attribution')
    out.mkdir(parents=True);np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,input_arrays_sha256=sha(root/'arrays.npz'),config=cfg,optimizer_steps=0));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();diagnose(a.root,a.out)
