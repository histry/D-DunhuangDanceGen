"""Actual sampled angular seams and configuration; independent from analytic C2 admission."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,relative_fk,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.reperformance_r44.motion import decode

def seam_jets(R):
    velocity=[];acceleration=[]
    for t in (103,152):
        Q=R[t].swapaxes(-1,-2)[None]@R[t-2:t+3];v=Rotation.from_matrix(Q.reshape(-1,3,3)).as_rotvec().reshape(5,22,3);before=(3*v[2]-4*v[1]+v[0])/2;after=(-3*v[2]+4*v[3]-v[4])/2;velocity.append((after-before)*30);acceleration.append((v[4]-2*v[3]+v[2]-(v[2]-2*v[1]+v[0]))*900)
    return dict(sampled_boundary_body_rate_jump_rms_radps=float(np.sqrt(np.mean(np.asarray(velocity)**2))),sampled_boundary_body_acceleration_jump_rms_radps2=float(np.sqrt(np.mean(np.asarray(acceleration)**2))))

def run(main,out):
    main=Path(main);meta=read(main/'result.json');z=Arrays(main/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];rows=[]
    for r in meta['records']:
        key=r['key'];methods={}
        for name in r['methods']:
            R,_=decode(z[key+'_'+name+'_139'][0]);P=relative_fk(R,J,parents)[104:152];m=dict(r['metrics']['methods'][name]);m.update(seam_jets(R));m.update(torso_extension_min_m=float(np.linalg.norm(P[:,12]-P[:,0],axis=-1).min()),ankle_separation_min_m=float(np.linalg.norm(P[:,7]-P[:,8],axis=-1).min()),body_vertical_span_mean_m=float(np.ptp(P[:,:,1],axis=1).mean()));methods[name]=m
        rows.append(dict(key=key,domain=r['source']['domain'],role=r['source']['role'],baseline_local_chart_eligible=r['certificates']['G0']['chart_eligible'],methods=methods,certificates=r['certificates'],screens=r['screens'],physical_qualified=False,naturalness_truth='UNKNOWN_NO_INDEPENDENT_REVIEW'))
    write(out,dict(records=rows,input_arrays_sha256=sha(main/'arrays.npz'),policy='ANALYTIC_BODY_JET_MATCH_IS_DISTINCT_FROM_30HZ_FINITE_STENCIL_ERRORS; NO_HIDDEN_TARGET_USED_FOR_METHOD_SELECTION',optimizer_steps=0,operational_qualified=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.out)
