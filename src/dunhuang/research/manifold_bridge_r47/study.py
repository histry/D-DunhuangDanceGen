"""Same-observation bridge experiment, no neural sampling or hidden-pose construction."""
import argparse,shutil,subprocess
from pathlib import Path
import numpy as np
from dunhuang.contracts import sixd
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.reperformance_r44.motion import decode,world,kinematics
from dunhuang.research.bridge_audit_r46.audit import angular
from dunhuang.research.bridge_audit_r46.admission import certificate
from .geometry import segmented,spline

def build(out,controls=False):
    cfg=read(Path(__file__).with_name('config.json'));root=Path('outputs/manifold_bridge_r47_01');parent=Path(cfg['parent'])/'main';out=Path(out)
    if controls and not read(root/'receipts/main_browser_acceptance.json')['passed']:raise ValueError('Deliver actual G1 before comparisons')
    if out.exists():raise ValueError('Fresh R47 bridge')
    out.mkdir(parents=True);meta=read(parent/'result.json');z=Arrays(parent/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];arr=dict(skeleton_J22=J,parents=parents);records=[];a,b=cfg['gap'];visible=np.ones(256,bool);visible[a:b]=False
    for r in meta['records']:
        key=r['key'];raw=z[key+'_Geometry_139'];R=decode(raw[0])[0];R[a:b]=np.nan;new,cert=segmented(R,cfg['gap'],cfg['segment_counts'],cfg['chart_radius_rad']);poison=R.copy();poison[a:b]=np.eye(3);again,other=segmented(poison,cfg['gap'],cfg['segment_counts'],cfg['chart_radius_rad']);assert cert==other
        if new is not None:assert np.array_equal(new,again)
        candidates={'G0':raw.copy()};certs={'G0':certificate(R,cfg['gap']),'G1':cert}
        if new is not None:
            values=raw.copy();values[:,a:b,7:]=sixd(new[a:b],row=True).reshape(1,b-a,132);candidates['G1']=values
        if controls:
            rr,sc=spline(R,cfg['gap'],cfg['spline_internal_segments'],cfg['chart_radius_rad']);certs['G2']=sc
            if rr is not None:values=raw.copy();values[:,a:b,7:]=sixd(rr[a:b],row=True).reshape(1,b-a,132);candidates['G2']=values
        truth=z[key+'_reference_R'];metrics={};curves={};continuous=r['source']['domain']!='ChangE-cross'
        for method,values in candidates.items():
            assert np.array_equal(values[:,visible],raw[:,visible]) and np.array_equal(values[:,:,:7],raw[:,:,:7]);xyz=np.stack([world(v,J,parents) for v in values]);Rout=decode(values[0])[0];P=relative_fk(Rout,J,parents);err=angular(truth,Rout);m=kinematics(values[0],J,parents);m.update(protected_exact=True,root_contact_exact=True,so3_orthogonality_error=float(np.max(abs(Rout.swapaxes(-1,-2)@Rout-np.eye(3)))),determinant_error=float(np.max(abs(np.linalg.det(Rout)-1))),gap_angle_error_rad=float(err[a:b].mean()) if continuous else None,frame117_angle_error_rad=float(err[117].mean()) if continuous else None,gap_target_scope='CONTINUOUS_RECORDING_TRUTH' if continuous else 'NO_UNIQUE_CROSS_SOURCE_TRUTH',shoulder_wrist_min_m=float(np.min(np.linalg.norm(P[a:b,[20,21]]-P[a:b,[16,17]],axis=-1))),wrist_separation_range_m=[float(x) for x in (np.linalg.norm(P[a:b,20]-P[a:b,21],axis=-1).min(),np.linalg.norm(P[a:b,20]-P[a:b,21],axis=-1).max())]);metrics[method]=m;arr[key+'_'+method+'_139']=values;arr[key+'_'+method+'_22']=xyz;arr[key+'_'+method+'_clock']=np.arange(256,dtype=float);curves[method]=np.r_[np.zeros(3),np.sqrt(np.mean(np.diff(xyz,n=3,axis=1)**2,axis=(0,2,3)))*27000].tolist()
        m0=metrics['G0'];screens={}
        for method in candidates:
            if method=='G0':continue
            m=metrics[method];screens[method]=dict(gap_D3_ratio=m['gap_D3']/max(m0['gap_D3'],1e-12),boundary_D3_ratio=m['boundary_D3']/max(m0['boundary_D3'],1e-12),geometry_eligible=True,physical_qualified=False,source_reconstruction_improved=None if not continuous else m['gap_angle_error_rad']<m0['gap_angle_error_rad'],numerical_screen=bool(m['gap_D3']<=cfg['gap_D3_ratio_max']*m0['gap_D3'] and m['boundary_D3']<=cfg['boundary_D3_ratio_max']*m0['boundary_D3']))
        arr[key+'_reference_R']=truth;shutil.copyfile(parent/r['audio'],out/r['audio']);source={k:v for k,v in r['source'].items() if k!='audit'};records.append(dict(key=key,source=source,seeds=r['seeds'],audio=r['audio'],music=r['music'],phase_maps=[],methods=list(candidates),metrics=dict(methods=metrics,curves=curves),certificates=certs,screens=screens,scope='R47 OBSERVED-ONLY SO3 BRIDGE. G0 frozen; G1 reanchored quintic with body-rate/acceleration jets; G2 SciPy RotationSpline comparison. Root/contact inherited and unchanged, source hidden gap evaluation only. Geometric admission is not naturalness or Native qualification.'))
        print('R47_BRIDGE',key,cert['status'],cert.get('selected_segments'),'methods',list(candidates),flush=True)
    np.savez_compressed(out/'arrays.npz',**arr);write(out/'result.json',dict(records=records,config=dict(cfg,registered_video_cases=['female_FeiTian_0__music','female_meditation_0__music','paired_0','old_failed']),parent_arrays_sha256=sha(parent/'arrays.npz'),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_steps=0,new_neural_samples=0,whole_song_generated=False,can_route=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');a=p.parse_args();build(a.out,a.controls)
