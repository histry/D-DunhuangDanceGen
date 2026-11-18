"""New metadata-selected continuous gaps and cross-event boundaries, effect first."""
import argparse,collections
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.bridge_audit_r46.admission import certificate
from dunhuang.research.manifold_bridge_r47.geometry import segmented,spline
from .geometry import optimize

ROOT=Path('outputs/bridge_dynamics_r48_01')
CFG='src/dunhuang/research/event_feasibility_r48/config.json'

def metrics(R,J,parents,gap,truth=None):
    a,b=gap;P=relative_fk(R,J,parents);d2=np.sqrt(np.mean((np.diff(P,n=2,axis=0)*900)**2,axis=(1,2)));d3=np.sqrt(np.mean((np.diff(P,n=3,axis=0)*27000)**2,axis=(1,2)))
    speed=Rotation.from_matrix((R[:-1].swapaxes(-1,-2)@R[1:]).reshape(-1,3,3)).as_rotvec().reshape(-1,22,3)*30
    acc=np.diff(speed,axis=0)*30
    return dict(gap_D2=float(np.quantile(d2[a-2:b],.95)),gap_D3=float(np.quantile(d3[a-3:b],.95)),boundary_D3=float(max(d3[a-3:a+1].max(),d3[b-3:b+1].max())),sampled_boundary_rate_jump=float(max(np.linalg.norm(speed[a-1]-speed[a-2],axis=-1).mean(),np.linalg.norm(speed[b]-speed[b-1],axis=-1).mean())),sampled_boundary_acc_jump=float(max(np.linalg.norm(acc[a-1]-acc[a-2],axis=-1).mean(),np.linalg.norm(acc[b-1]-acc[b-2],axis=-1).mean())),shoulder_wrist_min_m=float(np.linalg.norm(P[a:b,[20,21]]-P[a:b,[16,17]],axis=-1).min()),wrist_separation_max_m=float(np.linalg.norm(P[a:b,20]-P[a:b,21],axis=-1).max()),gap_angle_error_rad=None if truth is None else float(Rotation.from_matrix((truth[a:b].swapaxes(-1,-2)@R[a:b]).reshape(-1,3,3)).magnitude().mean()),jerk_curve=np.r_[np.zeros(3),d3].tolist(),physical_qualified=False)

def population(index,lib):
    nodes=[];tasks=[]
    for uid in sorted(lib.rows):
        eligible=sorted([u for u in index['units'] if u['source_uid']==uid and u['duration_frames']>=104 and np.diff(u['support_component'])[0]>=256],key=lambda u:u['source_interval'])
        if len(eligible)<2:raise ValueError('Two genuine components required')
        for num,k in enumerate((len(eligible)//3,2*len(eligible)//3)):
            u=eligible[k];nodes.append(u);mid=sum(u['source_interval'])//2;lo,hi=u['support_component'];start=min(max(mid-128,lo),hi-256);raw=lib.rotations[uid][start:start+256].copy()
            tasks.append(dict(key='%s_%d'%(uid,num),left=u,right=u,kind='CONTINUOUS_RECORDING',source_window=[start,start+256],R=raw,truth=raw.copy()))
    for i,u in enumerate(nodes):
        v=nodes[(i+2)%len(nodes)];assert u['source_uid']!=v['source_uid'];a=u['source_interval'][1];b=v['source_interval'][0]
        R=np.full((256,22,3,3),np.nan);R[:104]=lib.rotations[u['source_uid']][a-104:a];R[152:]=lib.rotations[v['source_uid']][b:b+104]
        tasks.append(dict(key='cross_%02d'%i,left=u,right=v,kind='CROSS_SOURCE_NO_HIDDEN_TRUTH',source_window=None,R=R,truth=None))
    return tasks

def reference(index,lib,J,parents):
    D2=[];D3=[];shape=[];files=[]
    for s in index['sources']:
        if s['role']!='train':continue
        uid=s['source_uid'];files.append(dict(source_uid=uid,native_sha256=s['native_sha256']))
        for a,b in s['accepted_components']:
            P=relative_fk(lib.rotations[uid][a:b],J,parents)
            if len(P)<4:continue
            D2.extend(np.sqrt(np.mean((np.diff(P,n=2,axis=0)*900)**2,axis=(1,2))));D3.extend(np.sqrt(np.mean((np.diff(P,n=3,axis=0)*27000)**2,axis=(1,2))));shape.extend(np.linalg.norm(P[:,[20,21]]-P[:,[16,17]],axis=-1).ravel())
    return dict(D2=float(np.quantile(D2,.95)),D3=float(np.quantile(D3,.95)),shoulder_wrist_lower_m=float(np.quantile(shape,.01)),scope='TRAIN_SOURCE_Q95_BODY_RELATIVE_NUMERICAL_REFERENCE_NOT_NATIVE',sources=files)

def run(out,controls=False):
    out=Path(out)
    if out.exists():raise ValueError('Fresh R48 bridge output')
    if controls and not read(ROOT/'receipts/main_browser_acceptance.json')['passed']:raise ValueError('Main effect must be watched before controls')
    cfg=read(CFG);index=read('outputs/event_coverage_r47_01/index/result.json');lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);sk=np.load('outputs/paired_event_r41_01/data/skeleton.npz');J=sk['J'];parents=sk['parents'];ref=reference(index,lib,J,parents);arrays=dict(J=J,parents=parents);records=[]
    a,b=cfg['gap'];visible=np.ones(256,bool);visible[a:b]=False
    for task in population(index,lib):
        key=task['key'];R=task.pop('R');truth=task.pop('truth');observed=R.copy();observed[a:b]=np.nan;certs={'G0':certificate(observed,(a,b))};methods={}
        try:methods['G0']=observed_bridge(observed,(a,b))
        except ValueError:pass
        g1,c=segmented(observed,(a,b),(4,),cfg['chart_radius_rad']);certs['G1']=c
        if g1 is not None:methods['G1']=g1
        g3,c=optimize(observed,J,parents,ref,cfg);certs['G3']=c
        if g3 is not None:methods['G3']=g3
        if controls:
            g2,c=spline(observed,(a,b),4,cfg['chart_radius_rad']);certs['G2']=c
            if g2 is not None:methods['G2']=g2
        ms={};screens={};arrays[key+'_source']=np.nan_to_num(relative_fk(np.nan_to_num(R,nan=0),J,parents)) if truth is None else relative_fk(truth,J,parents)
        arrays[key+'_source_valid']=visible if truth is None else np.ones(256,bool)
        for name,Q in methods.items():
            assert np.array_equal(Q[visible],R[visible]);m=metrics(Q,J,parents,(a,b),truth);ms[name]=m;arrays[key+'_'+name+'_R']=Q;arrays[key+'_'+name+'_P']=relative_fk(Q,J,parents)
            own=certs[name].get('eligible',certs[name].get('chart_eligible',certs[name].get('status','').startswith('GEOMETRICALLY_ADMITTED')))
            if name=='G2':own=certs[name]['status'].startswith('SPLINE_CHART_ELIGIBLE')
            screens[name]=dict(own_chart_eligible=bool(own),absolute_dynamic_reference_pass=m['gap_D2']<=ref['D2'] and m['gap_D3']<=ref['D3'],shape_reference_pass=m['shoulder_wrist_min_m']>=ref['shoulder_wrist_lower_m'],protected_exact=True,root_contact_scope='NOT_GENERATED_NOT_QUALIFIED')
        qualified=[n for n in ('G0','G1','G3') if n in screens and all(screens[n][v] for v in ('own_chart_eligible','absolute_dynamic_reference_pass','shape_reference_pass'))]
        choice=min(qualified,key=lambda n:(ms[n]['gap_D3'],n)) if qualified else None
        records.append(dict(**task,methods=list(methods),certificates=certs,metrics=ms,screens=screens,selected=choice,rejection=None if choice else 'NO_METHOD_PASSES_REGISTERED_SOURCE_REFERENCE_SCREEN',hidden_target_used_for_selection=False,can_route=False))
        print('R48_BRIDGE',key,choice,{k:round(m['gap_D3'],2) for k,m in ms.items()},flush=True)
    out.mkdir(parents=True);np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,reference=ref,config=cfg,controls=controls,neural_training_steps=0,new_neural_samples=0,whole_song_generations=0,can_route=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');a=p.parse_args();run(a.out,a.controls)
