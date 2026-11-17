"""Array-based permissions, true Core response and absolute kinematics, with official FK."""
import argparse,collections
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.transition_r37.evaluate import decode,world_fk
from dunhuang.research.transition_r37.geometry import c2_envelope


def rms(x):return float(np.sqrt(np.mean(np.square(x))))


def kinematics(R,J,parents):
    p=relative_fk(R,J,parents);core=[p[8:101],p[155:248]]
    return dict(core_activity=rms(np.concatenate([np.diff(x,axis=0)*30 for x in core])),core_D2=rms(np.concatenate([np.diff(x,n=2,axis=0)*900 for x in core])),core_D3=rms(np.concatenate([np.diff(x,n=3,axis=0)*27000 for x in core])),gap_D3=rms(np.diff(p[101:155],n=3,axis=0)*27000),boundary_D2=rms(np.concatenate([np.diff(p[a-3:a+4],n=2,axis=0)*900 for a in (104,152)])))


def run(main,controls,out,official=False):
    main=Path(main);controls=Path(controls);out=Path(out)
    if out.exists():raise ValueError('Fresh R43 qualification')
    m=read(main/'result.json');c=read(controls/'result.json');z=np.load(main/'arrays.npz');q=np.load(controls/'arrays.npz');assert m['data_sha256']==c['data_sha256'];lookup={r['key']:r for r in c['records']};J=z['skeleton_J22'];parents=z['parents'];origin={r['case']:r for r in read('outputs/reference_r26_01/data/result.json')['records']};records=[];rt=None
    if official:
        import torch
        from dunhuang.research.bridge_r38.study import runtime
        from dunhuang.research.coverage_r34.common import forbid_optimizers
        forbid_optimizers();rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'))
    for r in m['records']:
        key=r['key'];anchor=z[key+'_Geometry_139'];A=[decode(x)[0] for x in anchor];R=z[key+'_reference_R'];T=relative_fk(R,J,parents);limits=z[key+'_body_budget'];rootenv=z[key+'_root_envelope'];methods={};effect={};base=z[key+'_Core_139']
        for name in r['methods']+[n for n in lookup[key]['methods'] if n!='Geometry']:
            source=z if name in r['methods'] else q;raw=source[key+'_'+name+'_139'];values=[]
            for seed,x in enumerate(raw):
                P,root=decode(x);_,aroot=decode(anchor[seed]);angle=Rotation.from_matrix((A[seed].swapaxes(-1,-2)@P).reshape(-1,3,3)).magnitude().reshape(256,22);budget=.6*c2_envelope(256,[104,152])[:,None] if name=='Frozen' else limits;rootlimit=.5*c2_envelope(256,[104,152]) if name=='Frozen' else .03*rootenv
                if name=='Geometry':budget=np.zeros_like(limits);rootlimit=np.zeros(256)
                assert np.isfinite(x).all() and (angle<=budget+3e-6).all();assert (np.linalg.norm(root-aroot,axis=-1)<=rootlimit+3e-6).all();protected=(np.max(np.broadcast_to(budget,(256,22)),axis=-1)==0)&(rootlimit==0);assert np.array_equal(x[protected],anchor[seed,protected])
                if name!='Frozen':assert np.array_equal(x[:,:4],anchor[seed,:,:4])
                world=world_fk(P,root,J,parents);assert abs(world-source[key+'_'+name+'_22'][seed]).max()<2e-6
                if official:
                    with torch.no_grad():actual=rt.f(torch.tensor(x[None],device='cuda'),rt.s)[0,:,:22].cpu().numpy()
                    assert abs(world-actual).max()<2e-6
                p=relative_fk(P,J,parents);s=kinematics(P,J,parents);s.update(gap_D0_error=rms(p[104:152]-T[104:152]),gap_D3_error=rms(np.diff((p-T)[101:155],n=3,axis=0)*27000),core_source_angle_rad=float(np.r_[angle[:104].ravel(),angle[152:].ravel()].mean()),root_foot_speed_p95_proxy=float(np.quantile(np.linalg.norm(np.diff(world[:,[10,11]],axis=0)*30,axis=-1),.95)),protected_exact=True);values.append(s)
            methods[name]=values
            if name not in ('Geometry','Frozen','Core'):
                differences=[]
                for a,b in zip(base,raw):
                    AA=decode(a)[0];BB=decode(b)[0];v=Rotation.from_matrix((AA.swapaxes(-1,-2)@BB).reshape(-1,3,3)).magnitude().reshape(256,22);differences.append(dict(core_angle_response=float(np.r_[v[:104].ravel(),v[152:].ravel()].mean()),gap_angle_response=float(v[104:152].mean())))
                effect[name]=differences
        records.append(dict(key=key,domain=r['domain'],role=r['role'],recording_uid=r['recording_uid'],source_group=origin[r['case']]['source_group'] if r['domain']=='AIST++' else r['recording_uid'],methods=methods,condition_effect=effect,hidden_truth_is_continuous=r['domain']!='ChangE-cross'))
    if rt is not None:rt.check();rt.close()
    groups=collections.defaultdict(list)
    for r in records:
        if r['domain']=='AIST++' and r['role']=='train':groups[r['source_group']].append(r)
    music={}
    for group,part in groups.items():
        v={name:{metric:float(np.mean([s[metric] for r in part for s in r['methods'][name]])) for metric in ('gap_D0_error','gap_D3_error')} for name in ('Core','NoMusic','new_plan')};music[group]=dict(values=v,passed=all(v['Core'][k]<v[n][k] for k in ('gap_D0_error','gap_D3_error') for n in ('NoMusic','new_plan')))
    change={}
    for r in records:
        if r['domain']=='ChangE' and r['role']=='train':
            a={k:float(np.mean([s[k] for s in r['methods']['Core']])) for k in ('core_activity','core_D3','gap_D3')};b={k:float(np.mean([s[k] for s in r['methods']['Geometry']])) for k in a};ratio=a['core_activity']/max(b['core_activity'],1e-8);change[r['recording_uid']]=dict(activity_ratio=ratio,core_D3_ratio=a['core_D3']/max(b['core_D3'],1e-8),gap_D3_ratio=a['gap_D3']/max(b['gap_D3'],1e-8),passed=.85<=ratio<=1.15 and a['core_D3']<=1.05*b['core_D3'] and a['gap_D3']<=1.05*b['gap_D3'])
    empirical_gate=sum(v['passed'] for v in music.values())>=3 and sum(v['passed'] for v in change.values())>=3
    value=dict(records=records,train_music_gate=music,train_change_gate=change,empirical_training_gate_passed=empirical_gate,training_admitted=False,additional_requirement_if_empirical_gate_passes='VERIFIED_CROSS_RECORDING_REFERENCE_AND_35D_139D_TARGET_CONTRACT',official_fk_executed=official,main_arrays_sha256=sha(main/'arrays.npz'),control_arrays_sha256=sha(controls/'arrays.npz'),scope='CONDITION_RESPONSE_IS_NOT_MUSIC_COORDINATION; ABSOLUTE_D3_SEPARATE_FROM_PAIRED_ERRORS',native24='NOT_EVALUABLE_22_OF24',contact_truth='UNKNOWN',optimizer_steps=0,whole_song_generations=0,operational_qualified=False);write(out,value)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--controls',required=True);p.add_argument('--out',required=True);p.add_argument('--official',action='store_true');a=p.parse_args();run(a.main,a.controls,a.out,a.official)
