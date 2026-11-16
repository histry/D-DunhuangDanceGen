"""Fresh CPU replay comparison, SO3 budgets, state FK and all metrics recomputed."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk
from .generate_contract import verify_plan
from .evaluate import decode,metrics,world_fk

def exact(a,b):
    x=np.load(a);y=np.load(b);assert set(x.files)==set(y.files)
    for k in x.files:assert np.array_equal(x[k],y[k]),k

def run(cfg,root,out):
    root=Path(root);checks={};summary=[]
    for name in ('planning','smoothing','generation'):
        if name!='planning':exact(root/name/'arrays.npz',root/(name+'_replay')/'arrays.npz')
        for p in (root/name).glob('*.json'):
            x=p.read_text().replace(str(root/name),'<RUN>');y=(root/(name+'_replay')/p.name).read_text().replace(str(root/(name+'_replay')),'<RUN>');assert x==y,p.name
        checks[name+'_fresh_process_exact_replay']=True
    lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog'])
    planning_comparison=[]
    from .planning import boundary_descriptor,edge
    rig=np.load(Path(cfg['r35'])/'generation/arrays.npz');J0=rig['skeleton_J22'];P0=rig['parents']
    def describe_plan(plan):
        descriptors=[];speeds=[];points=[]
        from .clock import source_positions
        from dunhuang.research.complete_r36.core import rotation_sample
        for e in plan['events']:
            descriptors.append(dict(e,_desc=boundary_descriptor(e,lib,J0,P0)))
            R=rotation_sample(lib.rotations[e['source_uid']],source_positions(e,np.arange(*e['music_interval'])));p=relative_fk(R,J0,P0);points.append(p)
            speeds.extend(np.linalg.norm(np.diff(p,axis=0)*30,axis=-1).ravel().tolist())
        p=np.concatenate(points)
        return dict(mean_within_event_body_speed_m_s=float(np.mean(speeds)),global_arm_extent_rms_m=float(np.sqrt(np.mean(np.ptp(p[:,[16,17,18,19,20,21]],axis=0)**2))),
                    source_count=len({e['source_uid'] for e in plan['events']}),family_count=len({e['family_uid'] for e in plan['events']}),
                    unified_mean_boundary_proxy=float(np.mean([edge(a,b)['cost'] for a,b in zip(descriptors,descriptors[1:])])),
                    scope='GEOMETRIC_DESCRIPTORS_NOT_INDEPENDENT_ARTISTIC_OR_PHYSICAL_QUALITY')
    for song in read(root/'planning/result.json')['songs']:
        plan=read(song['new_plan']);verify_plan(plan,lib)
        planning_comparison.append(dict(id=song['id'],before=describe_plan(read(song['old_plan'])),after=describe_plan(plan)))
    checks['source_support_maps_no_padding']=True
    z=np.load(root/'generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];meta=read(root/'generation/result.json')
    for row in meta['records']:
        key=row['key'];reference=z[key+'_reference_R'];record=dict(key=key,new_plan_consumed=row['consumes_new_r37_plan'],methods={})
        if row['consumes_new_r37_plan']:
            import json,hashlib
            new=read(root/'planning/song0_plan.json');assert hashlib.sha256(json.dumps(new,sort_keys=True).encode()).hexdigest()==row['plan_sha256']
        boundaries=[e['local_interval'][1] for e in row['binding']['events'][:-1]]
        record['all_reference_boundaries_local']=boundaries;record['selected_boundary_local']=128
        record['methods']['Reference']=[metrics(reference,reference,np.zeros((256,3)),J,parents,boundaries,row['gap'],128)]
        G=z[key+'_Geometry_R'];record['methods']['Geometry']=[metrics(G,reference,np.zeros((256,3)),J,parents,boundaries,row['gap'],128)]
        for arm in ('R36Like','Joint'):
            stats=[]
            anchor=reference if arm=='R36Like' else G
            for raw,stored in zip(z[key+'_'+arm+'_139'],z[key+'_'+arm+'_22']):
                R,Root=decode(raw);P=world_fk(R,Root,J,parents);assert np.allclose(P,stored,atol=2e-5)
                error=Rotation.from_matrix((anchor.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22)
                assert (error<=z[key+'_'+arm+'_budget'][:,None]+2e-5).all()
                stats.append(metrics(R,reference,Root,J,parents,boundaries,row['gap'],128))
            record['methods'][arm]=stats
            # Four-frame raw-x0 vs projected-x0 uses identical state and stencil.
            trace=z[key+'_'+arm+'_trace'];v=[]
            for step in trace:
                samples=[]
                for seed in step:
                    pair=[]
                    for raw4 in seed:
                        # Stored trace is NORMALIZED; body channels are R31 identity-normalized.
                        R,_=decode(raw4);P=relative_fk(R,J,parents)
                        pair.append(float(np.linalg.norm(np.diff(P,n=3,axis=0)*27000,axis=-1).mean()))
                    samples.append(pair)
                v.append(samples)
            record[arm+'_same_state_pre_post_boundary_jerk']=v
        summary.append(record)
    checks['saved_motion_fk_matches_viewer_tracks']=True;checks['budgets_satisfied']=True;checks['same_state_projection_diagnostics_saved']=True
    # Real retrieval inputs and Train-only normalization, independent of the optimization process.
    from .content_prepare import CONFIG
    from .content import observations
    import torch
    data=root/'content_data';manifest=read(data/'manifest.json');rows=manifest['rows'];normalization=read(data/'normalization.json')
    checkpoint=torch.load(root/'content_model/model.pt',map_location='cpu',weights_only=True)
    assert checkpoint['manifest_sha256']==sha(data/'manifest.json')
    assert read(root/'content_model/result.json')['optimizer_steps']==80 and read(root/'content_replay/result.json')['optimizer_steps']==0
    exact(root/'content_model/evaluation.npz',root/'content_replay/evaluation.npz')
    moments={name:dict(sum=np.zeros(n),square=np.zeros(n),count=np.zeros(n)) for name,n in [('motion',274),('music',50)]}
    for r in rows:
        assert r['role'] in ('train','val') and r['paired_clock_verified']
        assert sha(r['path'])==r['sha256'];unit=np.load(r['path'])
        if r['role']=='train':
            for name,m in moments.items():
                valid=unit[name+'_mask'];v=np.where(valid,unit[name],0).astype(float);m['sum']+=v.sum(0);m['square']+=(v*v).sum(0);m['count']+=valid.sum(0)
    for name,m in moments.items():
        count=np.maximum(m['count'],1);mean=m['sum']/count;std=np.sqrt(np.maximum(m['square']/count-mean*mean,1e-4));old=normalization['statistics'][name]
        assert np.array_equal(mean,np.array(old['mean'])) and np.array_equal(std,np.array(old['std'])) and np.array_equal(m['count'],np.array(old['observed_count']))
    original=Path(CONFIG['source']);sealed={r['raw_motion']['sha256']:r for r in read(original/'source_seals.json')}
    from dunhuang.io import checked
    chosen=[]
    for role in ('train','val'):
        ids=[i for i,r in enumerate(rows) if r['role']==role];chosen.extend([ids[0],ids[-1]])
    for i in chosen:
        r=rows[i];s=sealed[r['raw_motion_sha256']];cal=read(checked(s['source_clock']));assert cal['calibration']['verified']
        native=np.load(checked(s['native']));from dunhuang.contracts import matrices
        values,mask=observations(matrices(native[:,7:139].reshape(-1,22,6)),J,parents);a,b=r['source_interval'];x=values[a:b].copy();x[:,-2]=np.linspace(0,1,b-a);x[:,-1]=(b-a)/30
        u=np.load(r['path']);assert np.array_equal(x,u['motion']) and np.array_equal(mask[a:b],u['motion_mask'])
        audio=np.load(original/'audio'/(r['audio_sha256']+'.npz'));t=r['audio_frame_interval'][0]+np.arange(b-a);lo=np.floor(t).astype(int);hi=np.ceil(t).astype(int);alpha=(t-lo)[:,None]
        M=(audio['values'][lo]*(1-alpha)+audio['values'][hi]*alpha).astype(np.float32)
        assert np.array_equal(M,u['music']) and np.array_equal(audio['mask'][lo]&audio['mask'][hi],u['music_mask'])
    checks['all_variable_pair_hashes_and_train_only_statistics']=True;checks['four_preregistered_units_rebuilt_from_continuous_sources']=True
    checks['retrieval_weight_evaluation_exact_replay_zero_updates']=True
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checks=checks,generation=summary,planning_comparison=planning_comparison,
                optimizer_steps=0,verified_main_optimizer_updates=80,content_units=len(rows),reconstructed_unit_indices=chosen,
                training_reproduction_claim=False,whole_song_generations=0,operational_qualified=False,sdmr_can_route=False,native24='NOT_EVALUABLE_22_OF24')
    write(out,result)
    files=[]
    for name in ('planning','planning_replay','smoothing','smoothing_replay','generation','generation_replay'):
        for p in sorted((root/name).rglob('*')):
            if p.is_file():files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    for name in ('content_model','content_replay'):
        for p in sorted((root/name).rglob('*')):
            if p.is_file():files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    for name in ('manifest.json','normalization.json'):
        p=data/name;files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    write(root/'artifacts_manifest.json',dict(files=files,optimizer_steps=0,operational_qualified=False))
    print('R37_VERIFIED',len(files),checks,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(read('src/dunhuang/research/transition_r37/config.json'),a.root,a.out)
