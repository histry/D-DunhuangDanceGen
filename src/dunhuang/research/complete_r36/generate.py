"""First visible effect: fixed R35 actual plan windows, both registered seeds, one editor."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.coverage_r34.common import Runtime,forbid_optimizers,same_arrays
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.transition_r35.compiler import compile,make_packets
from dunhuang.research.transition_r35.planning import Sources
from .core import read,write,sha,smooth_budget
from .edit import sample_edit

def diagnosis(p,q):
    a=p-p[:,:1];b=q-q[:,:1]
    return dict(source_error=np.sqrt(np.mean((a-b)**2,axis=(1,2))).tolist(),
        jerk=[0.,0.,0.]+np.linalg.norm(np.diff(a,n=3,axis=0)*27000,axis=-1).mean(-1).tolist(),
        root_speed=[0.]+np.linalg.norm(np.diff(p[:,0],axis=0)*30,axis=-1).tolist())

def run(cfg,root,out,replay=None):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh output required')
    out.mkdir(parents=True);forbid_optimizers();parent=Path(cfg['r35']);old=read(parent/'generation/result.json');frozen=np.load(parent/'generation/arrays.npz')
    r35=read('src/dunhuang/research/transition_r35/config.json');rt=Runtime(r35,context_training=True);restore_context(rt,root/'context/context.pt')
    src=Sources();arrays={};records=[]
    # Pre-registered R35 Graph windows, not selected on this candidate's output.
    for binding in [r for r in old['records'] if r['plan_arm']=='Graph']:
        key=binding['key'];song=next(x for x in read(parent/'planning/songs.json') if x['id']==binding['song_id'])
        plan=read(song['plan']['path']);compiled,now=compile(plan,binding['selected_slot'],song['audio']['sha256']);assert now['window_start']==binding['window_start']
        packet,context=make_packets(compiled,now);start=now['window_start'];reference=[]
        for t in range(start,start+256):
            e=next(e for e in plan['events'] if e['music_interval'][0]<=t<e['music_interval'][1]);r,p=src.body(e,[t]);reference.append(r[0])
        reference=np.array(reference);body=reference[...,:2,:].reshape(256,132).astype(np.float32)
        base=torch.tensor(frozen[key+'_Frozen_normalized'],device='cuda');music=torch.tensor(np.repeat(frozen[key+'_music35'][None],2,axis=0),device='cuda')
        budget=smooth_budget(256,24,cfg['body_core_max_rotation_rad'],cfg['body_edge_max_rotation_rad'])
        # Around actual plan boundaries edits may be larger, with a smooth falloff.
        for e in plan['events']:
            boundary=e['music_interval'][1]-start;distance=abs(np.arange(256)-boundary)
            u=np.clip(distance/24,0,1);u=u*u*(3-2*u)
            budget=np.maximum(budget,cfg['body_core_max_rotation_rad']+(cfg['body_edge_max_rotation_rad']-cfg['body_core_max_rotation_rad'])*(1-u))
        # Responsibility may extend into editable neighbor edges, never their protected body.
        from dunhuang.research.transition_r35.graph import zones
        index=binding['selected_slot'];left=zones(plan['events'][index-1]);right=zones(plan['events'][index+1])
        allowed=[max(now['write_interval'][0],left['protected_core'][1]),min(now['write_interval'][1],right['protected_core'][0])]
        if not allowed[0]<allowed[1]:raise ValueError('No legal edit responsibility')
        editable=(np.arange(256)+start>=allowed[0])&(np.arange(256)+start<allowed[1])
        budget[~editable]=0
        pred,nr,edit=sample_edit(rt,music,torch.tensor(np.repeat(body[None],2,axis=0),device='cuda'),base,binding['seeds'],budget,cfg['edit_start_t'],packet,context)
        raw=rt.n.unnormalize(pred.clone())
        with torch.no_grad():points=rt.f(raw,rt.s)[:,:,:22].cpu().numpy()
        for arm in ('Frozen','Context'):
            arrays[key+'_'+arm+'_22']=frozen[key+'_'+arm+'_22'];arrays[key+'_'+arm+'_139']=frozen[key+'_'+arm+'_139']
        arrays[key+'_Edit_22']=points;arrays[key+'_Edit_139']=raw.cpu().numpy();arrays[key+'_Edit_normalized']=pred.cpu().numpy();arrays[key+'_reference22']=frozen[key+'_reference22']
        arrays[key+'_budget']=budget;reference_points=arrays[key+'_reference22'];metrics={}
        for arm in ('Frozen','Context','Edit'):
            metrics[arm]=[diagnosis(p,reference_points) for p in arrays[key+'_'+arm+'_22']]
        angular=[]
        from scipy.spatial.transform import Rotation
        from dunhuang.contracts import matrices
        for b in range(2):
            R=matrices(arrays[key+'_Edit_139'][b,:,7:].reshape(256,22,6),row=True)
            error=Rotation.from_matrix((reference.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22)
            assert (error<=budget[:,None]+2e-5).all()
            angular.append(dict(mean_rad=float(error.mean()),maximum_rad=float(error.max())))
        row=dict(key=key,name=song['name']+' · 同一R35计划的来源编辑',binding=binding,edit=edit,noise=nr,angle=angular,
            responsible_interval=allowed,body_read_only_outside_responsibility=True,
            body_retention_is_enforced_not_learned=True,
            candidate_context_sha256=sha(root/'context/context.pt'),candidate_includes_context_task_correction=True,causal_ablation_claim=False,
            common_row_noise_matches_saved_baseline=any(x.get('row_sha256')==nr['row_sha256'] and x.get('seeds')==nr['seeds'] for x in old['noise']),
            diagnostics={arm:v[0] for arm,v in metrics.items()},all_seed_metrics=metrics,
            full_native_status='NOT_EVALUABLE_22_OF24',operational_qualified=False,source_status=binding['source_domain_status'])
        records.append(row);shutil.copyfile(parent/'generation'/(key+'.wav'),out/(key+'.wav'))
        responsible=editable;region={}
        for arm in ('Frozen','Context','Edit'):
            p=arrays[key+'_'+arm+'_22'];local=p-p[:,:,:1];truth=reference_points-reference_points[:,:1]
            region[arm]=dict(local_position_rmse_m=float(np.sqrt(np.mean((local[:,responsible]-truth[None,responsible])**2))),
                jerk_inside_responsibility=float(np.mean(np.linalg.norm(np.diff(local,n=3,axis=1)*27000,axis=-1)[:,responsible[3:]&responsible[2:-1]&responsible[1:-2]&responsible[:-3]])),
                activity_inside_responsibility=float(np.mean(np.linalg.norm(np.diff(local,axis=1)*30,axis=-1)[:,responsible[1:]&responsible[:-1]])))
        row['responsible_region_metrics']=region
        np.savez_compressed(out/'arrays.npz',**arrays);print('R36_EDIT',key,angular,flush=True)
    rt.check();rt.close();result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),records=records,
        parent_result_sha256=sha(parent/'generation/result.json'),parent_arrays_sha256=sha(parent/'generation/arrays.npz'),
        optimizer_steps=0,new_generated_trajectories=2*len(records),reused_baselines='EXACT_FROZEN_R35_ARRAYS',whole_song_generations=0,additional_ablation_calls=0,operational_qualified=False)
    write(out/'result.json',result)
    if replay:
        same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args()
    run(read('src/dunhuang/research/complete_r36/config.json'),a.root,a.out,a.replay_from)
