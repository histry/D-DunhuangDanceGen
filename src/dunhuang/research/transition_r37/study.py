"""No-training planning and pose-content qualification, followed by a watchable release."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk
from .planning import search
from .memory import reuse
from .clock import source_positions
from .content import observations,mapped_observations

def compute_plan(payload):
    base,cfg,song,out,resume=payload;out=Path(out);parent=Path(cfg['r36'])/'planning';lib=Library(base['catalog']);p=out/(song['id']+'_plan.json')
    if resume and p.exists():
        from .generate_contract import verify_plan
        plan=read(p);verify_plan(plan,lib);return plan
    catalog=read(parent/'catalog.json');music=read(parent/(song['id']+'_music.json'));z=np.load(Path(cfg['r35'])/'generation/arrays.npz')
    plan=search(catalog,music,lib,{**base,**cfg},z['skeleton_J22'],z['parents']);write(p,plan);return plan

def run(cfg,out,replay=None,resume=False):
    out=Path(out)
    if out.exists() and (not resume or (out/'result.json').exists()):raise ValueError('Fresh output, or explicit incomplete-plan resume only')
    out.mkdir(parents=True,exist_ok=resume);parent=Path(cfg['r36'])/'planning';base=read('src/dunhuang/research/complete_r36/config.json');lib=Library(base['catalog'])
    catalog=read(parent/'catalog.json');z=np.load(Path(cfg['r35'])/'generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents']
    songs=[];content=[];old_songs=read(parent/'result.json')['songs']
    # Independent songs share no optimization state. Parallelism changes no arithmetic within any search.
    with ProcessPoolExecutor(max_workers=min(3,len(old_songs))) as pool:
        plans=list(pool.map(compute_plan,[(base,cfg,s,str(out),resume) for s in old_songs]))
    for old,plan in zip(old_songs,plans):
        before=read(old['new_plan'])
        # Same retimed source positions are consumed by content, display and generator.
        for e in plan['events']:
            (values,mask),q=mapped_observations(e,lib.rotations[e['source_uid']],J,parents)
            assert values.shape==(e['music_interval'][1]-e['music_interval'][0],274)
            assert np.array_equal(q,source_positions(e,np.arange(*e['music_interval'])))
        row=dict(id=old['id'],name=old['name'],audio=old['audio'],frames=old['frames'],old_plan=old['new_plan'],new_plan=str(out/(old['id']+'_plan.json')),
                 old_units=len(before['events']),new_units=len(plan['events']),before=reuse(before['events']),after=plan['reuse'],
                 mean_edge_proxy_before=None,mean_edge_proxy_after=float(np.mean([x['cost'] for x in plan['boundaries']])) if plan['boundaries'] else 0.)
        songs.append(row);print('R37_PLAN',row['id'],row['before']['reuse_ratio'],row['after']['reuse_ratio'],flush=True)
    for uid in sorted(lib.rotations):
        R=lib.rotations[uid]
        values,mask=observations(R,J,parents)
        # Static pose readout diagnostic: dynamics zero on both poses, content remains distinguishable.
        a=np.repeat(R[:1],16,axis=0);b=np.repeat(R[len(R)//2:len(R)//2+1],16,axis=0)
        va,ma=observations(a,J,parents);vb,mb=observations(b,J,parents)
        content.append(dict(source_uid=uid,frames=len(R),feature_dims=274,unknown_channels=int((~mask[0]).sum()),
            static_dynamic_difference=float(abs(va[:,198:272]-vb[:,198:272]).max()),static_pose_geometry_difference=float(abs(va[:,:198]-vb[:,:198]).mean()),
            claim='INPUT_INFORMATION_DIAGNOSTIC_NOT_LEARNED_RETRIEVAL'))
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,
                catalog_sha256=sha(parent/'catalog.json'),songs=songs,content=content,sdmr=dict(training_updates=0,can_route=False,
                    status='POSE_AWARE_INTERFACE_AND_LABEL_CONTRACT_ONLY; NO_NEW_RETRIEVAL_QUALIFICATION',
                    missing_before_training=['FROZEN_VARIABLE_UNIT_PAIRED_POPULATION_AND_NEGATIVE_RELATIONS','TRAIN_ONLY_STANDARDIZATION_AND_SELECTION_PROTOCOL']),
                optimizer_steps=0,whole_song_generations=0,operational_qualified=False)
    write(out/'result.json',result)
    if replay:
        from .replay_contract import compare_planning
        compare_planning(replay,out,out/'replay_binding.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay-from');p.add_argument('--resume-incomplete-plans',action='store_true');a=p.parse_args();run(read('src/dunhuang/research/transition_r37/config.json'),a.out,a.replay_from,a.resume_incomplete_plans)
