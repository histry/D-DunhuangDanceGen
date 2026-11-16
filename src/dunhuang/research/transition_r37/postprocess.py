"""Frozen saved R36 outputs; fixed KF/RTS/residual RTS; full decoded motion before FK."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk,rotation_sample
from .smoothing import linear,rotation_filter,residual_rts
from .clock import source_positions
from .evaluate import decode,metrics,curves

def run(cfg,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh smoothing output required')
    out.mkdir(parents=True);parent=Path(cfg['r36']);raw=np.load(parent/'generation_repair01/arrays.npz');meta=read(parent/'generation_repair01/result.json')
    z=np.load(Path(cfg['r35'])/'generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog'])
    arrays=dict(skeleton_J22=J,parents=parents);records=[]
    for row in meta['records']:
        key=row['key'];binding=row['binding'];start=binding['window_start']
        song=next(s for s in read(Path(cfg['r35'])/'planning/songs.json') if s['id']==binding['song_id']);plan=read(song['plan']['path'])
        reference=[];intervals=[]
        for e in plan['events']:
            a=max(start,e['music_interval'][0]);b=min(start+256,e['music_interval'][1])
            if a>=b:continue
            intervals.append([a-start,b-start]);reference.append(rotation_sample(lib.rotations[e['source_uid']],source_positions(e,np.arange(a,b))))
        reference=np.concatenate(reference);budget=raw[key+'_budget'];boundaries=[b for a,b in intervals[:-1]];record=dict(key=key,seeds=binding['seeds'],binding=binding,intervals=intervals,methods={})
        arrays[key+'_reference_R']=reference;arrays[key+'_reference22']=relative_fk(reference,J,parents)
        for arm in ('Raw','KF','RTS','ResidualRTS'):
            values=[];stats=[];diag=[]
            for seed in range(2):
                motion=raw[key+'_Edit_139'][seed].copy();R,root=decode(motion);initial=R.copy()
                if arm in ('KF','RTS'):
                    R=rotation_filter(R,cfg['kalman_angle_sigma_rad'],cfg['kalman_angular_accel_sigma'],smooth=arm=='RTS')
                    root=linear(root,cfg['kalman_root_sigma_m'],cfg['kalman_root_accel_sigma'],smooth=arm=='RTS')
                elif arm=='ResidualRTS':
                    R=residual_rts(R,reference,intervals,budget,cfg)
                    # Root is a model trajectory, independently smoothed; it is not source truth.
                    root=linear(root,cfg['kalman_root_sigma_m'],cfg['kalman_root_accel_sigma'])
                motion[:,7:]=R[...,:2,:].reshape(256,132);motion[:,4:7]=root
                values.append(motion);stats.append(metrics(R,reference,root,J,parents,boundaries))
                diag.append(curves(R,reference,root,J,parents))
            arrays[key+'_'+arm+'_139']=np.stack(values);arrays[key+'_'+arm+'_22']=np.stack([relative_fk(decode(v)[0],J,parents)+decode(v)[1][:,None] for v in values])
            record['methods'][arm]=dict(metrics=stats,diagnostics=diag,body_budget_enforced=arm=='ResidualRTS',contact_values='COPIED_MODEL_OUTPUT_NOT_CONTACT_TRUTH')
        records.append(record)
    np.savez_compressed(out/'arrays.npz',**arrays);result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parent_sha256=sha(parent/'generation_repair01/arrays.npz'),
        config=cfg,records=records,optimizer_steps=0,sampler_calls=0,parameter_searches=0,dt_seconds=1/30,rotation_filter='RELINEARIZED_RIGHT_ERROR_DIAGONAL_AXIS_APPROXIMATION_NOT_EXACT_LIE_EKF',
        residual_chart='SPATIAL_LOCAL_PARENT_AXES_SEPARATE_PER_SOURCE_EVENT',viewer_only_smoothing=False,whole_song_generations=0,operational_qualified=False)
    write(out/'result.json',result)
    if replay:
        from dunhuang.research.coverage_r34.common import same_arrays
        same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert result==read(Path(replay)/'result.json')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(read('src/dunhuang/research/transition_r37/config.json'),a.out,a.replay_from)
