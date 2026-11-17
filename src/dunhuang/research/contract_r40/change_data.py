"""Continuous ChangE body and relative-fit Root; no invented paired music/contact."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,Library

TRAIN=('female_FeiTian','male_pipa_1','female_lotus','male_ribbon')
HELD=('female_meditation','male_meditation','male_pipa_2','male_ribbon_FenHe')


def rooted_clip(displacement,start,length,lower,upper):
    d=np.asarray(displacement,float)[start:start+length]
    if len(d)!=length or not np.isfinite(d).all():raise ValueError('Continuous observed Root clip')
    # Gauge depends only on the first visible frame and the frozen model box.
    origin=(np.asarray(lower)+np.asarray(upper))/2
    root=origin+d-d[:1]
    return root,dict(gauge='FROZEN_MODEL_BOX_MIDPOINT_AT_FIRST_VISIBLE_FRAME',absolute_world_truth=False,model_box_supported=bool(((root>=lower)&(root<=upper)).all()),clipped=False)


def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh ChangE dataset')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/complete_r36/config.json');lib=Library(cfg['catalog']);path=Path('outputs/stabilize_r39_01/root_contract');z=np.load(path/'arrays.npz');meta=read(path/'result.json');normal=np.load('outputs/stabilize_r39_01/calibration/arrays.npz');scale=normal['root_normalizer_scale'];offset=normal['root_normalizer_offset'];lower=(-1-offset)/scale;upper=(1-offset)/scale;arrays={};rows=[]
    if set(TRAIN)&set(HELD) or set(TRAIN+HELD)!=set(lib.rotations):raise ValueError('Recording role partition')
    for uid in TRAIN+HELD:
        r=next(r for r in meta['records'] if r['source_uid']==uid);R=lib.rotations[uid];D=z[uid+'_displacement'];N=len(R)
        # Metadata-fixed early/middle/late intervals, not selection by Val quality.
        starts=[0,(N-256)//2,N-256]
        for j,start in enumerate(starts):
            key=uid+'_'+str(j);root,gauge=rooted_clip(D,start,256,lower,upper);arrays[key+'_R']=R[start:start+256];arrays[key+'_root']=root
            arrays[key+'_root_D1']=np.diff(root,axis=0)*30;arrays[key+'_root_D2']=np.diff(root,n=2,axis=0)*900
            # Every derivative is computed AFTER selection on the actual 30Hz target clock.
            rows.append(dict(key=key,source_uid=uid,raw_sha256=r['raw_sha256'],interval=[start,start+256],fps=30,role='train' if uid in TRAIN else 'held_from_this_adaptation',known_source_learning_control=uid in ('female_FeiTian','male_pipa_1'),historical_exposure=True,body_observed=True,relative_fit_translation_observed=True,paired_music=False,contact_truth='UNKNOWN',semantic_event_truth='UNKNOWN',gauge=gauge,body_training_admitted=True,relative_root_derivative_training_admitted=True,absolute_root_position_training_admitted=gauge['model_box_supported'],training_steps=0))
    np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=rows,root_contract_sha256=sha(path/'arrays.npz'),frozen_normalizer_sha256=sha('outputs/stabilize_r39_01/calibration/arrays.npz'),train_recordings=list(TRAIN),held_recordings=list(HELD),role_level='RECORDING_NOT_DANCER_OR_NEW_BLIND_GENERALIZATION',optimizer_steps=0,training_started=False,paired_music=False,contact_truth='UNKNOWN',native24='NOT_EVALUABLE_22_OF24',operational_qualified=False,scope='BODY_AND_RELATIVE_ROOT_DATASET_ONLY; POSITION_BOX_SUPPORT_DOES_NOT_GATE_BODY_OR_OBSERVED_DERIVATIVES; TRAINING_SEPARATELY_REGISTERED_AFTER_MAIN_CONTRACT_COMPARISON'))
    print('R40_CHANGE_PREFLIGHT',len(rows),'unsupported',sum(not r['absolute_root_position_training_admitted'] for r in rows),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
