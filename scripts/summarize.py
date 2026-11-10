"""Recompute a compact aggregate from saved real motions and their Native audits."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.evaluation import evaluate
from dunhuang.io import sha,write
p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);p.add_argument('report',type=Path);a=p.parse_args()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
rows=[];rejected=[]
for label in ['5s','15s','whole']:
    target=a.outputs/'supported_upper8'/label
    original=a.outputs/'experiment_v1'/('anchored_'+label)
    if 'Anchor outside frozen checkpoint normalization support' not in (original/'edge_backend.log').read_text():raise ValueError('Missing original failure evidence')
    rejected.append(dict(case=label,joints='12..21',status='rejected before sampling',reason='outside frozen normalizer support'))
    q=read(target/'request.json')
    for name,path in [('baseline',a.outputs/'experiment_v1'/('baseline_'+label)),('upper8',target)]:
        phys=read(path/'physical.json');g=read(path/'generation.json');gate=phys['absolute'];audit=phys['audit']
        x=np.load(path/'native.npy');fk=np.load(path/'native_fk.npy');ev=evaluate(x,fk,q['anchors'],q['beat_frames'])
        local_steps=[float(np.linalg.norm(fk[k]-fk[k-1],axis=-1).max()) for anchor in q['anchors'] for k in (anchor['frame'],anchor['frame']+1) if 0<k<len(fk)]
        rows.append(dict(case=label,mode=name,frames=len(x),windows=g['windows'],anchors=[v['frame'] for v in q['anchors']],
            motion_sha256=sha(path/'native.npy'),absolute_ok=gate['ok'],reasons=gate['reasons'],fk_max_abs_m=phys['fk_max_abs_m'],
            matched_anchor_mean_rad=float(np.mean([v['mean_radians'] for v in ev['anchor_errors']])),
            matched_anchor_max_rad=max(v['max_radians'] for v in ev['anchor_errors']),max_joint_step_near_anchors_m=max(local_steps),
            raw_normalized_denoiser_anchor_error_last=g['sampler'].get('raw_denoiser_anchor_max_abs_by_step',[None])[-1],
            raw_mixed_channel_overlap_max_abs=g['raw_window_overlap_max_abs'],window_boundaries=ev['window_boundaries'],
            beat_minimum_proximity_s=ev['beat_alignment_mean_seconds'],foot_skate_p95_mps=audit['foot_skate_mps_p95'],
            joint_jerk_max_mps3=audit['joint_jerk_mps3_max'],root_net_displacement_m=audit['root_horizontal_net_displacement_m'],
            weights_unchanged=g['weights_before']==g['weights_after']))
a.report.parent.mkdir(parents=True,exist_ok=True)
write(a.report,dict(baseline_generation_commit='69762598121770a48de12450ec09ef2aa295ee37',upper8_generation_commit=read(target/'request.json')['code_commit'],
    edge_commit=q['edge_commit'],native_commit=q['legacy_commit'],checkpoint_sha256=q['checkpoint']['sha256'],publisher_origin_verified=False,
    audio_sha256=q['audio']['sha256'],features_sha256=q['features']['sha256'],seed=q['seed'],n_songs=1,n_seeds=1,
    rejected=rejected,rows=rows,original_ten_joint_goal_met=False,style_independently_evaluated=False,full_generation_qualified=False))
for row in rows:print(row['case'],row['mode'],*[round(row[k],6) for k in ['matched_anchor_mean_rad','matched_anchor_max_rad','foot_skate_p95_mps','joint_jerk_max_mps3','max_joint_step_near_anchors_m','fk_max_abs_m','beat_minimum_proximity_s']])
