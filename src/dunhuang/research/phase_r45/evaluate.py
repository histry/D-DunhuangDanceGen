"""Train-only music proxy gate, same-knot factor analysis and conditional target audit."""
import argparse,collections
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.reperformance_r44.motion import decode
from dunhuang.research.sdmr_event_rank.evaluate import group_ci

def run(root,out):
    root=Path(root);meta=read(root/'result.json');cfg=meta['config'];z=Arrays(root/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];rows=[];groups=collections.defaultdict(lambda:collections.defaultdict(list));change=[]
    for r in meta['records']:
        key=r['key'];m=r['metrics']['methods'];base=m['Geometry'];p=m['Phase'];ratio=p['core_activity']/max(base['core_activity'],1e-12);d3=p['core_D3']/max(base['core_D3'],1e-12);passed=.85<=ratio<=1.15 and d3<=1.05 and p['protected_exact'];factor={}
        assert np.array_equal(z[key+'_MatchedC2_139'],z[key+'_R44C2_139']);assert np.array_equal(z[key+'_Phase_139'],z[key+'_FixedMapMusicB_139']);assert np.array_equal(z[key+'_NoMusic_139'],z[key+'_Geometry_139']);assert np.array_equal(z[key+'_Geometry_139'][0],z[key+'_Geometry_139'][1])
        for method in ('MatchedLinear','MatchedC2'):
            P=relative_fk(decode(z[key+'_'+method+'_139'][0])[0],J,parents);at=[];inside=[]
            for k in r['matched_anchor_maps']:
                a,b=k['interval'];jerk=np.diff(P[a:b],n=3,axis=0)*27000;starts=np.arange(a,b-3);knots=np.asarray(k['target'][1:-1]);mask=np.any((starts[:,None]<=knots[None])&(starts[:,None]+3>=knots[None]),axis=1) if len(knots) else np.zeros(len(starts),bool);at.extend(jerk[mask]);inside.extend(jerk[~mask])
            rms=lambda v:float(np.sqrt(np.mean(np.asarray(v)**2))) if len(v) else None;factor[method]=dict(knot_stencil_D3=rms(at),interior_D3=rms(inside),all_core_D3=m[method]['core_D3'])
        source=r['source'];rows.append(dict(key=key,domain=source['domain'],role=source['role'],activity_ratio=ratio,D3_ratio=d3,source_feasibility_filter_passed=passed,identity_cores=sum(c['identity'] for c in r['phase_maps']),factor=factor,metrics=m))
        if source['domain']=='ChangE' and source['role']=='train':change.append(passed)
        if source['domain']=='AIST++' and source['role']=='train':
            group=source['music_identity']['source']['sha256']
            for metric in ('rms_timing_seconds','wrist_acceleration_timing_seconds'):
                for control in ('WrongMusic','NoMusic'):
                    a=p[metric];b=m[control][metric]
                    if a is not None and b is not None:groups[metric+'/'+control][group].append(b-a)
    comparisons={}
    for metric in ('rms_timing_seconds','wrist_acceleration_timing_seconds'):
        for control in ('WrongMusic','NoMusic'):
            key=metric+'/'+control;g={k:float(np.mean(v)) for k,v in groups[key].items()};ci=group_ci(list(g.values()),cfg['seed'],cfg['bootstrap_draws']);comparisons[key]=dict(group_mean_improvement_seconds=g,evaluable_music_groups=len(g),registered_music_groups=8,ci95=ci,passed=len(g)>=6 and ci[0] is not None and ci[0]>0)
    music_gate=all(v['passed'] for v in comparisons.values());screen=music_gate and sum(change)>=3
    # Physical clocks are real weak labels, not automatically exact cross-performance pose supervision.
    manifest_path=Path('outputs/paired_event_r41_01/data/manifest.json');data=read(manifest_path);train=[r for r in data['rows'] if r['role']=='train'];pairs=[];seen=set()
    for i,a in enumerate(train):
        for b in train[i+1:]:
            if a['audio_sha256']!=b['audio_sha256'] or a['raw_motion_sha256']==b['raw_motion_sha256']:continue
            A=a['audio_frame_interval'];B=b['audio_frame_interval'];support=[max(A[0],B[0]),min(A[1],B[1])]
            if support[1]-support[0]<90 or not a['paired_clock_verified'] or not b['paired_clock_verified']:continue
            identity=(a['raw_motion_sha256'],b['raw_motion_sha256'],tuple(support))
            if identity in seen:continue
            seen.add(identity);pairs.append(dict(source_event_uid=a['event_uid'],target_event_uid=b['event_uid'],source_raw_sha256=a['raw_motion_sha256'],target_raw_sha256=b['raw_motion_sha256'],audio_sha256=a['audio_sha256'],shared_audio_support=support,same_choreography=a['choreography_uid']==b['choreography_uid'],source_root_scope=a['root_scope'],target_root_scope=b['root_scope'],stage_truth=[a['music_motion_phase_truth'],b['music_motion_phase_truth']],scope='VERIFIED_PHYSICAL_CLOCK_NOT_EXACT_REFERENCE_TARGET_REPERFORMANCE_TRUTH'))
    contract=dict(candidate_pairs=len(pairs),manifest_sha256=sha(manifest_path),exact_reference_target_pose_correspondence_verified=False,stage_correspondence_verified=False,training_target_status='UNVERIFIED_CROSS_PERFORMANCE_REPERFORMANCE_TARGET',pairs=pairs)
    out=Path(out)
    if out.exists():raise ValueError('Fresh R45 qualification')
    out.mkdir(parents=True);write(out/'reference_target_audit.json',contract);write(out/'result.json',dict(records=rows,train_gate=dict(music=music_gate,comparisons=comparisons,change_sources_passed=sum(change),change_sources=len(change),source_pass_is_by_construction=True,necessary_screen_passed=screen),conditional_training=dict(admitted=bool(screen and contract['exact_reference_target_pose_correspondence_verified'] and contract['stage_correspondence_verified']),reason='NO_CREDIBLE_MUSIC_PROXY_BENEFIT' if not screen else 'CROSS_PERFORMANCE_REFERENCE_TARGET_CORRESPONDENCE_NOT_VERIFIED'),input_arrays_sha256=sha(root/'arrays.npz'),paired_target_audit_sha256=sha(out/'reference_target_audit.json'),geometry_duplicates_not_independent=True,native24='NOT_EVALUABLE_22_OF24',contact_truth='UNKNOWN',human_reviewed=False,optimizer_steps=0,whole_song_generated=False,can_route=False,operational_qualified=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
