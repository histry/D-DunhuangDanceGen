"""Publish aggregates and SHA-bound receipts; never publish source/model arrays."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.planning.time_map import source_positions
from dunhuang.generation.soft_reference import resample_native
from .core import Library,read,write,sha

def run(cfg,root):
    root=Path(root);public=root/'public';public.mkdir(exist_ok=False);lib=Library(cfg['catalog'])
    plan=read(root/'planning/result.json');checks=read(root/'independent_checks.json');train=read(root/'context/result.json');replay=read(root/'context_replay/result.json');generation=read(root/'generation/result.json');arrays=np.load(root/'generation/arrays.npz')
    assert all(checks['checks'].values()) and train['optimizer_steps']==160 and replay['optimizer_steps']==0
    effects=[]
    for row in generation['records']:
        binding=row['binding'];key=row['key'];start=binding['window_start'];a,b=binding['music_interval'];responsible=np.arange(256)+start
        roi=(responsible>=row['responsible_interval'][0])&(responsible<row['responsible_interval'][1]);core=(responsible>=a)&(responsible<b)
        parent=next(s for s in read(Path(cfg['r35'])/'planning/songs.json') if s['id']==binding['song_id']);p=read(parent['plan']['path']);truth=np.empty((256,22,3,3))
        for e in p['events']:
            ticks=responsible[(responsible>=e['music_interval'][0])&(responsible<e['music_interval'][1])]
            if not len(ticks):continue
            native=resample_native(lib.native[e['source_uid']],source_positions(e,ticks))
            truth[ticks-start]=matrices(native[:,7:].reshape(-1,24,6))[:,:22]
        internal=core[3:]&core[2:-1]&core[1:-2]&core[:-3]
        cross=np.zeros(253,bool)
        for boundary in (a-start,b-start):cross|=(np.arange(253)<boundary)&(np.arange(253)+3>=boundary)
        arms={}
        for arm in ('Frozen','Context','Edit'):
            points=arrays[key+'_'+arm+'_22'];raw=arrays[key+'_'+arm+'_139'];R=matrices(raw[:,:,7:].reshape(-1,22,6),row=True).reshape(2,256,22,3,3)
            error=Rotation.from_matrix((truth[None].swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(2,256,22)
            local=points-points[:,:,:1];jerk=np.linalg.norm(np.diff(local,n=3,axis=1)*27000,axis=-1).mean(-1)
            arms[arm]=dict(mean_angle_rad_editable=float(error[:,roi].mean()),mean_angle_rad_current_event=float(error[:,core].mean()),
                body_jerk_internal_mean=float(jerk[:,internal].mean()),body_jerk_boundary_peak=float(jerk[:,cross].max()),
                generated_foot_horizontal_speed_proxy_m_s=float(np.linalg.norm(np.diff(points[:,:, [7,8,10,11]][:,:,:,[0,2]],axis=1)*30,axis=-1)[:,core[1:]&core[:-1]].mean()),
                **row['responsible_region_metrics'][arm])
        direct={}
        for arm in ('Context','Edit'):
            delta=arrays[key+'_'+arm+'_22']-arrays[key+'_Frozen_22'];direct[arm]=float(np.sqrt(np.mean(delta[:,roi]**2)))
        effects.append(dict(key=key,source_uid=binding['source_uid'],seeds=binding['seeds'],arms=arms,direct_world_xyz_rmse_vs_frozen_m=direct,
            context_and_edit_combined=True,source_retention_partly_enforced=True,not_causal_ablation=True,foot_proxy_is_not_contact_ground_truth=True))
    receipt_dirs=['complete_r36_plan_receipt_01','complete_r36_dispatch_receipt_01','complete_r36_preview_receipt_01'];receipts={}
    for name in receipt_dirs:
        folder=Path('outputs')/name;receipts[name]={}
        for p in sorted(folder.iterdir()):
            if p.is_file():receipts[name][p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
        receipts[name]['exit_status']=(folder/'exit_status.txt').read_text().strip()
    assert receipts['complete_r36_preview_receipt_01']['exit_status']=='0' and receipts['complete_r36_dispatch_receipt_01']['exit_status']=='0'
    test_log=(Path('outputs/complete_r36_preview_receipt_01/tests.log')).read_text()
    summary=dict(schema='r36_public_release',release_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        planning_execution_commit=plan['code_commit'],main_execution_commit=generation['code_commit'],units=plan['units'],families=plan['families'],source_counts=plan['source_counts'],
        planning=checks['songs'],effects=effects,training=dict(optimizer_steps=160,replay_optimizer_steps=0,snapshots_verified=len(train['observations']),final_digest=train['final_digest'],task_contract=train['task_contract'],cache_policy=train['cache_policy']),
        regression_summary=test_log.splitlines()[-1],new_main_trajectories=generation['new_generated_trajectories'],seeds_per_case=2,
        additional_ablation_calls=0,whole_song_generations=0,sdmr_route=False,raw_deletions=0,
        source_semantic_completeness='UNVERIFIED_KINEMATIC_PROPOSALS',full_native_status='NOT_EVALUABLE_22_OF24',operational_qualified=False,
        limitations=['New long-unit plans are previews; main generation keeps the exact R35 plan for attribution.',
            'Rotation retention is explicitly bounded, not evidence of learned music reinterpretation.',
            'Lower planning speed remains a tradeoff; larger extent is affected by longer unit duration.',
            'Planned context training uses same-recording references, not cross-source paired target truth.',
            'Historical music and adaptation sources are exposed development populations.'])
    write(public/'r36_summary.json',summary);write(public/'r36_checks.json',checks)
    frozen={'planning','planning_replay','context','context_replay','generation','generation_replay'}
    frozen.update(p.name for p in root.glob('viewer*') if (p/'protocol.json').exists())
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file() and (p.relative_to(root).parts[0] in frozen or p.name=='independent_checks.json')}
    write(root/'artifacts_manifest.json',dict(schema='r36_archive_sha256',scope='FROZEN_SCIENTIFIC_ARTIFACTS; ACTIVE_RELEASE_LOGS_EXCLUDED',files=files))
    write(public/'r36_receipt.json',dict(archive_manifest_sha256=sha(root/'artifacts_manifest.json'),file_count=len(files),receipts=receipts,
        inputs=dict(source_catalog_sha256=sha(cfg['catalog']),parent_generation_sha256=generation['parent_arrays_sha256']),
        failed_initial_view_retained=True,code_sync='PUBLISHED_GITHUB_SHA_VERIFIED; SSH_GIT_BUNDLE_RELAY_AFTER_SERVER_TLS_FAILURE',
        ssh_temp_cleanup='VERIFY_AFTER_FINAL_SYNC',media_copy_status_at_release='NOT_YET_COLLECTED'))
    print('R36_PUBLIC',summary['regression_summary'],len(files),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(read('src/dunhuang/research/complete_r36/config.json'),a.root)
