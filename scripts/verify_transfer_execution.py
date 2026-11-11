"""Fresh Native replays and independent numeric measurements; no sampling."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,write,sha
from dunhuang.generation.train import phase_target
from dunhuang.research.event_metrics import identity_probe,representation_stats,dynamics,event_margin,event_timing


def read(path):return json.loads(Path(path).read_text())
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def verify_mechanism(root):
    report=read(root/'diagnostics.json');protocol=read(root/'protocol.json')
    saved=read(root/'status.json')
    if saved['status']!='COMPLETED_MECHANISM_DIAGNOSTIC_UNQUALIFIED' or saved['optimizer_calls']!=0:
        raise ValueError('Mechanism experiment did not complete')
    if saved['model_before']!=saved['model_after'] or saved['adapter_before']!=saved['adapter_after']:
        raise ValueError('Frozen parameter receipts differ')
    probe_results={}
    with np.load(root/'identity_features.npz') as arrays:
        for k,expected in report['identity_probes'].items():
            actual=dict(identity_probe(arrays[k],arrays['labels'],arrays['views']),**representation_stats(arrays[k]))
            for name in ('accuracy','effective_rank','centered_rms'):
                if not np.isclose(actual[name],expected[name],rtol=1e-8,atol=1e-10):
                    raise ValueError('Identity measurements changed: '+k)
            probe_results[k]=actual
    rows={r['window_uid']:r for r in protocol['selected_windows']}
    assets=read(checked(protocol['assets']))
    with np.load(checked(assets['scaler'])) as s:scale,bias=s['scale'],s['bias']
    targets={k:phase_target(np.load(checked(v['motion'])),scale,bias)[0] for k,v in rows.items()}
    groups={}
    with np.load(root/'denoiser_predictions.npz') as arrays:
        values=arrays['predictions']
        if len(values)!=len(report['denoiser']):raise ValueError('Missing denoiser observations')
        for x,r in zip(values,report['denoiser']):
            target=targets[r['window_uid']]
            actual=float(np.mean((x-target)**2))
            if not np.isclose(actual,r['full_mse'],rtol=1e-6,atol=1e-8):
                raise ValueError('Denoiser MSE mismatch')
            key=(r['seed'],r['timestep'],r['window_uid'])
            groups.setdefault(key,[]).append(r)
    for values in groups.values():
        if len(values)!=8 or len({r['noise_sha256'] for r in values})!=1:
            raise ValueError('Intervention noises differ')
    generated=read(root/'generation.json');noise_groups={}
    for case in generated['cases']:
        folder=Path(checked(case['motion'])).parent
        metric=dynamics(np.load(checked(case['motion'])),np.load(checked(case['fk'])))
        for key in ('jerk_peak_mps3','joint_speed_mean_mps','upper8_angular_speed_mean_radps'):
            if not np.isclose(metric[key],case['metrics'][key],rtol=1e-6,atol=1e-7):
                raise ValueError('Saved physical-unit diagnostics differ')
        if metric['jerk_peak_mps3']!=case['physical']['audit']['joint_jerk_mps3_max']:
            raise ValueError('Four-frame jerk does not match Native operator')
        raw=dynamics(np.load(folder/'raw_native.npy'),np.load(folder/'raw_fk.npy'))
        if not np.isclose(raw['jerk_peak_mps3'],case['raw_window_metrics']['jerk_peak_mps3'],rtol=1e-6):
            raise ValueError('Saved preassembly jerk differs')
        key=(case['seed'],case['music_index'])
        noise_groups.setdefault(key,[]).append(case['sampler']['noise_sha256'])
    if len(generated['cases'])!=12 or any(len(v)!=3 or len(set(v))!=1 for v in noise_groups.values()):
        raise ValueError('Registered factorial budget/noise matching differs')
    refs=list(rows.values())
    second=next(r for r in refs if r['music_uid']!=refs[0]['music_uid'])
    e1,e2=[np.load(checked(r['motion'])) for r in (refs[0],second)]
    for expected in generated['event_preference']:
        pair=[next(c for c in generated['cases'] if c['seed']==expected['seed'] and
                   c['music_index']==expected['music_index'] and c['event_index']==i) for i in (0,1)]
        metric=event_margin(*(np.load(checked(c['motion'])) for c in pair),e1,e2)
        if abs(metric['preference_margin_radians']-expected['preference_margin_radians'])>1e-10:
            raise ValueError('SO3 event preference did not replay')
    return dict(identity_measurements=probe_results,denoiser_observations=len(report['denoiser']),
                matched_noise_groups=len(groups),factorial_cases_verified=12,
                input_protocol=ref(root/'protocol.json'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True);ap.add_argument('--experiment',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    experiment=Path(a.experiment).resolve();baseline=Path(a.baseline).resolve()
    if read(experiment/'status.json')['status']!='COMPLETED_BOUNDED_EVENT_TRANSFER_RESEARCH':
        raise ValueError('Registered comparison incomplete')
    results=read(experiment/'results.json');proto=read(experiment/'protocol.json')
    if results['cross']['initialization']!=results['additive']['initialization']:
        raise ValueError('Parameter/initialization matching failed')
    for mode in ('cross','additive'):
        root=experiment/mode
        if read(root/'validation_replay.json')['status']!='READ_ONLY_EVENT_TRANSFER_REPLAY_VERIFIED':
            raise ValueError('Fresh neural validation replay missing')
        if read(root/'train/protocol.json')['code_commit']!=proto['code_commit']:
            raise ValueError('Training used a different code revision')
    # Zero-Adapter trajectories must remain bitwise equal across instrumentation/fusion.
    base_cases={c['case']:c for c in read(baseline/'generation.json')['cases'] if c['event_index']==-1}
    for mode in ('cross','additive'):
        for c in read(experiment/mode/'mechanism/generation.json')['cases']:
            if c['event_index']==-1:
                b=base_cases[c['case']]
                if c['motion']['sha256']!=b['motion']['sha256'] or c['sampler']['noise_sha256']!=b['sampler']['noise_sha256']:
                    raise ValueError('Unconditioned sampler baseline changed')
    mechanism_reports={};jobs=[]
    for name,root in [('baseline',baseline),('cross',experiment/'cross/mechanism'),('additive',experiment/'additive/mechanism')]:
        mechanism_reports[name]=verify_mechanism(root)
        for c in read(root/'generation.json')['cases']:
            checked(c['motion']);checked(c['fk'])
            jobs.append((name+'_'+c['case'],Path(c['motion']['path']).parent))
    for mode in ('cross','additive'):
        for case in ('original','replaced'):
            jobs.append((mode+'_dunhuang_'+case,experiment/mode/('dunhuang_'+case)))
    assets=read(checked(proto['generation_assets']));env=os.environ.copy()
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    replays=[]
    for label,source in jobs:
        dest=out/label;dest.mkdir()
        for name in ('edge.npy','official_fk.npy'):shutil.copyfile(source/name,dest/name)
        q=dict(assets,frames=150,output=str(dest),anchors=[],beat_frames=[])
        write(dest/'request.json',q)
        with (dest/'native.log').open('w') as f:
            subprocess.run([assets['native_python'],'-m','dunhuang.physical',str(dest/'request.json')],
                           env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        expected=read(source/'physical.json');actual=read(dest/'physical.json')
        if expected!=actual or sha(source/'native.npy')!=sha(dest/'native.npy') or sha(source/'native_fk.npy')!=sha(dest/'native_fk.npy'):
            raise ValueError('Fresh Native replay differs: '+label)
        stages=[]
        snapshots=source/'denoising_stages.npz'
        if snapshots.exists():
            # Native FK uses the same frame clock; stages are noised-denoiser predictions, not accepted motion.
            # Run separate namespaces: stage FK is delegated to a Native subprocess below.
            stage_request=dict(assets,stages=ref(snapshots),scaler=assets['scaler'],output=str(dest/'stages.json'))
            write(dest/'stages_request.json',stage_request)
            with (dest/'stages.log').open('w') as f:
                subprocess.run([assets['native_python'],str(Path(__file__).resolve()),'--stage-worker',
                                str(dest/'stages_request.json')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            stages=read(dest/'stages.json')
        replays.append(dict(case=label,physical=actual,native=ref(dest/'native.npy'),fk=ref(dest/'native_fk.npy'),
                            original_native=ref(source/'native.npy'),denoising_stage_dynamics=stages))
        write(out/'progress.json',dict(completed=len(replays),expected=len(jobs)))
    paired_assets=read(checked(proto['paired_assets']))
    manifest=read(checked(paired_assets['paired_generation_manifest']))
    by_uid={r['window_uid']:r for r in manifest['rows']}
    validation_pairs=read(experiment/'cross/train/protocol.json')['validation_pairs']
    targets=[by_uid[p['target_uid']] for p in validation_pairs]
    if any(r['role']!='val' for r in targets):raise ValueError('Target audit role changed')
    target_request=dict(assets,targets=targets,output=str(out/'target_physics.json'))
    write(out/'target_request.json',target_request)
    with (out/'target_audit.log').open('w') as f:
        subprocess.run([assets['native_python'],str(Path(__file__).resolve()),'--target-worker',
                        str(out/'target_request.json')],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(out/'verification.json',dict(status='FRESH_NATIVE_AND_MEASUREMENT_REPLAYS_VERIFIED',
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        experiment_protocol=ref(experiment/'protocol.json'),mechanism_measurements=mechanism_reports,
        native_replays=replays,native_replay_count=len(replays),target_physics=read(out/'target_physics.json'),
        optimizer_calls=0,new_sampling_calls=0,new_weight_updates=0,zero_adapter_sampler_exact_across_methods=True,
        music_qualified=False,full_motion_qualified=False))
    print('VERIFIED_TRANSFER_EXECUTION',len(replays),flush=True)


def stage_worker(request):
    from dunhuang.io import pinned
    from dunhuang.contracts import MotionAdapter
    q=read(request);sys.path.insert(0,str(pinned(q['legacy_root'],q['legacy_commit'])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.geometry import fk_24_np
    with np.load(checked(q['scaler'])) as s:scale,bias=s['scale'],s['bias']
    result=[]
    with np.load(checked(q['stages'])) as arrays:
        for step in sorted(arrays.files,key=int):
            try:
                raw=(arrays[step][0]-bias)/scale
                native=MotionAdapter.to_native(raw)
                result.append(dict(reverse_step=int(step),status='MEASURED',**dynamics(native,fk_24_np(native))))
            except ValueError as error:
                result.append(dict(reverse_step=int(step),status='INVALID_ROTATION_GEOMETRY',error=str(error)))
    write(q['output'],result)

def target_worker(request):
    from dunhuang.io import pinned
    q=read(request);sys.path.insert(0,str(pinned(q['legacy_root'],q['legacy_commit'])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
    from motion.geometry import fk_24_np
    from motion.physics import audit_motion_np
    from contracts.physical_quality import evaluate_physical_audit
    cases=[]
    for r in q['targets']:
        native=np.load(checked(r['motion']));fk=fk_24_np(native)
        audit=audit_motion_np(native,precomputed_joints=fk)
        cases.append(dict(window_uid=r['window_uid'],role=r['role'],motion=r['motion'],
            physical=evaluate_physical_audit(audit),dynamics=dynamics(native,fk)))
    write(q['output'],dict(scope='fixed8 fitted Val target diagnostics; not training/selection filtering or target admission',
        cases=cases,native_passes=sum(c['physical']['ok'] for c in cases),optimizer_calls=0))

if __name__=='__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--stage-worker':stage_worker(sys.argv[2])
    elif len(sys.argv)==3 and sys.argv[1]=='--target-worker':target_worker(sys.argv[2])
    else:main()
