"""Independent saved-output verification, with optional fresh inference replay."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--replay',action='store_true');ap.add_argument('--out')
    args=ap.parse_args();root=Path(args.root).resolve()
    destination=Path(args.out).resolve() if args.out else root/'readout.json'
    if (args.replay or args.out) and destination.exists():
        raise FileExistsError('Refuse to overwrite verification evidence: '+str(destination))
    status=read(root/'status.json');assert status['status']=='COMPLETED_RESEARCH_UNQUALIFIED'
    training=read(root/'probe/status.json');protocol=read(root/'probe/protocol.json')
    assert training['frozen_model_unchanged'] and training['weights_before']==training['weights_after']
    assert not(set(protocol['train_groups'])&set(protocol['validation_groups']))
    assert sha(root/'probe/adapter.pt')==training['checkpoint_sha256']
    assert sha(root/'probe/protocol.json')==training['protocol_sha256']
    result=dict(training=training,validation=read(root/'probe/validation.json'),split=protocol,
        reference_checks={str(n):read(root/f'reference_{n}/status.json') for n in (150,450,3601)},
        preflights={t:read(root/f'preflight_{t}/status.json') for t in ('motion','paired','upperbody_probe')},
        cases=[],matched={},fresh_replay={},music_qualified=False,full_motion_qualified=False)
    for name in ['official150','temporal150','none150','zero150','probe150','none225','zero225','probe225']:
        folder=root/name;q=read(folder/'request.json');g=read(folder/'generation.json');p=read(folder/'physical.json')
        assert p['fk_max_abs_m']<1e-5
        assert g['weights_before']==g['weights_after'] and g['training'] is False
        assert len(__import__('numpy').load(folder/'native.npy'))==q['frames']
        if g['mode']=='event_guided':assert sha(root/('zero_adapter.pt' if name.startswith('zero') else 'probe/adapter.pt'))==q['adapter']['sha256']
        result['cases'].append(dict(case=name,frames=q['frames'],motion_sha256=sha(folder/'native.npy'),
            generation_commit=q['code_commit'],mode=g['mode'],native=p['absolute'],metrics=p['audit'],
            fk_error_m=p['fk_max_abs_m'],weights_unchanged=True,soft_reference=g.get('soft_reference')))
    from dunhuang.generation.diagnostics import reference_error,upperbody_activity
    from dunhuang.evaluation import evaluate
    import numpy as np
    for row in result['cases']:
        folder=root/row['case'];q=read(folder/'request.json')
        n=q['frames'];source=root/('zero'+str(n))
        plan=read(source/'event_plan.json')
        beats=read(source/'music_timeline.json')['beats']
        with np.load(q['scaler']['path'],allow_pickle=False) as scaler:
            row['source_following']=reference_error(np.load(folder/'native.npy'),plan,scaler['scale'],scaler['bias'])
        row['upperbody_activity']=upperbody_activity(np.load(folder/'native.npy'))
        row['shared_music_diagnostic']=evaluate(np.load(folder/'native.npy'),np.load(folder/'native_fk.npy'),[],beats)
    for n in (150,225):
        folders=[root/(k+str(n)) for k in ('none','zero','probe')]
        hashes=[read(p/'generation.json')['sampler']['noise_sha256'] for p in folders]
        assert len(set(hashes))==1
        assert sha(folders[0]/'native.npy')==sha(folders[1]/'native.npy')
        result['matched'][str(n)]=read(root/f'matched_{n}.json')
        original=root/('probe'+str(n));out=root/('probe'+str(n)+'_replay')
        if args.replay or out.exists():
            if args.replay:
                q=read(original/'request.json')
                with (root/f'replay_{n}.log').open('w') as log:
                    subprocess.run([q['edge_python'],str(ROOT/'scripts/generate.py'),'--config',str(original/'request.json'),'--out',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            files=['native.npy','native_fk.npy','normalized.npy','windows.npy','edge.npy','official_fk.npy']
            comparison={f:sha(original/f)==sha(out/f) for f in files}
            assert all(comparison.values()),comparison
            assert read(original/'physical.json')==read(out/'physical.json')
            result['fresh_replay'][str(n)]=dict(scientific_arrays_equal=comparison,native_audit_exact=True,
                inference_only=True,optimizer_executed=False)
    result['all_generation_native_passes']=sum(r['native']['ok'] for r in result['cases'])
    if args.replay or args.out:write(destination,result)
    # Manifest excludes itself and active outer launcher logs.
    manifest={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    if args.replay:write(root/'manifest.json',manifest)
    print('VERIFIED',len(result['cases']),'generations; Native passes',result['all_generation_native_passes'],'replays',len(result['fresh_replay']))
if __name__=='__main__':main()
