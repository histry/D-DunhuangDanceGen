"""Read-only receipt validation and intervention readout; optional fresh network run."""
import argparse,json,subprocess,sys,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import sha,write
from dunhuang.research.network import GROUPS
from dunhuang.generation.diagnostics import reference_error,upperbody_activity
from dunhuang.evaluation import evaluate

def read(p):return json.loads(Path(p).read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out')
    ap.add_argument('--network-replay-out')
    args=ap.parse_args();root=Path(args.root).resolve()
    if args.out and Path(args.out).exists():raise FileExistsError(args.out)
    status=read(root/'status.json');assert status['status']=='COMPLETED_DIAGNOSTIC_ONLY'
    protocol=read(root/'protocol.json');assets=read(protocol['assets']['path'])
    assert sha(protocol['assets']['path'])==protocol['assets']['sha256']
    assert sha(protocol['adapter']['path'])==protocol['adapter']['sha256']
    support=read(root/'support/support.json');network=read(root/'network/network.json')
    masks=np.load(root/'support/support_windows.npz')['channel_supported']
    assert masks.shape==(support['windows'],151)
    for c in support['channels']:assert int(masks[:,c['index']].sum())==c['supported_windows']
    assert network['model_state_before']==network['model_state_after']
    assert network['adapter_state_before']==network['adapter_state_after']
    assert network['optimizer_steps']==network['weight_updates']==0
    assert all(r['hidden_reference_max']==0 for values in network['vjp'].values() for r in values)
    predictions=np.load(root/'network/denoiser_predictions.npz')['predictions']
    variants=network['variants'];assert predictions.shape==(6,len(variants),150,151)
    assert np.isfinite(predictions).all()
    original=predictions[:,variants.index('original')]
    zero=predictions[:,variants.index('zero_adapter')]
    readout=dict(protocol=protocol,support=support,geometry=read(root/'geometry/geometry.json'),
        network=network,denoiser_intervention_vs_original={},adapter_effect_vs_zero={},generation=[],replay={})
    for variant in variants:
        delta=predictions[:,variants.index(variant)]-original
        readout['denoiser_intervention_vs_original'][variant]={name:float(np.sqrt(np.mean(delta[:,:,ids]**2,axis=(1,2))).mean()) for name,ids in GROUPS.items()}
    readout['adapter_effect_vs_zero']={name:float(np.sqrt(np.mean((original-zero)[:,:,ids]**2,axis=(1,2))).mean()) for name,ids in GROUPS.items()}
    generated=read(root/'generation.json');prior=Path(protocol['prior'])
    old_limits=read(prior/'probe150/physical.json')['limits']
    for row in generated['cases']:
        folder=root/row['case'];q=read(folder/'request.json');p=read(folder/'physical.json')
        assert sha(folder/'native.npy')==row['motion_sha256']
        assert p['limits']==old_limits and p['legacy_commit']==assets['legacy_commit']
        assert p['fk_max_abs_m']<1e-5
        g=read(folder/'generation.json');assert g['weights_before']==g['weights_after']
        assert g['training'] is False
        plan=read(folder/'event_plan.json') if (folder/'event_plan.json').exists() else read(root/'plans/original.json')
        native=np.load(folder/'native.npy');fk=np.load(folder/'native_fk.npy')
        timeline=read(prior/'probe150/music_timeline.json')
        with np.load(assets['scaler']['path']) as scaler:
            follow=reference_error(native,plan,scaler['scale'],scaler['bias'])
        readout['generation'].append(dict(row,source_following=follow,activity=upperbody_activity(native),
            music_diagnostic=evaluate(native,fk,[],timeline['beats'])))
    assert sha(root/'official150/native.npy')==sha(prior/'official150/native.npy')
    assert sha(root/'temporal150/native.npy')==sha(prior/'temporal150/native.npy')
    assert sha(root/'dense_s1/native.npy')==sha(prior/'probe150/native.npy')
    readout['preserved_original_paths_exact']=True
    readout['final_sample_interventions']={}
    for original_name,changed_name in [('dense_s1','matched_s1'),('dense_s1','replaced_s1'),('dense_s1','warp_s1'),('dense_s2','replaced_s2')]:
        assert read(root/changed_name/'generation.json')['sampler']['noise_sha256']==read(root/original_name/'generation.json')['sampler']['noise_sha256']
        a=np.load(root/original_name/'normalized.npy');b=np.load(root/changed_name/'normalized.npy')
        readout['final_sample_interventions'][changed_name]=dict(reference=original_name,normalized_rms=float(np.sqrt(np.mean((a-b)**2))),
            byte_identical=sha(root/original_name/'native.npy')==sha(root/changed_name/'native.npy'))
    if args.network_replay_out:
        q=protocol;out=Path(args.network_replay_out).resolve()
        subprocess.run([assets['edge_python'],'-m','dunhuang.research.v07','--stage','network',
            '--assets',q['assets']['path'],'--prior',str(prior),'--plans',str(root/'plans/index.json'),'--out',str(out)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),check=True)
        repeated=np.load(out/'denoiser_predictions.npz')['predictions']
        assert np.array_equal(predictions,repeated),'Denoiser forward replay differs'
        again=read(out/'network.json')
        assert again['model_state_before']==again['model_state_after']==network['model_state_before']
        assert again['adapter_state_before']==again['adapter_state_after']==network['adapter_state_before']
        differences=[]
        for group in network['vjp']:
            for a,b in zip(network['vjp'][group],again['vjp'][group]):
                for key in a:
                    differences.append(abs(a[key]-b[key]))
                    assert np.isclose(a[key],b[key],rtol=1e-4,atol=1e-10),(group,key)
        readout['replay']=dict(denoiser_predictions_exact=True,gradient_absolute_error_max=max(differences),
            optimizer_steps=0,model_and_adapter_unchanged=True,path=str(out))
    if args.out:write(Path(args.out),readout)
    print('VERIFIED',len(readout['generation']),'Native cases;',support['windows'],'source windows; no optimization.')
if __name__=='__main__':main()
