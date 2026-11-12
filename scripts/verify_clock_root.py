"""Fresh source/measurement replay: no optimizer and no neural resampling."""
import argparse,json,sys,subprocess,os,hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,sha,write
from dunhuang.alignment.clock_data import read,prepare,encode,ref
from dunhuang.alignment.clock_experiment import digest,evaluate
from dunhuang.alignment.transport import ClockNullTransport
from dunhuang.alignment.model import RetrievalModel
from dunhuang.evaluation_benchmarks.native_shadow import LegacyBackend
from dunhuang.research.root_diagnosis import decompose

def main():
    p=argparse.ArgumentParser()
    for k in ('assets','clock','root','diagnosis','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    assets=read(a.assets);clock=Path(a.clock);root=Path(a.root);diagnosis=Path(a.diagnosis)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    protocol=read(clock/'protocol.json');global_ref=protocol['global_checkpoint'];sync_ref=protocol['sync_checkpoint']
    rows,arrays=prepare(assets,global_ref,sync_ref,out/'fresh_data')
    with np.load(clock/'data/observations.npz') as previous:
        assert all(np.array_equal(v,previous[k]) for k,v in arrays.items())
    base=RetrievalModel().cuda().eval();base.load_state_dict(torch.load(checked(global_ref),map_location='cuda',weights_only=True)['state'])
    before=digest(base);encoded=encode(base,arrays,out/'encodings.npz','cuda')
    with np.load(clock/'encodings.npz') as prior:assert all(np.array_equal(v,prior[k]) for k,v in encoded.items())
    bank={k:torch.from_numpy(v).cuda() for k,v in encoded.items() if k!='tcn'}
    val=[i for i,r in enumerate(rows) if r['role']=='val']
    wrong={int(k):v for k,v in read(clock/'wrong_controls.json')['pairs'].items()}
    for mode in ('A','B','C'):
        saved=torch.load(clock/mode/'head.pt',map_location='cuda',weights_only=True);cfg=saved['config']
        head=base.sync if mode=='A' else ClockNullTransport(band_seconds=cfg['band_seconds'],epsilon=cfg['epsilon'],iterations=cfg['sinkhorn_iterations']).cuda()
        head.load_state_dict(saved['state']);head.eval()
        measured,raw=evaluate(head,mode,bank,val,rows,wrong)
        assert measured==read(clock/mode/'validation.json')
        with np.load(clock/mode/'measurements.npz') as prior:
            assert all(np.array_equal(v,prior[k]) for k,v in raw.items() if k!='diagnostics')
        print('FRESH_CLOCK_VERIFIED',mode,flush=True)
    # Native is loaded only after the neural feature work; it remains the same frozen backend.
    backend=LegacyBackend(assets);source_count=0
    for row in read(diagnosis/'source_rows.json'):
        x=np.load(checked(row['motion']))
        assert decompose(x,backend.fk(x))==row['analysis'];source_count+=1
    cases=read(diagnosis/'generated.json')
    for i,row in enumerate(cases):
        x=np.load(checked(row['motion']))
        assert decompose(x,backend.fk(x))==row['original']
        for name,r in row['counterfactuals'].items():
            q=np.load(diagnosis/('case_%03d'%i)/(name+'.npy'));fk=backend.fk(q)
            assert decompose(q,fk)==r['analysis']
            assert backend.gate(backend.audit(q,backend.cfg,precomputed_joints=fk))==r['legacy_engineering_observation']
    generations=read(root/'generation/results.json')
    for i,row in enumerate(generations):
        folder=Path(row['motion']['path']).parent;x=np.load(checked(row['motion']));fk=np.load(folder/'native_fk.npy')
        assert decompose(x,fk)==row['root_diagnostics']
        q=read(folder/'request.json');dest=out/('native_%02d'%i);dest.mkdir()
        for name in ('edge.npy','official_fk.npy'):
            import shutil
            shutil.copyfile(folder/name,dest/name)
        q['output']=str(dest);write(dest/'request.json',q)
        env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
        with (dest/'native.log').open('w') as log:
            subprocess.run([assets['native_python'],'-m','dunhuang.physical',str(dest/'request.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert read(dest/'physical.json')==read(folder/'physical.json')
        assert sha(dest/'native.npy')==row['motion']['sha256']
    for arm in ('full','masked'):
        status=read(root/'result.json')['results'][arm]
        assert status['weights_before']==status['weights_after'];checked(status['checkpoint'])
    assert read(clock/'result.json')['encoder_digest_before']==before
    write(out/'result.json',dict(status='FRESH_CLOCK_ROOT_REPLAY_VERIFIED',source_windows=source_count,
        clock_validation_windows=len(val),clock_methods=['A','B','C'],root_generated_replays=len(generations),
        counterfactual_cases=len(cases),optimizer_calls=0,new_generations=0,operational_qualified=False,
        clock_protocol_sha256=sha(clock/'protocol.json'),root_protocol_sha256=sha(root/'protocol.json')))
    print('CLOCK_ROOT_REPLAY_VERIFIED',flush=True)
if __name__=='__main__':main()
