"""Read-only all-source hash checks, aggregation and six numerical window replays."""
import argparse,sys,pickle,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,pinned,write,sha
from dunhuang.data.sources import read,ref,raw_path,validate_links
from dunhuang.data.conversion import decode,contacts,normalized_window
from dunhuang.data.paired_manifest import inspect
from dunhuang.contracts import MotionAdapter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--audit',required=True)
    ap.add_argument('--raw-root',required=True);ap.add_argument('--features',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    assets=read(a.assets);root=Path(a.raw_root);audit=Path(a.audit);feature=Path(a.features)
    report=read(audit/'report.json');pop=read(checked(report['population']))
    validate_links(pop['records']);checked(assets['checkpoint'])
    scaler_path=checked(assets['scaler']);producer=read(scaler_path.with_name('receipt.json'))
    assert producer['checkpoint']['sha256']==assets['checkpoint']['sha256']
    assert producer['scaler']['sha256']==assets['scaler']['sha256'] and producer['normalizer_refitted'] is False
    rows=read(checked(report['window_ledger']));sources=read(checked(report['source_ledger']))
    with np.load(checked(report['support'])) as s:
        supported=s['channel_supported'];errors=s['max_normalized_excess']
    assert supported.shape==errors.shape==(len(rows),151) and supported.dtype==bool
    np.testing.assert_array_equal(supported,errors<=32*np.finfo(np.float32).eps)
    by_source={r['pair_uid']:r for r in sources};population={r['pair_uid']:r for r in pop['records']}
    expected=[];verified=0
    for row in pop['records']:
        path=raw_path(root,row['motion_asset']);verified+=1
        if row['purpose_role']=='excluded_official_ignore':
            assert by_source[row['pair_uid']]['status']=='EXCLUDED_OFFICIAL_IGNORE';continue
        if row['dataset_id']=='aistpp':
            with path.open('rb') as f:n=(len(pickle.load(f)['smpl_poses'])+1)//2
        else:n=np.load(path,mmap_mode='r').shape[0]
        assert n==by_source[row['pair_uid']]['frames_30hz']
        expected.extend((row['pair_uid']+':'+str(start),row['purpose_role'],row['group_uid'])
                        for start in range(0,n-149,75))
    assert expected==[(r['window_uid'],r['role'],r['source_group']) for r in rows]
    for dataset in ('aistpp','finedance'):
        indices=[i for i,r in enumerate(rows) if r['dataset']==dataset]
        part=supported[indices];summary=report['datasets'][dataset]
        assert summary['numerical_full151']==int(part.all(1).sum())
        assert summary['unsupported_window_counts']==(~part).sum(0).tolist()
        assert summary['worst_normalized_excess']==errors[indices].max(0).tolist()
        assert summary['representation_eligible']==sum(r['representation_eligible'] for r in rows if r['dataset']==dataset)
    edge=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(edge))
    import torch
    from vis import SMPLSkeleton
    torch.set_num_threads(4);skeleton=SMPLSkeleton('cpu')
    with np.load(scaler_path) as s:scale,bias=s['scale'],s['bias']
    selected={}
    for i,r in enumerate(rows):selected.setdefault((r['dataset'],r['role']),i)
    replay=[]
    for key,index in sorted(selected.items()):
        record=rows[index];source=population[record['pair_uid']]
        native,observed=decode(raw_path(root,source['motion_asset']),record['dataset'])
        native[:,:4]=contacts(MotionAdapter.to_edge(native),skeleton)
        _,n,v=normalized_window(native,record['start_frame'],scale,bias)
        np.testing.assert_array_equal(v.all(0),supported[index])
        np.testing.assert_array_equal(np.maximum(abs(n)-1,0).max(0),errors[index])
        replay.append(dict(dataset=key[0],role=key[1],window_uid=record['window_uid'],support_exact=True,excess_exact=True))
    prepared=read(feature/'assets.json');_,_,preflight=inspect(prepared)
    assert preflight==read(feature/'preflight.json') and not preflight['ready']
    manifest=read(checked(prepared['paired_generation_manifest']))
    motion_replay=[];extractor=sha(edge/'data/audio_extraction/jukebox_features.py')
    for row in manifest['rows']:
        receipt=read(checked(row['jukebox_receipt']))
        assert receipt['extractor_sha256']==extractor and not receipt['silence_padding']
        assert row['role']=='train'
        native,_=decode(checked(row['raw_motion']),row['dataset'])
        native[:,:4]=contacts(MotionAdapter.to_edge(native),skeleton)
        target,_,valid=normalized_window(native,row['start_frame'],scale,bias)
        np.testing.assert_array_equal(target,np.load(checked(row['motion'])))
        assert valid.all()
        motion_replay.append(dict(window_uid=row['window_uid'],native_target_exact=True,
                                  jukebox=ref(checked(row['jukebox'])),jukebox_reextracted=False))
    result=dict(status='VERIFIED_READ_ONLY',raw_motions_sha_verified=verified,
        decoded_sources=len(sources)-sum(r['status']=='EXCLUDED_OFFICIAL_IGNORE' for r in sources),
        windows=len(rows),channel_aggregation_exact=True,window_identity_and_roles_exact=True,
        numerical_window_replays=replay,candidate_motion_replays=motion_replay,
        paired_preflight=preflight,scaler_producer=ref(scaler_path.with_name('receipt.json')),
        all_original_targets_unmodified=True,optimizer_steps=0,training_steps=0,
        native_acceptance_evaluated=False,full_generation_qualified=False)
    write(out,result);print('VERIFIED',verified,'raw motions,',len(rows),'windows,',len(replay),'numeric replays')
if __name__=='__main__':main()
