"""Read-only identity, measured-clock, feature and training receipt verification."""
import argparse,sys,json,pickle
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import read,ref,raw_path
from dunhuang.data.aist_authority import authority,identify,calibrate,POLICY
from dunhuang.data.paired_manifest import inspect

def main():
    ap=argparse.ArgumentParser()
    for name in ('assets','raw-root','authority','calibration','index','features','training','out'):ap.add_argument('--'+name,required=True)
    a=ap.parse_args();output=Path(a.out)
    if output.exists():raise FileExistsError(output)
    root=Path(a.raw_root);c=Path(a.calibration);index=Path(a.index);features=Path(a.features);training=Path(a.training)
    auth=authority(a.authority);pop=read(root/'population.json')
    expected={r['pair_uid']:r for r in pop['records'] if r['dataset_id']=='aistpp'}
    ledger=read(c/'source_ledger.json');assert len(ledger)==len(expected) and not read(c/'protocol.json')['smoke']
    counts=Counter();replay_candidates={}
    for record in ledger:
        source=expected[record['pair_uid']];counts[record['status']]+=1
        assert record['role']==source['purpose_role'] and record['source_group']==source['group_uid']
        identity=identify(source,auth)
        assert record['recording_uid']==identity['recording_uid'] if 'recording_uid' in record else identity['status'].startswith('EXCLUDED')
        if record['status']=='VERIFIED_PAIR':
            assert record['authority_sha256']==auth['identity']
            checked(record['video']);raw_path(root,source['motion_asset']);raw_path(root,source['audio_asset'])
            value=record['calibration']
            stream=value['stream_metadata']
            assert abs(float(stream['audio'].get('start_time',0))-float(stream['video'].get('start_time',0)))<1e-12
            assert value['verified'] and value['minimum_ncc']>=POLICY['minimum_ncc']
            replay_candidates.setdefault(record['role'],record)
    assert not counts['EXCLUDED_TECHNICAL']
    replays=[]
    for role,record in sorted(replay_candidates.items()):
        source=expected[record['pair_uid']]
        with raw_path(root,source['motion_asset']).open('rb') as f:n=len(pickle.load(f)['smpl_poses'])
        actual=calibrate(checked(record['video']),raw_path(root,source['audio_asset']),n)
        assert actual['verified'] and actual['audio_offset_seconds']==record['calibration']['audio_offset_seconds']
        delta=max(abs(x['ncc']-y['ncc']) for x,y in zip(actual['anchors'],record['calibration']['anchors']))
        assert delta<1e-12
        replays.append(dict(role=role,pair_uid=record['pair_uid'],offset_exact=True,anchor_ncc_max_difference=delta))
    prepared=read(features/'assets.json');windows,roles,preflight=inspect(prepared);assert preflight['ready']
    assert preflight==read(features/'preflight.json')
    manifest=read(checked(prepared['paired_generation_manifest']))
    look={r['pair_uid']:r for r in ledger}
    identity_roles={};generic_titles=defaultdict(set);producers={};crop_roles=defaultdict(set)
    for row in manifest['rows']:
        uid=row['window_uid'].rsplit(':',1)[0];source=look[uid];assert source['status']=='VERIFIED_PAIR'
        assert row['recording_uid']==source['recording_uid'] and row['choreography_uid']==source['choreography_uid']
        assert row['audio_offset_seconds']==source['calibration']['audio_offset_seconds']
        for kind,key in (('capture',row['recording_uid']),('choreography',row['choreography_uid']),('music',row['audio']['sha256'])):
            identity=(kind,key)
            assert identity not in identity_roles or identity_roles[identity]==row['role']
            identity_roles[identity]=row['role']
        generic_titles[source['choreography_title'].lower()].add(row['role'])
        receipt=read(checked(row['jukebox_receipt']));producer=read(checked(receipt['producer']))
        assert producer['features_sha256']==row['jukebox']['sha256']
        assert producer['audio']==row['audio']['sha256'] and producer['sample_start']==receipt['sample_start']
        assert producer['sample_end']==receipt['sample_end']
        producers[row['jukebox']['sha256']]=receipt['producer']
        crop_roles[producer['crop']['sha256']].add(row['role'])
    status=read(training/'status.json');protocol=read(training/'protocol.json');validation=read(training/'validation.json')
    history=read(training/'history.json')
    assert status['status']=='PAIRED_COMPLETION_TRAINED_UNQUALIFIED' and status['training_steps']==300
    assert status['weights_before']==status['weights_after'] and status['frozen_model_unchanged']
    assert not status['full_motion_qualified'] and not status['music_qualified']
    assert status['protocol_sha256']==sha(training/'protocol.json') and status['checkpoint_sha256']==sha(training/'adapter.pt')
    assert set(protocol['train_groups']).isdisjoint(protocol['validation_groups'])
    assert all(roles[g]=='train' for g in protocol['train_groups'])
    assert all(roles[g]=='val' for g in protocol['validation_groups'])
    assert all(roles[r['group']]=='val' for r in validation['evaluated_windows'])
    assert len(history)==6 and [r['step'] for r in history]==list(range(50,301,50))
    write(output,dict(status='VERIFIED_SOURCE_CLOCK_FEATURE_TRAINING_RECEIPTS',source_counts=dict(counts),
        total_sources=len(ledger),retained_windows=len(manifest['rows']),
        window_roles=dict(Counter(r['role'] for r in manifest['rows'])),
        unique_feature_arrays=len(producers),capture_music_official_choreography_roles_disjoint=True,
        generic_movement_titles_crossing_roles=[k for k,v in generic_titles.items() if len(v)>1],
        exact_audio_crop_hashes_crossing_roles=[k for k,v in crop_roles.items() if len(v)>1],
        stream_origins_match_for_all_admitted_recordings=True,
        independence_scope='official capture IDs, audio bytes and genre-namespaced basic workbook entries; not generic-movement semantic independence or a fresh blind test',
        calibration_replays=replays,preflight=preflight,training_status=status,training_protocol=protocol,
        validation=validation,history=history,optimizer_steps_in_verifier=0,native_generation_evaluated=False))
    print('VERIFIED_AIST',dict(counts),'WINDOWS',len(manifest['rows']),'FEATURES',len(producers),flush=True)
if __name__=='__main__':main()
