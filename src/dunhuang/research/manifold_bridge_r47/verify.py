"""Independent exact replay, observation contracts and frozen-parent seal verification."""
import argparse,datetime
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import compare,digest
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.coverage_r34.common import forbid_optimizers

def run(out):
    forbid_optimizers();G=Path('outputs/manifold_bridge_r47_01');C=Path('outputs/event_coverage_r47_01');P=Path('outputs/music_phase_r47_01');packs=[]
    for name in ('main','controls'):
        receipt=G/(name+'_replay_receipt.json');compare(G/name,G/(name+'_replay'),receipt);packs.append(dict(name=name,receipt_sha256=digest(receipt)))
    def same(a,b):
        aa=read(a);bb=read(b);aa.pop('code_commit',None);bb.pop('code_commit',None);assert aa==bb
    same(C/'index/result.json',C/'index_replay/result.json');same(P/'main/result.json',P/'main_replay/result.json');assert read(G/'connections.json')==read(G/'connections_replay.json')
    a=Arrays(G/'main/arrays.npz');b=Arrays(G/'controls/arrays.npz')
    for key in a.files:assert np.array_equal(a[key],b[key],equal_nan=True),key
    a.close();b.close();parents=[]
    for name in ('bridge_r38_01','stabilize_r39_01','contract_r40_01','paired_event_r41_01','source_context_r42_01','music_reperformance_r43_01','sdmr_event_rank_01','reperformance_r44_01','retrieval_audit_r44_01','phase_r45_01','review_r45_01','bridge_audit_r46_01','source_coverage_r46_01'):
        root=Path('outputs')/name;m=root/'artifacts_manifest.json';files=read(m)['files']
        for f in files:assert (root/f['path']).stat().st_size==f['bytes'] and digest(root/f['path'])==f['sha256'],str(root/f['path'])
        parents.append(dict(root=str(root),files=len(files),manifest_sha256=digest(m)))
    quality=[]
    for name in ('main','controls'):
        q=read(G/(name+'_quality.json'));assert q==read(G/(name+'_quality_replay.json')) and q['official_fk_executed'];quality.append(dict(name=name,count=q['official_fk_trajectories'],maximum_error_m=q['maximum_fk_error_m']))
    for root,folder in ((G,'viewer'),(G,'videos'),(G,'viewer_initial_caption'),(G,'viewer_controls'),(C,'viewer')):
        for f in read(root/folder/'protocol.json')['files']:assert digest(root/folder/f['name'])==f['sha256']
    for root in (G,C):
        r=read(root/'receipts/main_browser_acceptance.json');assert r['passed'] and r['protocol_sha256']==digest(root/'viewer/protocol.json');assert datetime.datetime.fromisoformat(r['observed_at_utc']).timestamp()<(G/'controls/result.json').stat().st_mtime
    phase=read(P/'main/result.json');assert phase['registered_pair_records']==476 and phase['test_tensor_reads']==0 and not phase['oracle_labels_read'] and not phase['training_admitted'];review=read('outputs/review_r45_01/evaluation/result.json');assert review['raw_review_count']==0
    write(out,dict(packs=packs,all_scientific_values_exact=True,primary_G0_G1_identical_in_controls=True,catalog_replayed=True,phase_audit_replayed=True,connections_replayed=True,official_fk=quality,parents=parents,first_effect_before_controls=True,raw_human_reviews=0,human_metrics_status=review['status'],optimizer_calls=0,new_neural_samples=0,whole_song_generations=0,can_route=False,operational_qualified=False));print('R47_INDEPENDENT_OK',sum(p['files'] for p in parents),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
