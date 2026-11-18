"""Independent saved-output replay, lineage, display identity and frozen-parent checks."""
import argparse,datetime
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest,compare
from dunhuang.research.music_reperformance_r43.replay import exact_arrays
from dunhuang.research.coverage_r34.common import forbid_optimizers

def verify(out):
    forbid_optimizers();G=Path('outputs/phase_r45_01');A=Path('outputs/review_r45_01');packs=[]
    for root,name in ((G,'main'),(G,'controls'),(G,'attribution'),(A,'scores')):
        receipt=root/(name+'_replay_receipt.json');compare(root/name,root/(name+'_replay'),receipt);packs.append(dict(root=str(root),name=name,receipt_sha256=digest(receipt)))
    for name in ('result.json','reference_target_audit.json'):assert read(G/'qualification'/name)==read(G/'qualification_replay'/name)
    assert read(G/'quality.json')==read(G/'quality_replay.json');assert read(G/'quality.json')['official_fk_executed']
    assert read(A/'package/admin/manifest.json')==read(A/'package_replay/admin/manifest.json');exact_arrays(A/'package/admin/arrays.npz',A/'package_replay/admin/arrays.npz');assert read(A/'package/reviewer/protocol.json')==read(A/'package_replay/reviewer/protocol.json');assert read(A/'evaluation/result.json')==read(A/'evaluation_replay/result.json');assert read(A/'evaluation/result.json')['raw_review_count']==0
    parents=[]
    for name in ('bridge_r38_01','stabilize_r39_01','contract_r40_01','paired_event_r41_01','source_context_r42_01','music_reperformance_r43_01','sdmr_event_rank_01','reperformance_r44_01','retrieval_audit_r44_01'):
        root=Path('outputs')/name;manifest=root/'artifacts_manifest.json';files=read(manifest)['files']
        for r in files:assert (root/r['path']).stat().st_size==r['bytes'] and digest(root/r['path'])==r['sha256'],str(root/r['path'])
        parents.append(dict(root=str(root),files=len(files),manifest_sha256=digest(manifest)))
    first=G/'viewer_first_delivery' if (G/'viewer_first_delivery').exists() else G/'viewer'
    for view in (G/'viewer',first,G/'viewer_controls',G/'videos',A/'package/reviewer'):
        for r in read(view/'protocol.json')['files']:assert digest(view/r['name'])==r['sha256']
    acceptance=read(G/'receipts/primary_browser_acceptance.json');assert datetime.datetime.fromisoformat(acceptance['observed_at_utc']).timestamp()<(G/'controls/result.json').stat().st_mtime;assert acceptance['protocol_sha256']==digest(first/'protocol.json');assert not read(G/'qualification/result.json')['conditional_training']['admitted']
    write(out,dict(packs=packs,all_scientific_values_exact=True,official_fk_replayed=True,review_package_replayed=True,raw_human_reviews=0,parents=parents,primary_delivery_preceded_controls=True,optimizer_calls=0,whole_song_generations=0,can_route=False,operational_qualified=False));print('R45_INDEPENDENT_OK',len(packs),sum(p['files'] for p in parents),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();verify(a.out)
