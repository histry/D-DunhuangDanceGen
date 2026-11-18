"""Fresh-process R46 replay and frozen lineage verification, with optimizer prohibition."""
import argparse,datetime
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.reperformance_r44.replay import compare,digest
from dunhuang.research.reperformance_r44.archive import Arrays

def run(out):
    forbid_optimizers();root=Path('outputs/bridge_audit_r46_01');packs=[]
    for name in ('main','oracle'):
        receipt=root/(name+'_replay_receipt.json');compare(root/name,root/(name+'_replay'),receipt);packs.append(dict(name=name,receipt_sha256=digest(receipt)))
    old=Arrays(root/'first_clock_display/main/arrays.npz');new=Arrays(root/'main/arrays.npz');parent=Arrays('outputs/phase_r45_01/main/arrays.npz');assert old.files==new.files
    for key in new.files:
        if key.endswith('_Phase_clock'):assert np.array_equal(new[key],parent[key])
        else:assert np.array_equal(old[key],new[key],equal_nan=True),key
    old.close();new.close();parent.close()
    parents=[]
    for name in ('bridge_r38_01','stabilize_r39_01','contract_r40_01','paired_event_r41_01','source_context_r42_01','music_reperformance_r43_01','sdmr_event_rank_01','reperformance_r44_01','retrieval_audit_r44_01','phase_r45_01','review_r45_01'):
        p=Path('outputs')/name;m=p/'artifacts_manifest.json';files=read(m)['files']
        for f in files:assert (p/f['path']).stat().st_size==f['bytes'] and digest(p/f['path'])==f['sha256'],str(p/f['path'])
        parents.append(dict(root=str(p),files=len(files),manifest_sha256=digest(m)))
    quality=[]
    for name in ('main','oracle'):
        a=read(root/(name+'_quality.json'));assert a==read(root/(name+'_quality_replay.json')) and a['official_fk_executed'];quality.append(dict(name=name,count=a['official_fk_trajectories'],maximum_error_m=a['maximum_fk_error_m']))
    b=read(root/'receipts/main_browser_acceptance.json');assert b['passed'] and digest(root/'viewer/protocol.json')==b['viewer_protocol_sha256'];assert datetime.datetime.fromisoformat(b['observed_at_utc']).timestamp()<(root/'oracle/result.json').stat().st_mtime
    review=read('outputs/review_r45_01/evaluation/result.json');assert review['raw_review_count']==0
    contract=read('outputs/phase_r45_01/qualification/reference_target_audit.json');same=[p for p in contract['pairs'] if p['same_choreography'] is True and p['shared_audio_support'][1]-p['shared_audio_support'][0]>=90]
    write(out,dict(packs=packs,all_scientific_values_exact=True,display_clock_correction_only=True,official_fk=quality,parents=parents,first_effect_before_oracle=True,raw_human_reviews=0,human_metrics_status=review['status'],cross_performance_audit=dict(physical_clock_pairs=len(contract['pairs']),same_choreography_supported_pairs=len(same),exact_stage_correspondence_verified=False,training_admitted=False),optimizer_calls=0,whole_song_generations=0,can_route=False,operational_qualified=False));print('R46_INDEPENDENT_OK',len(packs),sum(p['files'] for p in parents),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
