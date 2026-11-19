"""Public scientific metadata export; no coordinates, music or model tensors."""
from pathlib import Path
from dunhuang.research.timing_r64.common import ROOT,read,write,sha,binding

def small_record(r):
    m=r['methods']['G1']
    return dict(key=r['key'],case_key=r['case_key'],gap_frames=r['gap_frames'],
        offset_frames=r['timing_offset_frames'],allocation=r['allocation'],status=r['status'],
        source_uids={s:r[s].get('source_uid') for s in ('left','right')},
        source_intervals={s:r[s].get('source_interval') for s in ('left','right')},
        target_intervals={s:r[s].get('target_interval') for s in ('left','right')},
        time_warp_sha256={s:r[s].get('time_warp_sha256') for s in ('left','right')},
        cache_key=r['execution_cache_key'],registered_parent=r['registered_frozen_r63_input'],
        constructible=m['constructible'],original_screen=m['passes_original_screen'],
        full_R_sha256=m.get('full_R_sha256'),bridge_metrics=m.get('metrics'),
        full_dynamics=m.get('full_dynamics'),retention_guard_passed=r['retention_guard_passed'],
        core_retention={s:r[s].get('retention_vs_fixed48') for s in ('left','right')},
        balanced_cost=r['balanced_cost'],music_proxy={k:r['music_reward'].get(k) for k in
            ('phase_beat_score','phase_accent_score','section_proxy_score','primary_music_score')},
        shuffled_music_proxy={k:r.get('music_reward_shuffled',{}).get(k) for k in
            ('phase_beat_score','phase_accent_score','section_proxy_score','primary_music_score')},
        section_relation_proxy=r['section_relation_proxy'])

def main():
    out=Path('reports/r64_release')
    if out.exists():raise ValueError('Fresh R64 release directory required')
    main=read(ROOT/'main/result.json');controls=read(ROOT/'controls/result.json');ind=read(ROOT/'independent/result.json')
    assert ind['scientific_values_exact'] and ind['arrays_bytes_exact']
    out.mkdir(parents=True)
    for mode,r in (('main',main),('controls',controls)):
        write(out/(mode+'_summary.json'),r['summary'])
        write(out/(mode+'_selections.json'),r['selections'])
    write(out/'candidate_metrics.json',[small_record(r) for r in controls['records']])
    write(out/'reward_control_effect.json',controls['reward_control_effect'])
    write(out/'protocol.json',main['protocol']);write(out/'independent.json',ind)
    clocks=[]
    for song,c in main['music_clocks'].items():
        clocks.append(dict(song=song,audio_sha256=c['audio_sha256'],frames=c['frames'],
            audio_samples=c['audio_samples'],sample_rate=c['sample_rate'],clock_sha256=c['clock_sha256'],
            raw_legacy_events=len(c['raw_legacy_beat_frames']),interior_beat_candidates=len(c['beat_frames']),
            excluded_sentinels=c['excluded_legacy_boundary_sentinels'],
            accents=len(c['accent_frames']),section_proxies=len(c['section_proxy_frames']),
            chroma_phrase_proxies=len(c['phrase_proxy_frames']),downbeat_status=c['downbeat_status'],
            phrase_truth=c['phrase_truth'],beat_scope=c['beat_scope'],human_reviewed=False))
    write(out/'music_clock_audit.json',clocks)
    write(out/'qualification.json',dict(P0=main['P0'],P2=main['P2'],
        unavailable_hierarchy_policies=main['hierarchy_policies'],
        optimized_scores_are_independent_quality=False,all_cases_historically_exposed=True,
        independent_edge_sampling=False,human_reviewed=False,Native24='NOT_EVALUABLE',can_route=False))
    receipts=[]
    for p in sorted((ROOT/'receipts').glob('*.json')):
        write(out/('receipt_'+p.name),read(p));receipts.append(p.name)
    files=[]
    for folder in ('main','controls','graph','independent'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file():files.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
    write(out/'scientific_files.json',dict(files=files))
    write(out/'public_manifest.json',dict(binding=binding(),
        main_result_sha256=sha(ROOT/'main/result.json'),controls_result_sha256=sha(ROOT/'controls/result.json'),
        main_execution_commit=main['binding']['commit'],controls_execution_commit=controls['binding']['commit'],
        scientific_files_sha256=sha(out/'scientific_files.json'),independent_sha256=sha(ROOT/'independent/result.json'),
        receipts=receipts,public_coordinates=False,public_audio=False,public_raw_arrays=False,
        can_route=False,Native24='NOT_EVALUABLE'))
    print('R64_PUBLIC_METADATA_READY',flush=True)

if __name__=='__main__':main()
