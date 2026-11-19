"""Export R65 public metadata without exposing blind assignments or identities."""
from pathlib import Path
from dunhuang.research.quality_r65.common import ROOT,CFG,read,write,sha,binding

def main():
    out=Path('reports/r65_release')
    if out.exists():raise ValueError('Fresh R65 public release required')
    main=read(ROOT/'main.json');controls=read(ROOT/'controls/result.json');ind=read(ROOT/'independent/result.json')
    empty=read(ROOT/'evaluation_empty.json');prep=main['preparation']
    assert empty['actual_input_documents']==0 and ind['packet_files_bytes_exact'] and ind['frozen_score_controls_exact']
    out.mkdir(parents=True)
    cells=[n for methods in prep['assignment_position_counts'].values() for row in methods.values() for n in row]
    write(out/'preparation_summary.json',dict(case_count=29,frozen_method_outputs=116,full_audio_items=3,
        forms=CFG['review_forms'],position_cell_count_range=[min(cells),max(cells)],maximum_position_imbalance=1,
        exact_position_counts_withheld_until_unblinding=True,identical_trajectory_group_count=len(prep['identical_trajectory_groups']),
        method_names=CFG['methods'],all_cases_included=True,originals_and_duplicates_retained=True,
        assignment_mapping_and_seed_public=False,human_records=0,human_reviewed=False))
    write(out/'qualification.json',dict(P0='NOT_EVALUABLE_NO_INDEPENDENT_MUSIC_RECORDS',
        P1='NOT_EVALUABLE_NO_INDEPENDENT_QUALITY_RATINGS',P2='NOT_EVALUABLE_NO_INDEPENDENT_HUMAN_REFERENCE',
        P3='NOT_EVALUABLE_NO_NEW_MUSIC_POPULATION',review_tools='EXECUTED_AND_BROWSER_CONTRACT_CHECKED',
        frozen_proxy_controls='EXECUTED_SCORE_ONLY_NOT_QUALITY_VALIDATION',human_reviewed=False,
        Native24='NOT_EVALUABLE',new_geometry_solves=0,training_steps=0,can_route=False))
    write(out/'protocol.json',CFG)
    write(out/'independent.json',ind)
    write(out/'empty_review_state.json',dict(actual_documents=0,actual_review_identities=0,
        event_accuracy=None,music_agreement=None,quality_agreement=None,proxy_human_correlation=None,
        status='NOT_EVALUABLE',semantic_truth='UNKNOWN',synthetic_tests_are_human_records=False))
    keep=('record_key','case_key','control','status','beat','accent','section','primary','counts','count_exact',
          'music_interval','interval_frames','shift_frames','shift_seconds','circular_shift_frames',
          'IOI_change','distance_change_fraction','distance_change_by_kind','event_time_set_change_fraction',
          'original_r64_scalar_scores','original_r64_music_reward_sha256','original_r64_score_replaced',
          'local_identity_may_differ_from_full_track_reward','phase_and_junction_locations_changed',
          'trajectories_changed','clock_sha256','audio_sha256','control_scope','sigma_frames',
          'independent_quality_metric','semantic_wrongness_truth','new_method_selections','geometry_solver_calls','optimizer_calls')
    rows=[{k:r[k] for k in keep if k in r} for r in controls['records']]
    rows.sort(key=lambda r:(str(r.get('case_key')),str(r.get('record_key')),str(r['control'])))
    write(out/'frozen_score_controls.json',dict(records=rows,selected_occurrences=controls['selected_occurrences'],
        scalar_diagnostic_records=controls['scalar_diagnostic_records'],
        controls_per_occurrence=controls['controls_per_occurrence'],
        independent_quality_metric=False,quality_winner_reported=False,all_selected_occurrences_retained=True,
        r65_blind_ids_and_selection_order_withheld=True))
    receipts=[]
    for p in sorted((ROOT/'receipts').glob('*.json')):
        # Delivery receipts contain opaque reviewer filenames. Publish their SHA,
        # not content linking private reviewer files to exact media hashes.
        r=read(p)
        if p.name.startswith('execution_'):
            assert r['exit_status']==0
            write(out/('receipt_'+p.name),r)
        receipts.append(dict(name=p.name,sha256=sha(p)))
    write(out/'public_manifest.json',dict(binding=binding(),parent_commit=CFG['parent_commit'],
        main_sha256=sha(ROOT/'main.json'),controls_sha256=sha(ROOT/'controls/result.json'),
        independent_sha256=sha(ROOT/'independent/result.json'),
        packet_manifest_sha256=main['reviewer_manifest_sha256'],receipts=receipts,
        no_blind_mapping=True,no_exact_position_counts=True,no_reviewer_identities=True,
        no_coordinates=True,no_audio=True,no_rotation_arrays=True,can_route=False))
    print('R65_PUBLIC_METADATA_READY',flush=True)

if __name__=='__main__':main()
