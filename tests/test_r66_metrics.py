"""R66 deterministic TEST_FIXTURE scores, not empirical pairing evidence."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

try:
    from dunhuang.research.timing_validity_r66 import metrics as m
except ModuleNotFoundError:
    spec=importlib.util.spec_from_file_location('TEST_FIXTURE_r66_metrics',Path(__file__).with_name('metrics.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(uid,source='SOURCE_A',group='MUSIC_A',dataset='AIST++',role='train',scores=None,**kwargs):
    return dict(uid=uid,source_uid=source,group=group,dataset=dataset,role=role,
                scores={'Primary':scores if scores is not None else [0.,0.,1.,0.,0.],
                        'Energy':[0.,0.,1.,0.,0.]},**kwargs)


def test_all_ties_get_fair_chance_and_expected_mae():
    q=m.pair_metrics([1.]*5)
    assert q['top1_accuracy']==.2 and q['mae_frames']==9.2
    assert q['pairwise_win_rate']==.5 and q['pairwise_tie_rate']==1.
    assert q['best_offsets']==list(m.OFFSETS) and q['tie_count']==5


def test_partial_top_ties_do_not_use_argmax_order():
    q=m.pair_metrics([1.,0.,1.,0.,0.])
    assert q['top1_accuracy']==.5 and q['mae_frames']==7.5
    assert q['pairwise_win_rate']==.875
    assert q['pairwise'][0]['win']==.5


def test_tolerance_is_fixed_and_zero_is_direct_clock_target():
    q=m.pair_metrics([0.,0.,1.,1.+.5e-12,0.])
    assert q['best_offsets']==[0,8] and q['top1_accuracy']==.5
    assert m.pair_metrics([1.,0.,0.,0.,0.])['mae_frames']==15
    for args in [dict(truth=8),dict(truth=False),dict(tolerance=1e-7),
                 dict(offsets=[-15,-8,False,8,15]),dict(offsets=[-15.,-8.,0.,8.,15.])]:
        with pytest.raises(ValueError):m.pair_metrics([1.]*5,**args)


def test_missing_evidence_never_becomes_zero_score_or_chance_success():
    q=m.pair_metrics([None]*5)
    assert not q['complete'] and q['top1_accuracy'] is None and q['pairwise_win_rate'] is None
    assert q['finite_conditions']==0 and q['available_pair_count']==0
    q=m.pair_metrics([1.,None,2.,None,None])
    assert q['top1_accuracy'] is None and q['pairwise_win_rate'] is None
    assert q['available_pairs_win_rate']==1 and q['available_pair_count']==1


@pytest.mark.parametrize('scores',[[0]*4,[0]*6,[False]*5,[float('nan')]*5,[float('inf')]*5,'bad'])
def test_invalid_score_vectors(scores):
    with pytest.raises(ValueError):m.pair_metrics(scores)


def test_window_then_recording_then_music_group_balance():
    # A1 has nine wins and one loss: mean .9; A2 mean0 -> groupA=.45.
    # B1 mean1 -> final(.45+1)/2=.725, not a window weighted mean.
    rows=[dict(group='A',source_unit='A1',value=1.) for _ in range(9)]
    rows += [dict(group='A',source_unit='A1',value=0.),dict(group='A',source_unit='A2',value=0.),
             dict(group='B',source_unit='B1',value=1.)]
    q=m.balanced_summary(rows,'value')
    assert q['mean']==pytest.approx(.725)
    assert q['groups']==2 and q['sources']==3 and q['evaluable_queries']==12
    assert q['group_means']['A']['mean']==pytest.approx(.45)
    assert q==m.balanced_summary(rows,'value')
    assert q['ci95'] is not None and q['below_eight_registered_music_groups']


def test_no_ci_for_single_music_group_or_no_evidence():
    q=m.balanced_summary([dict(group='A',source_unit='S',value=1.)],'value')
    assert q['mean']==1 and q['ci95'] is None
    q=m.balanced_summary([dict(group='A',source_unit='S',value=None)],'value')
    assert q['mean'] is None and q['groups']==0 and q['ci95'] is None


def test_summary_retains_registered_nulls_and_dataset_role_strata():
    rows=[row('T'),row('V',source='S_V',group='G_V',role='val'),
          row('F',source='S_F',group='G_F',dataset='FineDance',role='genre_train_diagnostic')]
    rows[1]['scores']['Primary']=[None]*5;rows[1]['motion_event_count']=0
    q=m.summarize(rows)
    assert set(q['strata'])=={'AIST++|train','AIST++|val','FineDance|genre_train_diagnostic'}
    v=q['strata']['AIST++|val']['methods']['Primary']
    assert v['coverage']['registered_queries']==1 and v['coverage']['complete_queries']==0
    assert v['coverage']['no_motion_events_explicit']==1
    assert v['metrics']['top1_accuracy']['mean'] is None
    assert not q['cross_dataset_or_role_pooling_performed'] and not q['operational_qualified']


def test_unreported_null_reason_is_not_invented_as_no_motion():
    r=row('Q');r['scores']['Primary']=[None]*5
    q=m.summarize([r])['strata']['AIST++|train']['methods']['Primary']['coverage']
    assert q['no_score_queries']==1 and q['no_motion_events_explicit']==0
    assert q['motion_event_status_unreported']==1


def test_method_difference_uses_common_complete_queries():
    rows=[row('Q1'),row('Q2')]
    rows[0]['scores']['Energy']=[1.]*5
    rows[1]['scores']['Primary']=[None]*5
    q=m.summarize(rows)['strata']['AIST++|train']['paired_method_differences']['Primary_minus_Energy']
    assert q['registered_queries']==2 and q['common_complete_queries']==1
    assert q['metrics']['top1_accuracy']['mean']==pytest.approx(.8)
    assert q['metrics']['mae_frames']['mean']==pytest.approx(-9.2)


def test_wrong_song_is_diagnostic_and_never_has_iid_donor_ci():
    rows=[row('Q1',group='G1'),row('Q2',group='G2',source='S2')]
    for r in rows:r['wrong_scores']={'Primary':[1.],'Energy':[0.]}
    q=m.summarize(rows)['strata']['AIST++|train']['methods']['Primary']['wrong_song']
    assert q['registered_donors']==q['evaluable_donors']==2
    stats=q['metrics']['correct_vs_wrong_win_rate']
    assert stats['mean']==.5 and stats['ci95'] is None and stats['bootstrap_draws']==0
    assert 'NOT_ARTISTIC' in m.wrong_song_metrics(1.,[0.])['scope']


def test_source_aliases_share_a_balance_unit_and_evidence_is_reported():
    rows=[row('Q1',source='ALIAS1',raw_motion_sha256='a'*64,audio_sha256='b'*64,source_interval=[0,150]),
          row('Q2',source='ALIAS2',raw_motion_sha256='a'*64,audio_sha256='b'*64,source_interval=[0,150])]
    q=m.summarize(rows)
    audit=q['population']
    assert audit['registered_source_uids']==2 and audit['recording_balance_units']==1
    assert audit['source_duplicate_clusters'][0]['duplicate_source_aliases']
    assert len(audit['metadata_duplicate_window_clusters'])==1
    g=audit['music_group_evidence'][0]
    assert g['source_uid_count']==2 and g['source_balance_unit_count']==1 and g['music_id_count']==1
    assert q['strata']['AIST++|train']['methods']['Primary']['metrics']['top1_accuracy']['sources']==1


def test_split_duplicate_source_or_audio_cannot_fake_group_bootstrap():
    rows=[row('Q1',raw_motion_sha256='a'*64),row('Q2',source='ALIAS',group='OTHER',raw_motion_sha256='a'*64)]
    with pytest.raises(ValueError):m.summarize(rows)
    rows=[row('Q1',audio_sha256='b'*64),row('Q2',source='OTHER',group='OTHER',audio_sha256='b'*64)]
    with pytest.raises(ValueError):m.summarize(rows)


def test_population_id_tampering_duplicates_and_unknown_methods():
    with pytest.raises(ValueError):m.summarize([row('DUP'),row('DUP')])
    bad=row('Q');bad['scores']['NewTunedMethod']=[1.]*5
    with pytest.raises(ValueError):m.summarize([bad])
    bad=row('Q',motion_event_count=0,no_motion_events=False)
    with pytest.raises(ValueError):m.summarize([bad])


def test_empty_population_has_no_fake_labels_or_operational_qualification():
    q=m.summarize([])
    assert q['registered_queries']==0 and q['strata']=={} and q['queries']==[]
    assert q['status']=='NOT_EVALUABLE_NO_REGISTERED_QUERIES' and not q['can_route']
