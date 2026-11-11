import json
import numpy as np
import pytest
from dunhuang.evaluation_benchmarks.protocols import GroundAndSkeletonContract,integrity,shadow_decision,threshold_registry
from dunhuang.evaluation_benchmarks.contact_evidence import conditioned_metrics,load_evidence
from dunhuang.evaluation_benchmarks.calibration import grouped_quantile,partition,jerk_report
from dunhuang.planning.replanner import route,budget


def motion(n=150):
    x=np.zeros((n,151),np.float32)
    x[:,7:]=np.tile([1,0,0,0,1,0],24)
    return x


def test_no_static_contact_population_is_not_zero_skate():
    fk=np.zeros((150,24,3),np.float32);fk[:,:,0]=np.arange(150)[:,None]/30
    unknown=conditioned_metrics(fk,0)
    assert unknown['independent_static_speed']['status']=='NOT_EVALUABLE'
    assert unknown['independent_static_speed']['mean'] is None
    assert unknown['contact_type_counts']['2']==0
    static=conditioned_metrics(np.zeros_like(fk),0,labels=np.ones((150,4),np.int8))
    assert static['independent_static_speed']['status']=='COMPUTED'
    assert static['independent_static_speed']['mean']==0
    sliding=conditioned_metrics(fk,0,labels=np.full((150,4),2,np.int8))
    assert sliding['independent_static_speed']['p95'] is None
    assert sliding['independent_sliding_speed']['p95']>0.99


def test_external_labels_bind_motion_clock_and_independent_method(tmp_path):
    from dunhuang.io import sha
    array=tmp_path/'states.npy';np.save(array,np.ones((150,4),np.int8))
    data=dict(schema='independent_contact_labels_v1',validated=True,motion_sha256='a'*64,
        fps=30,frames=150,foot_joint_indices=[7,8,10,11],source='manual_video_annotation',
        reviewer='independent-reviewer',independent_of_evaluation_metric=True,
        labels=dict(path=str(array),sha256=sha(array)))
    receipt=tmp_path/'labels.json';receipt.write_text(json.dumps(data))
    reference=dict(path=str(receipt),sha256=sha(receipt))
    assert load_evidence(reference,'a'*64,150)[0].shape==(150,4)
    with pytest.raises(ValueError,match='identity'):load_evidence(reference,'b'*64,150)
    data['source']='thresholded_foot_speed';receipt.write_text(json.dumps(data));reference['sha256']=sha(receipt)
    with pytest.raises(ValueError,match='independent'):load_evidence(reference,'a'*64,150)


def test_integrity_and_scene_ground_are_separate_from_empirical_gate():
    c=GroundAndSkeletonContract('skeleton','a'*64,0,'frozen_common_reference','b'*64,150)
    x=motion();j=np.zeros((150,24,3),np.float32)
    assert integrity(x,j,c)['ok']
    shifted=j.copy();shifted[:,:,1]+=.01
    assert not integrity(x,j,c,shifted)['ok']
    bad=x.copy();bad[0,79:85]=0
    assert not integrity(bad,j,c)['ok']
    with pytest.raises(ValueError,match='ground truth'):
        GroundAndSkeletonContract('skeleton','a'*64,0,'estimated_candidate_floor','b'*64,150,ground_truth=True)
    assert shadow_decision(integrity(x,j,c))['result']=='UNKNOWN_UNQUALIFIED'


def test_record_balancing_and_role_isolation():
    rows=[dict(role='train',recording_uid='a',choreography_uid='ca',source_group='g1',
               music_uid='ma',metrics={'score':0}) for _ in range(100)]
    rows+=[dict(role='train',recording_uid='b',choreography_uid='cb',source_group='g2',
                music_uid='mb',metrics={'score':10})]
    q=grouped_quantile(rows,'score',q=.75,iterations=30)
    assert q['value']==10 and q['groups']==2
    with pytest.raises(ValueError,match='Train only'):partition([dict(rows[0],role='test')])
    single=grouped_quantile(rows[:100],'score',iterations=30)
    assert single['interval95'] is None


def test_jerk_noise_diagnostics_never_rewrite_motion():
    t=np.arange(150)/30
    fk=np.zeros((150,24,3),np.float64);fk[:,:,0]=t[:,None]**3
    before=fk.copy();r=jerk_report(fk,5)
    assert np.array_equal(before,fk)
    assert r['raw']['max']==pytest.approx(6,abs=1e-6)
    assert r['violation_fraction']==1 and r['filtered_diagnostic']['used_for_acceptance'] is False


def test_shadow_cannot_drive_legacy_replanner():
    with pytest.raises(ValueError,match='Shadow'):
        route(dict(native_result='PASS',native_result_source='Native-Calib-v2'),budget({}),0)
    assert route(dict(native_result='PASS',native_result_source='Native-Legacy-v1'),budget({}),0)=='ACCEPT_NATIVE_REQUESTED_SCOPE'


def test_threshold_provenance_does_not_confuse_contact_populations():
    limits=dict(foot_penetration_min_m=-.05,foot_contact_height_m_max=.1,foot_skate_mps_p95=.18)
    rows={r['metric']:r for r in threshold_registry(limits)['metrics']}
    assert rows['foot_penetration_min_m']['contact_source']=='all_foot_joint_heights_not_contact_conditioned'
    assert rows['foot_contact_height_m_max']['contact_source']=='rederived_contacts_only'
    assert rows['foot_skate_mps_p95']['contact_source']=='rederived_contact_OR_low_height_assumed_static'


def test_reference_exceedance_never_becomes_false_rejection_without_labels():
    from dunhuang.evaluation_benchmarks.calibration import reference_report
    from dunhuang.evaluation_benchmarks.labelled_calibration import candidates
    rows=[dict(role='train',recording_uid=str(i),choreography_uid=str(i),source_group=str(i),
        music_uid=str(i),window_uid=str(i),metrics={'joint_jerk_mps3_max':1000+i},
        candidate_metrics={'joint_jerk_mps3_max':1000+i},legacy_pass=True) for i in range(6)]
    registry=threshold_registry({'joint_jerk_mps3_max':1620})
    report=reference_report(rows,registry,iterations=10)
    assert report['metrics'][0]['candidate_threshold'] is None
    assert report['metrics'][0]['false_reject_rate'] is None
    labelled=candidates(rows,registry,{},iterations=10)
    assert labelled['records'][0]['candidate_threshold'] is None
    assert not labelled['operational_qualified']

def test_threshold_aggregation_distinguishes_stage_window_and_local_jerk():
    rows={r['metric']:r for r in threshold_registry(dict(
        root_horizontal_window_displacement_max_m=1.5,joint_jerk_window_p95_max_mps3=1080,
        foot_penetration_min_m=-.05,root_y_robust_range_m=.9))['metrics']}
    assert '10s' in rows['root_horizontal_window_displacement_max_m']['metric_definition']['aggregation']
    assert rows['joint_jerk_window_p95_max_mps3']['metric_definition']['aggregation']=='max of1s-windowP95'
    assert rows['foot_penetration_min_m']['metric_definition']['aggregation']=='minimum'
    assert rows['root_y_robust_range_m']['metric_definition']['aggregation']=='P99 minus P01'

def test_error_rates_resample_sources_instead_of_overlapping_windows():
    from dunhuang.evaluation_benchmarks.labelled_calibration import clustered_error_rate
    rows=[dict(recording_uid='one',source_group='a') for _ in range(100)]
    rows+=[dict(recording_uid='two',source_group='b')]
    result=clustered_error_rate(rows,[0]*100+[1],iterations=30)
    assert result['window_rate']==pytest.approx(1/101)
    assert result['component_balanced_rate']==.5
    assert result['recording_balanced_rate']==.5
    assert result['source_components']==2
    assert clustered_error_rate(rows[:100],[0]*100,iterations=30)['component_bootstrap_interval95'] is None
