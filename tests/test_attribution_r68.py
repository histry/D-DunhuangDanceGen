"""Synthetic published-metadata checks; no R68 experimental output is read."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

script=Path(__file__).with_name('attribution_r68.py')
if not script.is_file():script=Path(__file__).resolve().parents[1]/'scripts/attribution_r68.py'
spec=importlib.util.spec_from_file_location('attribution_r68_tested',script)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def write(path,value):path.write_bytes((json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode())


def legal(case,method,delta=0):
    bridge=format(int(case['key'][1:])+1,'064x');shared=dict(count=20,mean=delta,minimum=min(0,delta),maximum=max(0,delta))
    base_magnitude=500 if method=='E1W' else 100
    groups={name:dict(stencil_count=20,spaces={space:dict(signed_magnitude_delta=copy.deepcopy(shared),
        actual=dict(mean=base_magnitude+delta),baseline=dict(mean=base_magnitude)) for space in ('FK','SO3')}) for name in a.GROUPS}
    groups['fully_outside_edit_domain']=dict(stencil_count=100,spaces={space:dict(signed_magnitude_delta=dict(count=100,mean=0,minimum=0,maximum=0)) for space in ('FK','SO3')})
    for name,count in zip(('target_event','compensation','protected','untouched'),(15,115,2,105)):
        groups['partition_'+name]=dict(stencil_count=count)
    metrics=dict(stencils=dict(D3=dict(full_stencil_count=237,groups=groups)),
        footprint=dict(protected_q_and_R_bytes_exact=True,unchanged_outside_declared_edit_exact=True),
        whole=dict(actual=dict(FK=dict(D3=dict(p95=10+delta))),baseline=dict(FK=dict(D3=dict(p95=10)))),
        dose=dict(maximum_abs_source_position_change_frames=abs(delta)),
        activity_normalized_oscillation=dict(actual=dict(FK_velocity_6Hz_to_Nyquist_power_fraction=.2+.001*delta),baseline=dict(FK_velocity_6Hz_to_Nyquist_power_fraction=.2)))
    row=dict(case_key=case['key'],source_uid=case['source_uid'],source_interval=case['source_interval'],method=method,status='LEGAL',reason=None,metrics=metrics,
        connection=dict(bridge_R_sha256=bridge,metrics=dict(gap_D3=5.)),
        bridge_invariance=dict(boundary_readset_identical=True,actual_bridge_bytes_identical=True,baseline_bridge_R_sha256=bridge),
        R_sha256='1'*64,q_sha256='2'*64,audio_sha256='3'*64)
    row['music_cross_invariance']=dict(motion_R_sha256=row['R_sha256'],q_sha256=row['q_sha256'],
        shared_same_dynamics_sha256='4'*64,shared_same_pair_dynamics_sha256='5'*64,
        shared_same_bridge_R_sha256=bridge,correct_audio_sha256=row['audio_sha256'],alternate_audio_sha256='6'*64,
        shared_same_motion_array=True,shared_same_dynamics_record=True)
    if method=='Uniform':
        metrics['footprint']['protected_q_and_R_bytes_exact']=False
        row['bridge_invariance']['actual_bridge_bytes_identical']=False
        row['connection']['bridge_R_sha256']='f'*64
    return row


def seal(directory,records,cases):
    directory.mkdir(parents=True,exist_ok=True)
    write(directory/'controls_records.json',records);write(directory/'controls_population.json',dict(cases=cases))
    write(directory/'public_manifest.json',dict(files=[dict(path=name,sha256=a.sha(directory/name),bytes=(directory/name).stat().st_size)
        for name in ('controls_records.json','controls_population.json')]))


def fixture(directory):
    cases=[dict(key='c'+str(i),source_uid='s'+str(i%8),source_interval=[100,340]) for i in range(12)]
    values={'A_early':{0:2,8:6,1:10},'A_late':{0:0,8:0,1:0,2:50},
            'B_early':{0:3,1:3,3:90},'B_late':{0:4,1:4}}
    records=[]
    for case in cases:
        i=int(case['key'][1:])
        for method in a.METHODS:
            if method in ('E1W','Uniform') or i in values[method]:
                row=legal(case,method,values.get(method,{}).get(i,0))
            else:row=dict(case_key=case['key'],source_uid=case['source_uid'],source_interval=case['source_interval'],method=method,status='NOT_FEASIBLE',reason='FROZEN_RATE_BOUND',metrics=None)
            records.append(row)
    seal(directory,records,cases)
    return records,cases


def test_common_population_and_source_balance_preserve_all_registered_failures(tmp_path):
    fixture(tmp_path/'release');result=a.analyze(tmp_path/'release')
    assert len(result['rows'])==72 and all(x['registered_cases']==12 for x in result['methods'].values())
    assert result['methods']['A_early']['legal_cases']==3
    assert result['methods']['A_late']['legal_cases']==4
    paired=result['comparisons']['A_early_minus_late'];field='D3_touch_target_signed_magnitude_delta_mean'
    assert paired['common_legal_cases']==3 and len(paired['excluded_cases'])==9
    assert paired['metrics'][field]['position_weighted_mean']==6
    assert paired['metrics'][field]['source_interval_balanced_mean']==7
    assert paired['metrics'][field]['ci95'] is None
    b=next(x for x in result['rows'] if x['case_key']=='c0' and x['method']=='B_early')
    assert b['metrics'][field]==3  # Own B-mask baseline, not the E1W A-mask row.
    assert b['metrics']['G1_gap_D3_difference']==0
    assert b['checks']['music_cross_metadata']['independently_rehashed_motion_arrays'] is False
    uniform=next(x for x in result['rows'] if x['method']=='Uniform')
    assert uniform['checks']['strict_internal_gate_applied'] is False
    assert result['inputs']['all_published_file_hashes_verified'] is True
    assert result['confidence_intervals'] is None and result['can_route'] is False


def test_strict_internal_metadata_must_preserve_partition_outside_delta_and_bridge(tmp_path):
    records,cases=fixture(tmp_path/'base')
    for failure in ('partition','outside','protection','bridge'):
        current=copy.deepcopy(records);row=current[0]
        if failure=='partition':row['metrics']['stencils']['D3']['groups']['partition_target_event']['stencil_count']+=1
        elif failure=='outside':row['metrics']['stencils']['D3']['groups']['fully_outside_edit_domain']['spaces']['FK']['signed_magnitude_delta']['maximum']=1e-12
        elif failure=='protection':row['metrics']['footprint']['unchanged_outside_declared_edit_exact']=False
        else:row['bridge_invariance']['baseline_bridge_R_sha256']='f'*64
        folder=tmp_path/failure;seal(folder,current,cases)
        with pytest.raises(ValueError):a.analyze(folder)


def test_manifest_tampering_unsafe_paths_and_cli_overwrite_are_rejected(tmp_path,monkeypatch):
    release=tmp_path/'release';fixture(release);out=tmp_path/'analysis.json'
    monkeypatch.setattr(sys,'argv',['attribution_r68.py','--release',str(release),'--out',str(out)])
    a.main();before=out.read_bytes()
    with pytest.raises(ValueError,match='Fresh'):a.main()
    assert out.read_bytes()==before
    (release/'controls_records.json').write_bytes(b'[]')
    with pytest.raises(ValueError,match='Published bytes'):a.analyze(release)
    records,cases=fixture(release)
    manifest=a.read(release/'public_manifest.json');manifest['files'].append(dict(path='../outside.json',sha256='f'*64,bytes=1));write(release/'public_manifest.json',manifest)
    with pytest.raises(ValueError,match='Unsafe'):a.analyze(release)
