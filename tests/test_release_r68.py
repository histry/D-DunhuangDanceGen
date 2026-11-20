"""Guard R68 publication identity, failed denominators and signal exclusion."""
import importlib.util
import json
from pathlib import Path

import pytest

_script=Path(__file__).with_name('release_r68.py')
if not _script.is_file():_script=Path(__file__).resolve().parents[1]/'scripts/release_r68.py'
spec=importlib.util.spec_from_file_location('release_r68_tested',_script)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def archive(repo,mode='preview',commit='a'*40):
    stage=repo/'outputs/local_timing_r68_01'/mode;stage.mkdir(parents=True)
    records=[dict(case_key='c',method='E1W',status='LEGAL',source_interval=[10,250],
        metrics=dict(whole=dict(actual=dict(FK=dict(D3=dict(mean=2.,p95=3.,curve=[1.,2.,3.]))))),
        q=[10,11,12],P=[[[1.,2.,3.]]],connection=dict(passes_original_screen=True,boundary_jets=[[[9.]]])),
        dict(case_key='c',method='A_minus6',status='NOT_FEASIBLE',reason='FROZEN_RATE_BOUND',metrics=None)]
    docs={'records.json':records,'population.json':dict(cases=[dict(key='c',source_uid='S',source_interval=[10,250])]),
          'registration.json':dict(binding=dict(commit=commit),protocol=dict(half_width_frames=60),neural_training_steps=0)}
    for name,value in docs.items():r.write(stage/name,value)
    (stage/'arrays.npz').write_bytes(b'SECRET_RAW_ROTATIONS_NOT_FOR_PUBLICATION')
    (stage/'viewer').mkdir();(stage/'viewer/index.html').write_text('SECRET_VIEWER_COORDINATES',encoding='utf8')
    r.write(stage/'manifest.json',dict(files={str(p.relative_to(stage)).replace('\\','/'):r.sha(p)
        for p in stage.rglob('*') if p.is_file()},binding=dict(commit=commit,code={'engine.py':'a'*64})))
    return stage


def test_project_keeps_scalar_identity_and_failed_rows_but_no_raw_signals():
    doc=dict(source_interval=[12,252],source_uid='s',status='NOT_FEASIBLE',reason='NO_LEGAL_DELTA',
        q=[12.,13.],source_positions=[12.,13.],actual_boundary_jets=[[[1.]]],
        tokens=[[1.,2.]],half_open_frame_ranges=[[10,20]],right_endpoint_frames=[10,11],
        metrics=dict(D3=dict(mean=1.,curve=[8.,9.]),boundary_matrix=[[2.]],
            frame_count=240),empty={},rows=[dict(status='NO_EVENT',metrics=None)])
    public=r.project(doc)
    assert public['source_interval']==[12,252] and public['status']=='NOT_FEASIBLE'
    assert public['rows']==[dict(status='NO_EVENT',metrics=None)]
    assert public['metrics']==dict(D3=dict(mean=1.),frame_count=240)
    assert r.project(dict(preview_methods=['E1W','A_early'],target_deltas_frames=[-6,6],
        rate_limits_dt_ds=[.75,1.33]))==dict(preview_methods=['E1W','A_early'],target_deltas_frames=[-6,6],rate_limits_dt_ds=[.75,1.33])
    assert not any(k in public for k in ('q','source_positions','actual_boundary_jets','tokens','half_open_frame_ranges','right_endpoint_frames'))
    with pytest.raises(ValueError):r.project(dict(score=float('nan')))


def test_seal_rehashes_actual_files_and_rejects_changed_bytes(tmp_path):
    stage=archive(tmp_path);sealed=r.hash_manifest(stage)
    assert sealed['status']=='ALL_MANIFEST_BYTES_REHASHED_EXACT'
    assert len(sealed['files'])==5
    (stage/'arrays.npz').write_bytes(b'changed scientific bytes')
    with pytest.raises(ValueError,match='changed'):r.hash_manifest(stage)


@pytest.mark.parametrize('name',['../outside.json','/absolute/private.json','C:/absolute/private.json'])
def test_unsafe_manifest_members_fail_before_reading_external_content(tmp_path,name):
    stage=archive(tmp_path);doc=r.read(stage/'manifest.json');doc['files'][name]='f'*64
    r.write(stage/'manifest.json',doc)
    with pytest.raises(ValueError,match='Unsafe'):r.hash_manifest(stage)


def test_independent_replay_requires_empty_differences_and_actual_source_binding(tmp_path):
    stage=archive(tmp_path);receipt=tmp_path/'independent.json'
    r.write(receipt,dict(scientific_outputs_exact=True,differences=[],source_manifest_sha256=r.sha(stage/'manifest.json')))
    assert r.replay(receipt,stage/'manifest.json')['status']=='EXACT'
    doc=r.read(receipt);doc['source_manifest_sha256']='f'*64;r.write(receipt,doc)
    assert r.replay(receipt,stage/'manifest.json')['status']=='NOT_EXACT_OR_UNBOUND'
    doc['source_manifest_sha256']=r.sha(stage/'manifest.json');doc['differences']=['records.json'];r.write(receipt,doc)
    assert r.replay(receipt,stage/'manifest.json')['status']=='NOT_EXACT_OR_UNBOUND'


def test_fresh_export_retains_all_failure_denominators_and_distinct_stage_commits(tmp_path,monkeypatch):
    archive(tmp_path,'preview','a'*40);archive(tmp_path,'controls','b'*40)
    incomplete=tmp_path/'outputs/local_timing_r68_execution_01';incomplete.mkdir()
    r.write(incomplete/'execution_preview.json',dict(exit_status=0,commit='a'*40,password='DO_NOT_PUBLISH'))
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:'c'*40+'\n')
    out=r.export(tmp_path)
    qualification=r.read(out/'qualification.json')
    assert qualification['source_commit_by_stage']==dict(preview='a'*40,controls='b'*40)
    assert qualification['independent_by_stage']==dict(preview='NOT_AVAILABLE',controls='NOT_AVAILABLE')
    summary=r.read(out/'preview_summary.json')
    failed=summary['methods']['A_minus6']
    assert failed['registered_records']==1 and failed['records_with_actual_metrics']==0
    assert failed['failure_reason_counts']=={'FROZEN_RATE_BOUND':1}
    metric=failed['metrics']['FK_D3_p95_mps3']
    assert metric['registered_records']==1 and metric['evaluable_records']==0 and metric['position_weighted_mean'] is None
    records=r.read(out/'preview_records.json')
    assert len(records)==2 and records[1]['metrics'] is None
    assert not (out/'arrays.npz').exists() and not (out/'viewer').exists()
    published='\n'.join(p.read_text(encoding='utf8') for p in out.iterdir() if p.is_file())
    assert 'SECRET_RAW_ROTATIONS_NOT_FOR_PUBLICATION' not in published
    assert 'SECRET_VIEWER_COORDINATES' not in published and 'DO_NOT_PUBLISH' not in published
    for row in r.read(out/'public_manifest.json')['files']:
        assert r.sha(out/row['path'])==row['sha256']
    with pytest.raises(ValueError,match='fresh'):r.export(tmp_path)
