"""Synthetic metadata tests; no dataset arrays or generated quality assertions."""
import json

import pytest

from dunhuang.research.rhythm_r66 import population_b as m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    marker = repo/'src/dunhuang/research/complete_r36/core.py'
    marker.parent.mkdir(parents=True)
    marker.write_text('# TEST_FIXTURE metadata root marker', encoding='utf8')
    parent = repo/'outputs/timing_r64_01'
    unit = dict(unit_uid='unitA', source_uid='sourceA', source_interval=[10,250],
                support_component=[0,500], source_sha256='a'*64, native_sha256='b'*64,
                phases=[dict(type='speed_peak', body_region='upper', frame=90)])
    phases = m._phases(unit)
    beats = [0,300,380,460,600]
    warp = m.multi_warp([10,250], [300,540], phases, beats)['time_warp']
    core = dict(source_uid='sourceA', source_interval=[10,250], target_interval=[300,540],
                time_warp=warp, full_core_R_sha256='c'*64, source_R_sha256='d'*64,
                phase_inputs=phases)
    pair = dict(key='pairA', audio_sha256='e'*64, audio_path='audio.wav', beats=beats)
    science = dict(cases=[pair], records=[dict(key='recordA',gap_frames=48,timing_offset_frames=0,left=core)],
                   selections=[dict(case_key='pairA',policy='Fixed48',record_key='recordA')])
    science_path = parent/'main/result.json'
    write(science_path, science)
    unit_path = repo/'outputs/event_coverage_r47_01/index/result.json'
    write(unit_path, dict(units=[unit]))
    roster_case = dict(key='caseA', parent_case_key='pairA',parent_record_key='recordA',parent_side='left',
        source_uid='sourceA',event_uid='unitA',source_interval=[10,250],support_component=[0,500],
        target_interval=[300,540],music_interval=[300,540],time_warp=warp,baseline_core_R_sha256='c'*64,
        music_id='songA',audio_sha256='e'*64,raw_source_sha256='a'*64,
        native_reference=dict(path='native.npy',sha256='b'*64))
    roster = dict(cases=[roster_case],parent_main_sha256=m.sha(science_path),
        parent_controls_sha256='f'*64,parent_viewer_sha256='',
        source_interval_count=1,cross_music_repeats=0,source_count=1,music_count=1,
        selection_rule='TEST_FIXTURE identity-only')
    registration = tmp_path/'registration'
    write(registration/'roster.json', roster)
    monkeypatch.setattr(m, 'HERE', registration)
    return dict(repo=repo,parent=parent,unit_path=unit_path,unit=unit,science_path=science_path,
                pair=pair,core=core,roster=roster,registration=registration,beats=beats)


def test_scientific_metadata_preserves_absolute_beats_and_frozen_map(frozen):
    before = frozen['science_path'].read_bytes()
    result = m.build(frozen['repo'])
    case = result['cases'][0]
    assert case['beats'] == frozen['beats']
    assert case['target_interval'] == [300,540]  # absolute music frames, not local0..239
    assert case['time_warp'] == frozen['core']['time_warp']
    assert case['original_E1W_exact'] and case['transition_budget_frames'] == 48
    assert not case['transition_regenerated'] and not case['previous_bridge_qualification_inherited']
    assert result['geometry_bridge_solves'] == 0 and result['shortfall'] == 11
    assert frozen['science_path'].read_bytes() == before


def test_viewer_metadata_restores_absolute_beats_with_verified_payload(frozen):
    frozen['science_path'].unlink()
    pair = dict(key='pairA', music_interval=[300,828], beats_local=[b-300 for b in frozen['beats']],
        music_clock=dict(source_audio_sha256='e'*64),
        panels=[dict(name='Fixed48',record_key='recordA',source_clock=dict(left=frozen['core']))])
    payload = dict(cases=[pair],displayed_result_sha256='f'*64)
    path = frozen['parent']/'controls_viewer/index.html'
    path.parent.mkdir(parents=True)
    path.write_text('<script id="r64-data" type="application/json">'+json.dumps(payload)+'</script>',encoding='utf8')
    frozen['roster']['parent_viewer_sha256'] = m.sha(path)
    write(frozen['registration']/'roster.json',frozen['roster'])
    result = m.build(frozen['repo'])
    assert result['cases'][0]['beats'] == frozen['beats']
    assert result['source_status'].endswith('VIEWER_METADATA_ONLY')


def test_aliased_unit_uid_cannot_change_raw_source_identity(frozen):
    frozen['unit']['source_sha256'] = '0'*64
    write(frozen['unit_path'],dict(units=[frozen['unit']]))
    with pytest.raises(ValueError,match='unit/source support'):
        m.build(frozen['repo'])


def test_changed_scientific_record_rejected_before_any_metadata_reuse(frozen):
    frozen['science_path'].write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='science result changed'):
        m.build(frozen['repo'])
