"""Twelve identity-selected Fixed48 Cores; no new choreography search.

The roster has eight distinct source intervals and four cross-music repeats.
Changing the internal map changes boundary jets. The reserved48-frame bridge
budget is not a renewed qualification of any previously constructed bridge.
"""
from copy import deepcopy
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re

from dunhuang.research.complete_r36.core import multi_warp


HERE = Path(__file__).parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _paths(parentroot):
    root = Path(parentroot).resolve()
    if (root/'src/dunhuang/research/complete_r36/core.py').is_file():
        return root, root/'outputs/timing_r64_01'
    return root.parent.parent, root


def _viewer(parent, roster):
    path = parent/'controls_viewer/index.html'
    if sha(path) != roster['parent_viewer_sha256']:
        raise ValueError('Frozen R64 viewer bytes differ from roster registration')
    text = path.read_text(encoding='utf8')
    match = re.search(r'<script[^>]*id="r64-data"[^>]*>(.*?)</script>', text, re.S)
    if match is None:
        raise ValueError('Frozen actual R64 metadata payload missing')
    payload = json.loads(match.group(1))
    if payload['displayed_result_sha256'] != roster['parent_controls_sha256']:
        raise ValueError('Viewer is not bound to the registered R64 science result')
    return payload


def _units(repo, roster):
    index = repo/'outputs/event_coverage_r47_01/index/result.json'
    if index.is_file():
        return {u['unit_uid']: u for u in read(index)['units']}
    # Read-only local metadata fallback. No coordinates leave this process.
    path = repo/'outputs/event_coverage_r47_01/viewer/data.js'
    if sha(path) != roster['R47_viewer_metadata_sha256']:
        raise ValueError('R47 fallback metadata differs from frozen roster source')
    text = path.read_text(encoding='utf8')
    encoded = json.loads(text[text.index('=')+1:].strip().removesuffix(';'))
    payload = json.loads(gzip.decompress(base64.b64decode(encoded)))
    units = {}
    for source in payload['cases']:
        for unit in source['events']:
            uid = unit['unit_uid']
            if uid in units and units[uid] != unit:
                raise ValueError('R47 unit identity appears with conflicting metadata')
            units[uid] = unit
    return units


def _phases(unit):
    phases = deepcopy(unit['phases'])
    for phase in phases:
        body = phase.get('body_region', '')
        if body and not phase['type'].startswith(body+'_'):
            phase['type'] = body+'_'+phase['type']
    return phases


def build(parentroot):
    """Return a metadata manifest; prefer scientific records over watch metadata."""
    repo, parent = _paths(parentroot)
    roster = read(HERE/'roster.json')
    scientific = parent/'main/result.json'
    cases = deepcopy(roster['cases'])
    if scientific.is_file():
        if sha(scientific) != roster['parent_main_sha256']:
            raise ValueError('R64 main science result changed')
        source = read(scientific)
        pairs = {c['key']: c for c in source['cases']}
        records = {r['key']: r for r in source['records']}
        selections = {(s['case_key'], s['policy']): s for s in source['selections']}
        mode = 'FROZEN_R64_MAIN_SCIENTIFIC_RECORDS'
    else:
        source = _viewer(parent, roster)
        pairs = {c['key']: c for c in source['cases']}
        records = selections = None
        mode = 'FROZEN_R64_HASH_VERIFIED_VIEWER_METADATA_ONLY'
    units = _units(repo, roster)
    for case in cases:
        pair = pairs[case['parent_case_key']]
        side = case['parent_side']
        if records is not None:
            selected = selections[(pair['key'], 'Fixed48')]
            if selected['record_key'] != case['parent_record_key']:
                raise ValueError('Frozen Fixed48 selection changed')
            record = records[selected['record_key']]
            if record['gap_frames'] != 48 or record['timing_offset_frames'] != 0:
                raise ValueError('A nonbaseline allocation entered the Core roster')
            meta = record[side]
            music_sha = pair['audio_sha256']
            phase_input = deepcopy(meta['phase_inputs'])
            beats = pair['beats']
        else:
            panels = [p for p in pair['panels'] if p['name'] == 'Fixed48']
            if len(panels) != 1 or panels[0]['record_key'] != case['parent_record_key']:
                raise ValueError('Frozen viewer Fixed48 identity differs')
            meta = panels[0]['source_clock'][side]
            music_sha = pair['music_clock']['source_audio_sha256']
            phase_input = _phases(units[case['event_uid']])
            beats = [b+pair['music_interval'][0] for b in pair['beats_local']]
        for field in ('source_uid', 'source_interval', 'target_interval', 'time_warp'):
            if meta[field] != case[field]:
                raise ValueError('Frozen Core field changed: '+field)
        if meta['full_core_R_sha256'] != case['baseline_core_R_sha256'] or music_sha != case['audio_sha256']:
            raise ValueError('Frozen Core or actual music identity differs')
        unit = units[case['event_uid']]
        if (unit['source_uid'] != case['source_uid'] or unit['source_interval'] != case['source_interval']
                or unit['support_component'] != case['support_component']
                or unit['source_sha256'] != case['raw_source_sha256']
                or unit['native_sha256'] != case['native_reference']['sha256']):
            raise ValueError('Runtime R47 unit/source support differs from roster')
        if phase_input != _phases(unit):
            raise ValueError('Frozen parent phases differ from the original R47 unit')
        warp = multi_warp(case['source_interval'], case['target_interval'], phase_input, beats)
        if warp is None or warp['time_warp'] != case['time_warp']:
            raise ValueError('Original Fixed48 E1-W map is not exactly reproducible')
        if case['target_interval'][1]-case['target_interval'][0] != 240:
            raise ValueError('Every fixed Core must retain exactly240 actual frames')
        case.update(phase_inputs=phase_input, phases=deepcopy(unit['phases']), beats=deepcopy(beats),
            audio_path=pair.get('audio_path', 'outputs/complete_r36_01/viewer/audio/'+case['music_id']+'.wav'),
            source_R_sha256=meta.get('source_R_sha256'),
            source_R_identity=dict(raw_source_sha256=case['raw_source_sha256'],
                native151_sha256=case['native_reference']['sha256'],
                decode='dunhuang.contracts.matrices(native[:,7:139].reshape(-1,22,6))',
                full_source_matrix_sha256=meta.get('source_R_sha256'),
                matrix_sha_status='FROZEN_SCIENTIFIC_RECORD' if meta.get('source_R_sha256') else 'PENDING_RUNTIME_LIBRARY_HASH'),
            original_E1W_exact=True, source_endpoints_unchanged=True,
            transition_budget_frames=48, transition_regenerated=False,
            previous_bridge_qualification_inherited=False,
            world_root_contact='UNKNOWN', human_reviewed=False, Native24='NOT_EVALUABLE')
    return dict(cases=cases, requested_cases=12, actual_cases=len(cases),
        source_interval_count=roster['source_interval_count'], cross_music_repeats=roster['cross_music_repeats'],
        source_count=roster['source_count'], music_count=roster['music_count'],
        selection_rule=roster['selection_rule'], source_status=mode,
        parent_sha=roster['parent_main_sha256'], parent_viewer_sha256=roster['parent_viewer_sha256'],
        roster_sha256=sha(HERE/'roster.json'), shortfall=max(0,12-len(cases)),
        source_phase_truth='KINEMATIC_PROXY_ONLY_NOT_SEMANTIC_COMPLETENESS',
        unique_source_intervals=roster['source_interval_count'], recording_independence_verified=False,
        music_exposure='HISTORICALLY_EXPOSED_THREE_TRACKS',
        new_planner_searches=0, geometry_bridge_solves=0, baseline_changed=False, can_route=False)


def load_assets(parentroot=None):
    """Load the original Native library/skeleton through frozen R63 checks.

    Invoke from the repository root. R63 population.build recompiles source
    references and checks saved identities, but performs no Bridge solve.
    """
    if parentroot is not None and _paths(parentroot)[0] != Path.cwd().resolve():
        raise ValueError('Call frozen R63 asset loading from the repository root')
    from dunhuang.research.budget_r63.population import build as frozen_build
    _, lib, J, parents, _, manifest = frozen_build()
    return dict(lib=lib, J=J, parents=parents, frozen_population_manifest=manifest,
                geometry_bridge_solves=0)
