"""Frozen R62 development pairs and source-disjoint, exposed R63 pairs.

The new split concerns only fitting the duration rule. All recordings and all
three music tracks have project-development exposure. No motion outcome is
consulted when choosing an evaluation unit or assigning its music interval.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from dunhuang.research.complete_r36.core import Library, identity, read, sha
from dunhuang.research.duration_r62.cross_source import build_roster
from dunhuang.research.warp_coupling_r53.common import array_hash
from dunhuang.research.warp_coupling_r53.population import sealed


R47 = Path('outputs/event_coverage_r47_01/index/result.json')
R51 = Path('outputs/validation_r51_01/controls/plans/result.json')
R62 = Path('outputs/duration_r62_01/main/cross_source')
MUSIC_ROOT = Path('outputs/complete_r36_01')
DEV_SOURCES = ('female_FeiTian', 'male_pipa_2', 'male_ribbon_FenHe')
BASELINE = dict(baseline_left_frames=240, baseline_gap_frames=48,
                baseline_right_frames=240, total_frames=528)
EXPOSURE = 'HISTORICALLY_EXPOSED_SOURCE_AND_MUSIC; R63_POLICY_FIT_SOURCE_DISJOINT_ONLY'


def native(value):
    """Public metadata must contain JSON-native scalars, including frame IDs."""
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def _side(unit):
    return native(dict(event_uid=unit['unit_uid'], source_uid=unit['source_uid'],
                       source_interval=unit['source_interval'],
                       support_component=unit['support_component'],
                       phases=deepcopy(unit.get('phases', [])),
                       source_role=unit.get('role', 'UNKNOWN'),
                       semantic_completeness='UNKNOWN', world_root='UNKNOWN', contact='UNKNOWN'))


def role_hash_isolation(source_rows, dev_source_uids, eval_source_uids):
    """Reject byte-identity leakage through aliased source UIDs across roles.

    Native identity is mandatory because Library verifies the referenced file.
    Raw source SHA is also checked wherever catalog entries provide it. Byte
    disjointness does not establish recording or choreography independence.
    """
    roles = {'DEV': sorted(set(dev_source_uids)), 'EVAL': sorted(set(eval_source_uids))}
    if set(roles['DEV']) & set(roles['EVAL']):
        raise ValueError('Source UID appears in both R63 roles')
    hashes, missing_raw = {}, {}
    for role, uids in roles.items():
        hashes[role] = dict(native_sha256={}, raw_source_sha256={})
        missing_raw[role] = []
        for uid in uids:
            if uid not in source_rows or not source_rows[uid]:
                raise ValueError('Registered source lacks a Native reference: '+uid)
            native_values, raw_values = set(), set()
            for entry in source_rows[uid]:
                value = entry.get('reference', {}).get('sha256')
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
                    raise ValueError('Native SHA256 identity missing or malformed: '+uid)
                native_values.add(value.lower())
                value = entry.get('source_sha256')
                if value is None or value == 'UNKNOWN':
                    continue
                if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdefABCDEF' for c in value):
                    raise ValueError('Raw source SHA256 identity malformed: '+uid)
                raw_values.add(value.lower())
            if len(native_values) != 1:
                raise ValueError('Source UID references multiple Native identities: '+uid)
            hashes[role]['native_sha256'][uid] = sorted(native_values)
            hashes[role]['raw_source_sha256'][uid] = sorted(raw_values)
            if not raw_values:
                missing_raw[role].append(uid)
    overlaps = {}
    for kind in ('native_sha256', 'raw_source_sha256'):
        dev = {v for values in hashes['DEV'][kind].values() for v in values}
        evaluation = {v for values in hashes['EVAL'][kind].values() for v in values}
        overlaps[kind] = sorted(dev & evaluation)
        if overlaps[kind]:
            raise ValueError('Cross-role '+kind+' alias detected despite distinct source UIDs')
    return dict(roles=hashes, overlaps=overlaps, native_hash_disjoint=True,
                available_raw_source_hash_disjoint=True,
                raw_source_hash_coverage_complete=not any(missing_raw.values()),
                raw_source_hash_missing=missing_raw, recording_independence_verified=False,
                scope='UID_AND_AVAILABLE_BYTE_IDENTITY_ROLE_SEPARATION_ONLY; HISTORICAL_EXPOSURE_RETAINED')


def select_eval_units(units, source_uids, dev_source_uids=DEV_SOURCES):
    """Choose once by length distance, then chronology; never replace a reject.

    Baseline legality is recorded after selection. In particular, an illegal
    nearest-length unit does not cause a search for a more convenient unit.
    Missing sources remain in the returned roster as NOT_MATCHABLE.
    """
    development = set(dev_source_uids)
    result = []
    for uid in sorted(set(source_uids)-development):
        pool = [u for u in units if u['source_uid'] == uid]
        if not pool:
            result.append(dict(source_uid=uid, status='NOT_MATCHABLE', unit=None,
                               reason='NO_R47_UNIT_FOR_REGISTERED_SOURCE', replacement_used=False))
            continue
        chosen = min(pool, key=lambda u: (
            abs(int(u['source_interval'][1])-int(u['source_interval'][0])-240),
            int(u['source_interval'][0]), int(u['source_interval'][1]), u['unit_uid']))
        length = int(chosen['source_interval'][1])-int(chosen['source_interval'][0])
        ratio = 239/(length-1) if length > 1 else None
        legal = ratio is not None and .75-1e-10 <= ratio <= 1.33+1e-10
        result.append(dict(source_uid=uid, status='SELECTED', unit=_side(chosen),
                           selection_distance_frames=abs(length-240), source_frames=length,
                           baseline_endpoint_rate=ratio, baseline_length_legal=bool(legal),
                           replacement_used=False,
                           selection='MIN_ABS_SOURCE_DURATION_MINUS_240_THEN_SOURCE_CHRONOLOGY_UID'))
    return native(result)


def evaluation_cases(selected, tracks):
    """All ordered, distinct-source pairs; fixed track assignment before solves."""
    if len(tracks) != 3:
        raise ValueError('Exactly three frozen historical music tracks required')
    rows = sorted(selected, key=lambda row: row['source_uid'])
    cases = []
    for left in rows:
        for right in rows:
            if left['source_uid'] == right['source_uid']:
                continue
            i = len(cases)
            song = i % 3
            track = tracks[song]
            frames = int(track['structure']['frames'])
            if frames < 528:
                raise ValueError('Frozen track is shorter than the registered 528-frame budget')
            start = (frames-528)//2
            interval = [start, start+528]
            source_pair = dict(left_source_uid=left['source_uid'], right_source_uid=right['source_uid'])
            pair_uid = identity(source_pair)
            matchable = left['unit'] is not None and right['unit'] is not None
            case = dict(key='eval%02d_' % i+pair_uid, role='EVAL', case=i,
                        pair_uid=pair_uid, source_pair_uid=pair_uid, **source_pair,
                        left=deepcopy(left['unit']), right=deepcopy(right['unit']),
                        status='REGISTERED' if matchable else 'NOT_MATCHABLE',
                        missing_source_uids=[s['source_uid'] for s in (left, right) if s['unit'] is None],
                        songidx=song, song=song, music_interval=interval,
                        beats=list(track['structure']['beats']),
                        audio_path=track['path'], audio_sha256=track['audio_sha256'],
                        music_structure_sha256=track['structure_sha256'],
                        music_assignment='ORDERED_SOURCE_PAIR_INDEX_MOD_3; CENTER_528_FRAMES',
                        evaluation_exposure=EXPOSURE, **BASELINE)
            case['case_identity'] = identity(dict(source_pair=source_pair,
                left=case['left'], right=case['right'], music_interval=interval,
                audio_sha256=case['audio_sha256']))
            cases.append(native(case))
    return cases


def _tracks():
    tracks = []
    for song in range(3):
        audio = sealed(MUSIC_ROOT, 'viewer/audio/dunhuangwu%d.wav' % (song+2))
        music = sealed(MUSIC_ROOT, 'planning/song%d_music.json' % song)
        structure = read(music)
        if int(structure['frames']) < 528 or not np.isfinite(structure['beats']).all():
            raise ValueError('Real music clock and finite registered beat list required')
        tracks.append(dict(path=str(audio), audio_sha256=sha(audio),
                           structure=structure, structure_sha256=sha(music)))
    return tracks


def build():
    """Return cases, Native-verified library, skeleton, gate and sealed manifest.

    Development cores are independently rebuilt by the unchanged R62 roster
    code and checked against saved R62 arrays. New EVAL identities are selected
    without reading R62 outcomes or executing any R63 candidate geometry.
    """
    config_path = Path('src/dunhuang/research/complete_r36/config.json')
    config = read(config_path)
    if sha(config['catalog']) != config['catalog_sha256']:
        raise ValueError('Frozen Native catalog hash changed')
    lib = Library(config['catalog'])
    index = read(R47)
    units = {u['unit_uid']: u for u in index['units']}
    for unit in units.values():
        uid = unit['source_uid']
        if uid not in lib.rows or unit.get('native_sha256') != lib.rows[uid][0]['reference']['sha256']:
            raise ValueError('R47 unit and loaded Native source identity differ')
    old = read(R62/'result.json')
    roster, J, parents, evidence = build_roster(limit=12)
    metadata = [{k: v for k, v in row.items() if k not in ('left_core', 'right_core')} for row in roster]
    if native(metadata) != old['roster'] or len(roster) != 9:
        raise ValueError('R62 nine development pair identities changed')
    dev_sources = sorted({p[s]['source_uid'] for p in roster for s in ('left', 'right')})
    if dev_sources != sorted(DEV_SOURCES):
        raise ValueError('R62 development source population changed')
    tracks = _tracks()
    plans = {r['key']: r for r in read(R51)['records']}
    cases = []
    with np.load(R62/'arrays.npz', allow_pickle=False) as saved:
        if not np.array_equal(J, saved['J']) or not np.array_equal(parents, saved['parents']):
            raise ValueError('R62 skeleton changed')
        for p in roster:
            i = int(p['case'])
            left_key, right_key = 'pair%d_left_core' % i, 'pair%d_right_core' % i
            if not (np.array_equal(p['left_core'], saved[left_key])
                    and np.array_equal(p['right_core'], saved[right_key])):
                raise ValueError('Independently rebuilt R62 cores differ from saved arrays')
            left, right = p['left'], p['right']
            for side in (left, right):
                unit = units[side['event_uid']]
                if (unit['source_uid'] != side['source_uid']
                        or unit['source_interval'] != side['source_interval']
                        or unit['support_component'] != side['support_component']):
                    raise ValueError('R62 frozen event differs from its R47 unit identity')
            A, B = int(left['target_interval'][0]), int(right['target_interval'][1])
            if (len(p['left_core']) != 240 or len(p['right_core']) != 240
                    or int(right['target_interval'][0])-int(left['target_interval'][1]) != 48
                    or B-A != 528):
                raise ValueError('R62 pair does not have the registered 240/48/240 budget')
            song = int(plans[p['frozen_plan_key']]['song'])
            track = tracks[song]
            case = dict(key='dev%02d_' % i+p['pair_uid'], case=i, role='DEV',
                        pair_uid=p['pair_uid'], source_pair_uid=identity(dict(
                            left_source_uid=left['source_uid'], right_source_uid=right['source_uid'])),
                        left=_side(units[left['event_uid']]), right=_side(units[right['event_uid']]),
                        status='REGISTERED', missing_source_uids=[],
                        songidx=song, song=song, music_interval=[A, B],
                        beats=list(track['structure']['beats']), audio_path=track['path'],
                        audio_sha256=track['audio_sha256'], music_structure_sha256=track['structure_sha256'],
                        frozen_plan_key=p['frozen_plan_key'], frozen_edge_index=int(p['frozen_edge_index']),
                        r62_arrays_path=str(R62/'arrays.npz'), r62_left_core_key=left_key,
                        r62_right_core_key=right_key, r62_case_key='cross%d_d48' % i,
                        r62_left_core_sha256=array_hash(p['left_core']),
                        r62_right_core_sha256=array_hash(p['right_core']),
                        r62_left=deepcopy(left), r62_right=deepcopy(right),
                        evaluation_exposure=EXPOSURE, **BASELINE)
            cases.append(native(case))
    selected = select_eval_units(index['units'], lib.rows, dev_sources)
    eval_cases = evaluation_cases(selected, tracks)
    eval_sources = sorted(row['source_uid'] for row in selected)
    if len(eval_sources) != 5 or len(eval_cases) != 20 or set(dev_sources) & set(eval_sources):
        raise ValueError('Expected five source-disjoint evaluation recordings and twenty directed pairs')
    role_hashes = role_hash_isolation(lib.rows, dev_sources, eval_sources)
    cases.extend(eval_cases)
    reference_path = Path('outputs/bridge_dynamics_r48_01/main/result.json')
    reference = read(reference_path)['reference']
    manifest = dict(selection_version='R63_ONCE_PER_SOURCE_CLOSEST_240_CHRONOLOGY_V1',
        exposure=EXPOSURE, development_sources=dev_sources, evaluation_sources=eval_sources,
        source_disjoint_for_R63_rule_fit=True, independent_unseen_sources=False,
        role_hash_isolation=role_hashes, recording_independence_verified=False,
        independent_unseen_music=False, development_pairs=9, evaluation_pairs=len(eval_cases),
        evaluation_units=selected, evaluation_case_order=[c['key'] for c in eval_cases],
        missing_units_retained=True, source_replacement_count=0,
        selection_uses_bridge_outcomes=False, selection_uses_hidden_target=False,
        r62_source_parity_exact=True, r62_roster_evidence=evidence,
        r62_result_sha256=sha(R62/'result.json'), r62_arrays_sha256=sha(R62/'arrays.npz'),
        R47_index_sha256=sha(R47), R51_result_sha256=sha(R51),
        source_config_sha256=sha(config_path), catalog_sha256=config['catalog_sha256'],
        native_sources={uid: lib.rows[uid][0]['reference'] for uid in sorted(lib.rows)},
        J_sha256=array_hash(J), parents_sha256=array_hash(parents),
        reference_sha256=sha(reference_path), original_numerical_gate=reference,
        tracks=[{k: v for k, v in t.items() if k != 'structure'} for t in tracks],
        baseline=BASELINE, can_route=False, human_reviewed=False, Native24='NOT_EVALUABLE')
    return cases, lib, J, parents, reference, native(manifest)
