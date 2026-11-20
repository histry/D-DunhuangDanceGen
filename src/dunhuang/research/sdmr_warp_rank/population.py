"""Metadata-only R59 exposed-Val roster; no outcome-dependent replacement."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


PARENT_SHA = '5d66076438baba182176576de91fb7ca53c7398b612db3211e1cb461cede9c38'
FRAMES = 240


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()


def _center(source):
    lo, hi = source['shared_start'], source['frames']
    if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi:
        raise ValueError('Actual integer shared source support required')
    if source['shared_support']['frames'] != hi:
        raise ValueError('Frozen R59 source support metadata differs')
    if hi-lo < FRAMES:
        return None
    a = lo+(hi-lo-FRAMES)//2
    return [a, a+FRAMES]


def paired(manifest):
    """Freeze the two already-queried recordings in each of six Val groups.

    ``manifest`` is the original file path, so its bytes can be checked against
    the published R59 seal before parsing. A new center crop is not an original
    R59 query. This does not read waveform, motion or feature arrays.
    """
    path = Path(manifest)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError('Frozen R59 paired-population manifest bytes changed')
    meta = json.loads(raw.decode('utf-8-sig'))
    rows, queries = meta['rows'], meta['queries']
    sources = {s['source_uid']: s for s in meta['sources']}
    if len(sources) != len(meta['sources']):
        raise ValueError('Ambiguous frozen source UID')
    if len(set(queries)) != len(queries) or any(type(i) is not int or not 0 <= i < len(rows) for i in queries):
        raise ValueError('Unique original registered query indices required')
    groups = {}
    for i in queries:
        row = rows[i]
        if (row['dataset'], row['role']) != ('AIST++', 'val'):
            continue
        source = sources[row['source_uid']]
        for field in ('dataset', 'role', 'group', 'recording_uid', 'raw_motion_sha256', 'audio_sha256'):
            if row[field] != source[field]:
                raise ValueError('Original query/source identity mismatch: '+field)
        groups.setdefault(row['group'], {}).setdefault(row['source_uid'], []).append(i)
    if len(groups) != 6 or any(len(s) != 2 for s in groups.values()):
        raise ValueError('Keep exactly six historical Val groups and their two registered query sources')
    ordered = {g: sorted(uids, key=lambda u: (sources[u]['raw_motion_sha256'], u))
               for g, uids in sorted(groups.items())}
    names = list(ordered)
    cases, excluded = [], []
    for group, uids in ordered.items():
        for uid in uids:
            source = sources[uid]
            interval = _center(source)
            offset = source['audio_offset_frames']
            if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(offset):
                raise ValueError('Finite calibrated source-to-audio offset required')
            donor_group = names[(names.index(group)+1) % len(names)]
            donor_uid = ordered[donor_group][0]
            donor = sources[donor_uid]
            donor_interval = _center(donor)
            if donor['audio_sha256'] == source['audio_sha256']:
                raise ValueError('Different Val music groups must provide genuinely different audio bytes')
            original = sorted(groups[group][uid], key=lambda i: (rows[i]['source_interval'], rows[i]['uid']))
            audio_interval = [v+offset for v in interval] if interval else None
            key = 'r67paired_'+digest([uid, interval, source['audio_sha256']])[:24]
            case = dict(key=key, dataset='AIST++', role='val', group=group,
                source_uid=uid, recording_uid=source['recording_uid'],
                choreography_uid=source.get('choreography_uid', 'UNKNOWN'),
                raw_motion_sha256=source['raw_motion_sha256'], audio_sha256=source['audio_sha256'],
                source_interval=interval, target_interval=audio_interval, audio_interval=audio_interval,
                frames=FRAMES, source_audio_offset_frames=offset,
                original_query_indices=original, original_query_uids=[rows[i]['uid'] for i in original],
                original_query_intervals=[deepcopy(rows[i]['source_interval']) for i in original],
                crop_scope='NEW_METADATA_CENTER240_FROM_REGISTERED_R59_FULL_RECORD_NOT_ORIGINAL150_QUERY',
                source=deepcopy(source), sourceMnormTrue=True,
                music_values_scope='R59_PRENORMALIZED_ON_CALIBRATED_SOURCE_CLOCK_OFFSET_ALREADY_APPLIED_ONCE',
                eligible=interval is not None and donor_interval is not None,
                replacement_used=False, selection_uses_scores=False,
                donor=dict(source_uid=donor_uid, source=deepcopy(donor), group=donor_group,
                    dataset=donor['dataset'], role=donor['role'], audio_sha256=donor['audio_sha256'],
                    source_interval=donor_interval,
                    audio_interval=[v+donor['audio_offset_frames'] for v in donor_interval] if donor_interval else None,
                    sourceMnormTrue=True,
                    selection='NEXT_LEXICOGRAPHIC_GROUP_FIRST_RAW_SHA_RECORDING_OWN_CENTER240'),
                exposure='SIX_HISTORICALLY_EXPOSED_AIST_VAL_MUSIC_GROUPS',
                clock_scope=source['clock_scope'], semantic_event_truth='UNKNOWN', human_reviewed=False,
                paired_truth='RECORDED_SOURCE_CLOCK_NOT_UNIQUE_ARTISTIC_OR_EVENT_CORRESPONDENCE',
                root_contact_truth='UNKNOWN', padding_frames=0, circular_wrapping=False)
            if not case['eligible']:
                excluded.append(dict(key=key, source_uid=uid,
                    reason='FIXED_SOURCE_OR_FIXED_DONOR_HAS_INSUFFICIENT_SHARED240_SUPPORT', replacement_used=False))
            cases.append(case)
    result = dict(cases=cases, sources={u:deepcopy(sources[u]) for ids in ordered.values() for u in ids},
        excluded=excluded, actual_cases=len(cases), eligible_cases=sum(c['eligible'] for c in cases),
        parent_manifest_path=str(path), parent_manifest_sha256=PARENT_SHA,
        original_registered_queries=len(queries), selected_source_count=12, music_groups=6,
        registration=dict(version='R67_FIXED_R59_QUERY_RECORDINGS_CENTER240_V1',
            frames=FRAMES, source_rule='ORIGINAL_R59_VAL_QUERY_SOURCES_ONLY_NO_SUBSTITUTES',
            order='GROUP_THEN_RAW_MOTION_SHA_THEN_SOURCE_UID',
            crop='SHARED_START_FLOOR((SHARED_END-SHARED_START-240)/2)',
            wrong_music='NEXT_SORTED_GROUP_FIRST_RAW_SHA_SOURCE_OWN_SHARED_CENTER240',
            scores_or_activity_consulted=False, sourceMnormTrue=True),
        known_cross_role_choreography=deepcopy(meta['shared_choreography_across_AIST_roles']),
        new_independent_music_certified=False, unseen_pretraining_lineage_certified=False,
        test_tensor_reads=0, training_updates=0, new_planner_runs=0, can_route=False)
    result['population_sha256'] = digest(result)
    return result
