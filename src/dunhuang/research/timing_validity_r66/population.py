"""Metadata-only replay of R59 registered queries; no waveform/model reads.

The source clock is an observed recording correspondence, not a unique musical
accent or artistic match. Actual raw audio sample bounds must still be checked
by the engine against the sealed WAV before any diagnostic is executed.
"""
from copy import deepcopy
import hashlib
import json
import math


ALLOWED = {('AIST++', 'train'), ('AIST++', 'val'),
           ('FineDance', 'genre_train_diagnostic')}
OFFSETS = (-15, -8, 0, 8, 15)


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf8')
    return hashlib.sha256(data).hexdigest()


def _interval(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('Two-element source interval required')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError('Finite source frame centers required')
    if not 0 <= value[0] < value[1]:
        raise ValueError('Positive half-open source interval required')
    return value


def _counts(rows):
    result = {}
    for row in rows:
        name = row['dataset']+'_'+row['role']
        result[name] = result.get(name, 0)+1
    return result


def select(meta, protocol):
    """Freeze existing query IDs and one common120-frame target for all shifts."""
    offsets = deepcopy(protocol.get('offsets', list(OFFSETS)))
    if offsets != list(OFFSETS):
        raise ValueError('R66 uses the registered [-15,-8,0,8,15] offsets only')
    seed = protocol.get('seed', 66066)
    maximum = protocol.get('wrong_song_donors', protocol.get('wrong_song_candidates', 3))
    if protocol.get('trim_each_side', 15) != 15 or protocol.get('fps', 30) != 30:
        raise ValueError('R66 requires 15-frame trim and the actual30FPS clock')
    if type(seed) is not int or type(maximum) is not int or not 1 <= maximum <= 3:
        raise ValueError('Integer seed and one to three metadata wrong-song candidates required')
    rows, indices = meta['rows'], meta['queries']
    sources = {s['source_uid']: s for s in meta['sources']}
    if len(sources) != len(meta['sources']):
        raise ValueError('Duplicate registered source UID')
    if (any(type(i) is not int or not 0 <= i < len(rows) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Unique valid existing R59 query indices required')
    if protocol.get('registered_query_count', len(indices)) != len(indices):
        raise ValueError('Frozen registered query count changed')
    # Representative selection uses identity only, not signal/activity scores.
    audio_pool = {}
    for source in sorted(meta['sources'], key=lambda s: (s['raw_motion_sha256'], s['source_uid'])):
        scope = (source['dataset'], source['role'], source['audio_sha256'])
        if scope[:2] in ALLOWED:
            audio_pool.setdefault(scope, source)
    cases, excluded, used_sources = [], [], {}
    for index in indices:
        row = rows[index]
        if (row['dataset'], row['role']) not in ALLOWED:
            raise ValueError('Test or unregistered role may not enter R66')
        source = sources[row['source_uid']]
        for field in ('dataset', 'role', 'group', 'recording_uid', 'raw_motion_sha256', 'audio_sha256'):
            if row[field] != source[field]:
                raise ValueError('Frozen row/source identity mismatch: '+field)
        a, b = _interval(row['source_interval'])
        A, B = _interval(row['audio_interval'])
        offset = source['audio_offset_frames']
        if (type(a) is not int or type(b) is not int or b-a != 150 or row['frames'] != 150
                or not math.isfinite(offset) or [A, B] != [a+offset, b+offset]):
            raise ValueError('R59 fixed150-frame source-to-audio clock changed')
        start, end = a+15, b-15
        audio_start, audio_end = start+offset, end+offset
        n, lo = source['frames'], source['shared_start']
        shifted = {str(delta): [audio_start+delta, audio_end+delta] for delta in offsets}
        valid = (source['shared_support']['frames'] == n and lo <= a < b <= n
                 and start >= lo and end <= n
                 and all(max(0, lo+offset) <= x < y <= n+offset for x, y in shifted.values())
                 and all(A <= x < y <= B for x, y in shifted.values()))
        if not valid:
            excluded.append(dict(query_index=index, query_uid=row['uid'], source_uid=row['source_uid'],
                reason='REGISTERED_COMMON_AUDIO_OR_MOTION_SUPPORT_INSUFFICIENT', replacement_used=False))
            continue
        pool = [s for (dataset, role, audio_sha), s in audio_pool.items()
                if (dataset, role) == (row['dataset'], row['role']) and audio_sha != row['audio_sha256']]
        pool.sort(key=lambda s: (digest([seed, row['uid'], s['audio_sha256']]), s['audio_sha256']))
        wrong = [dict(source_uid=s['source_uid'], audio=deepcopy(s['audio']),
            audio_sha256=s['audio_sha256'], dataset=s['dataset'], role=s['role'], group=s['group'],
            metadata_rank=i, rank_digest=digest([seed, row['uid'], s['audio_sha256']]),
            raw_audio_bounds='PENDING_ENGINE_ACTUAL_SAMPLE_HEADER_CHECK') for i, s in enumerate(pool[:maximum])]
        case = dict(key='r66_'+digest([row['uid'], [start, end], offsets])[:24],
            query_index=index, query_uid=row['uid'], source_uid=row['source_uid'],
            dataset=row['dataset'], role=row['role'], group=row['group'],
            recording_uid=row['recording_uid'], choreography_uid=row.get('choreography_uid', 'UNKNOWN'),
            raw_motion_sha256=row['raw_motion_sha256'], audio_sha256=row['audio_sha256'],
            source_interval=[a, b], audio_interval=[A, B],
            common_motion_interval=[start, end], common_audio_interval=[audio_start, audio_end],
            common_frames=120, source_audio_offset_frames=offset,
            shifted_audio_intervals=shifted, candidate_offsets_frames=offsets,
            source=deepcopy(source), wrong_song_candidates=wrong,
            registered_audio_and_motion_support_verified=True,
            raw_audio_sample_bounds='PENDING_ENGINE_ACTUAL_WAV_SHA_SAMPLE_RATE_AND_SAMPLE_COUNT_CHECK',
            wrong_song_selection='IDENTITY_HASH_TOP3_THEN_FROZEN_AUDIO_ONLY_RMS_TEMPO_ENGINE_RULE; NO_EXPANSION',
            exposure='HISTORICALLY_EXPOSED_AIST_TRAIN_VAL_OR_FINEDANCE_GENRE_TRAIN_DIAGNOSTIC',
            clock_scope=source['clock_scope'], phase_correspondence_truth='UNKNOWN',
            root_contact_truth='UNKNOWN', padding_frames=0, circular_wrapping=False)
        cases.append(case)
        used_sources[source['source_uid']] = deepcopy(source)
    if len({c['key'] for c in cases}) != len(cases):
        raise ValueError('Repeated frozen query identity')
    groups = {d+'_'+r: len({s['group'] for s in meta['sources'] if (s['dataset'], s['role']) == (d, r)})
              for d, r in sorted(ALLOWED)}
    registration = dict(version='R66_R59_EXISTING_QUERIES_COMMON120_V1', seed=seed,
        source_frames=150, edge_trim_frames=15, common_frames=120, offsets=offsets,
        wrong_song_candidates=maximum, wrong_song_rank='SHA256([seed,query_uid,candidate_audio_sha256])',
        wrong_song_pool='SAME_DATASET_AND_ROLE_DIFFERENT_AUDIO_SHA; UNIQUE_AUDIO_SHA',
        representative='LOWEST_RAW_MOTION_SHA_THEN_SOURCE_UID; METADATA_ONLY',
        original_query_order_preserved=True, new_query_windows=0, activity_scores_used=False,
        raw_audio_runtime_verification_required=True,
        wrong_song_runtime_rule=deepcopy(protocol.get('wrong_song_rule', protocol.get('wrong_song_runtime_rule'))))
    result = dict(registration=registration, input_r59_manifest_digest=digest(meta),
        protocol_digest=digest(protocol), cases=cases, sources=used_sources, excluded=excluded,
        frozen_query_indices=list(indices), registered_query_count=len(indices), selected_query_count=len(cases),
        registered_query_counts=_counts([rows[i] for i in indices]), selected_query_counts=_counts(cases),
        source_counts=deepcopy(meta.get('source_counts')), music_groups_by_role=groups,
        registered_source_count=len(sources), selected_source_count=len(used_sources),
        known_shared_choreography_across_AIST_roles=deepcopy(meta.get('shared_choreography_across_AIST_roles', [])),
        AIST_val_scope='HISTORICALLY_EXPOSED_SIX_MUSIC_GROUPS; NOT_UNSEEN_CHOREOGRAPHY',
        FineDance_scope='OFFICIAL_GENRE_TRAIN_DIAGNOSTIC; NOT_AIST_WAVEFORM_CALIBRATED; DANCER_ROLE_SEPARATE',
        new_independent_music_certified=False, pretrained_exposure_certified=False,
        test_tensor_reads=0, model_training_steps=0, new_planner_runs=0, semantic_truth='UNKNOWN', can_route=False)
    by_uid = {c['query_uid']: c for c in cases}
    display = []
    for dataset, role in sorted({(rows[i]['dataset'], rows[i]['role']) for i in indices}):
        chosen = min((rows[i] for i in indices if (rows[i]['dataset'], rows[i]['role']) == (dataset, role)), key=lambda r: r['uid'])
        case = by_uid.get(chosen['uid'])
        display.append(dict(dataset=dataset, role=role, query_uid=chosen['uid'],
                            case_key=case['key'] if case else None,
                            status='REGISTERED_BEFORE_SCORES' if case else 'EXCLUDED_SUPPORT_NO_DISPLAY_REPLACEMENT'))
    result['display_cases'] = display
    result['display_case_keys'] = [d['case_key'] for d in display if d['case_key'] is not None]
    result['display_rule'] = 'LEXICOGRAPHIC_REGISTERED_QUERY_UID_PER_DATASET_ROLE_BEFORE_SCORES; NO_REPLACEMENT'
    result['population_sha256'] = digest(result)
    return result
