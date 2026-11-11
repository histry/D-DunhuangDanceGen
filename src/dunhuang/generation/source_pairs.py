"""Independent recordings may be reference conditions, never substitute targets."""
from collections import defaultdict


def source_pairs(manifest, roles=('train', 'val')):
    if len(set(roles)) != len(roles) or not set(roles) <= {'train', 'val'}:
        raise ValueError('Only explicit Train/Val source tasks are permitted')
    rows = manifest['rows']
    by_id = {r['window_uid']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate target window identity')
    groups = defaultdict(list)
    for r in rows:
        if r['role'] not in roles:
            continue
        if not r.get('recording_uid') or not r.get('choreography_uid'):
            raise ValueError('Official source identities required')
        key = (r['role'], r['choreography_uid'], r['music_uid'],
               r['audio']['sha256'], r['start_frame'], r['audio_offset_seconds'])
        groups[key].append(r)
    result = []
    for rs in groups.values():
        for target in sorted(rs, key=lambda r: r['window_uid']):
            candidates = [s for s in rs if s['recording_uid'] != target['recording_uid']
                          and s['raw_motion']['sha256'] != target['raw_motion']['sha256']]
            if not candidates:
                continue
            source = min(candidates, key=lambda r: r['window_uid'])
            pair = dict(target_uid=target['window_uid'], source_uid=source['window_uid'],
                        role=target['role'], source_recording=source['recording_uid'],
                        target_recording=target['recording_uid'],
                        relation='official_shared_basic_routine_same_music_same_clock',
                        pose_exact_correspondence=False)
            validate_pair(pair, by_id)
            result.append(pair)
    return sorted(result, key=lambda r: r['target_uid'])


def validate_pair(pair, rows):
    t, s = rows[pair['target_uid']], rows[pair['source_uid']]
    if t['role'] != s['role'] or t['role'] != pair['role'] or t['role'] == 'test':
        raise ValueError('Cross-role/Test reference leakage')
    if t['recording_uid'] == s['recording_uid'] or t['raw_motion']['sha256'] == s['raw_motion']['sha256']:
        raise ValueError('Independent recording required')
    for key in ('choreography_uid', 'music_uid', 'start_frame', 'audio_offset_seconds', 'fps', 'frames'):
        if t[key] != s[key]:
            raise ValueError('Source-target correspondence differs: ' + key)
    if t['audio']['sha256'] != s['audio']['sha256']:
        raise ValueError('Source-target audio differs')
    if t['dataset'] != 'aistpp' or s['dataset'] != 'aistpp' or '_sBM_' not in t['recording_uid']:
        raise ValueError('Only authority-verified basic routines admitted')
    return True
