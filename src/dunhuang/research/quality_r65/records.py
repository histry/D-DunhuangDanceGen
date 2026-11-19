"""R65 review contracts. Format validation never authenticates a human reviewer.

The administrator manifest is private: it contains method identities and blind
orders. Imported reviews remain unverified unless a separately checked private
roster is explicitly supplied. No review changes any R64 clock or qualification.
"""
from copy import deepcopy
from datetime import datetime
import math
import re

METHODS = ('Fixed48', 'BeatOnly', 'LocalBeat', 'ShuffledBeat')
DIMENSIONS = ('music_coordination', 'transition_timing', 'continuity', 'richness')
TRACKS = ('beat', 'downbeat', 'accent', 'phrase')
PROXIES = ('beat', 'accent', 'section', 'primary')
MUSIC_SCHEMA = 'R65_MUSIC_REVIEW_V1'
QUALITY_SCHEMA = 'R65_QUALITY_REVIEW_V1'
STATUSES = ('COMPLETE', 'PARTIAL', 'UNSURE', 'NO_CLEAR_EVENTS')
CONFIDENCES = ('HIGH', 'MEDIUM', 'LOW')
_SHA = re.compile(r'^[0-9a-fA-F]{64}$')
_EPS = 1e-9


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(name+' must be an object')
    return value


def _list(value, name):
    if not isinstance(value, list):
        raise ValueError(name+' must be a list')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(name+' must be a nonempty unpadded string')
    return value


def _sha(value, name):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(name+' must be a SHA256 hex digest')
    return value.lower()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name+' must be a finite number, not bool')
    return float(value)


def _span(value, name, duration=None):
    _list(value, name)
    if len(value) != 2:
        raise ValueError(name+' must contain two half-open seconds')
    a, b = [_number(v, name) for v in value]
    if a < 0 or b <= a or duration is not None and b > duration+_EPS:
        raise ValueError(name+' leaves actual media support')
    if duration is not None and abs(b-duration) <= _EPS:
        b = duration
    return [a, b]


def _unique(values, name):
    _list(values, name)
    for value in values:
        _text(value, name)
    if len(set(values)) != len(values):
        raise ValueError(name+' contains duplicates')
    return values


def admin_index(admin):
    """Check manifest identities/orders; preserve extra administrator metadata."""
    _mapping(admin, 'admin')
    _text(admin.get('pack_id'), 'pack_id')
    _sha(admin.get('protocol_sha256'), 'protocol_sha256')
    cases, music, forms = {}, {}, {}
    for row in _list(admin.get('music'), 'admin.music'):
        _mapping(row, 'music row')
        key = _text(row.get('blind_item_id'), 'music blind_item_id')
        if key in music:
            raise ValueError('Duplicate music blind_item_id')
        _sha(row.get('raw_audio_sha256'), 'raw_audio_sha256')
        _sha(row.get('media_sha256'), 'media_sha256')
        if _number(row.get('duration_seconds'), 'duration_seconds') <= 0:
            raise ValueError('Positive music duration required')
        music[key] = row
    for row in _list(admin.get('cases'), 'admin.cases'):
        _mapping(row, 'case row')
        key = _text(row.get('blind_item_id'), 'case blind_item_id')
        if key in cases:
            raise ValueError('Duplicate case blind_item_id')
        _text(row.get('case_key'), 'case_key')
        _text(row.get('role'), 'role')
        music_id = _text(row.get('music_id'), 'music_id')
        if music_id not in music:
            raise ValueError('Case music_id does not reference administrator music')
        _span(row.get('music_interval_seconds'), 'music_interval_seconds', music[music_id]['duration_seconds'])
        variants = _list(row.get('variants'), 'variants')
        for variant in variants:
            _mapping(variant, 'variant')
        if len(variants) != 4 or {v.get('method') for v in variants} != set(METHODS):
            raise ValueError('Exactly the four frozen method identities required')
        _unique([v.get('blind_variant_id') for v in variants], 'variant IDs')
        for variant in variants:
            _sha(variant.get('trajectory_sha256'), 'trajectory_sha256')
            proxies = _mapping(variant.get('proxies'), 'proxies')
            if set(proxies) != set(PROXIES):
                raise ValueError('Exactly the four frozen proxy fields required')
            for value in proxies.values():
                if value is not None:
                    _number(value, 'proxy value')
        cases[key] = row
    for row in _list(admin.get('forms'), 'admin.forms'):
        _mapping(row, 'form row')
        key = _text(row.get('form_id'), 'form_id')
        if key in forms:
            raise ValueError('Duplicate form_id')
        order = _unique(row.get('case_order'), 'case_order')
        if set(order) != set(cases):
            raise ValueError('Form must preserve the whole registered case population')
        panels = _mapping(row.get('panel_orders'), 'panel_orders')
        if set(panels) != set(cases):
            raise ValueError('Panel order must cover every registered case')
        for case_id, order in panels.items():
            _unique(order, 'panel order')
            expected = {v['blind_variant_id'] for v in cases[case_id]['variants']}
            if len(order) != 4 or set(order) != expected:
                raise ValueError('Panel order differs from the sealed variants')
        forms[key] = row
    if not forms:
        raise ValueError('At least one sealed review form required')
    return cases, music, forms


def validate_document(doc, admin):
    """Return a detached, JSON-native normalized review, or raise ValueError.

    Partial reviews are valid records. Their content is never silently promoted
    into complete annotations or used to fill unreviewed intervals with negatives.
    """
    cases, music, forms = admin_index(admin)
    _mapping(doc, 'review')
    out = deepcopy(doc)
    schema = out.get('schema')
    if schema not in (MUSIC_SCHEMA, QUALITY_SCHEMA):
        raise ValueError('Unknown R65 review schema')
    if out.get('pack_id') != admin['pack_id'] or _sha(out.get('protocol_sha256'), 'protocol_sha256') != admin['protocol_sha256'].lower():
        raise ValueError('Review pack/protocol identity mismatch')
    for field in ('review_id', 'reviewer_id', 'reviewer_roster_ref', 'form_id'):
        _text(out.get(field), field)
    if out['form_id'] not in forms:
        raise ValueError('Unknown sealed form')
    value = _text(out.get('reviewed_at'), 'reviewed_at')
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, TypeError) as error:
        raise ValueError('reviewed_at must be ISO8601 with timezone') from error
    if stamp.utcoffset() is None:
        raise ValueError('reviewed_at requires timezone')
    out['reviewed_at'] = stamp.isoformat()
    independence = _mapping(out.get('independence'), 'independence')
    required = {'own_judgment', 'other_reviews_seen', 'method_identity_seen', 'prior_r64_exposure'}
    if set(independence) != required or any(type(independence[k]) is not bool for k in required):
        raise ValueError('Explicit four boolean independence observations required')
    seen = set()
    for item in _list(out.get('items'), 'items'):
        _mapping(item, 'review item')
        key = _text(item.get('blind_item_id'), 'blind_item_id')
        if key in seen:
            raise ValueError('Duplicate item review')
        seen.add(key)
        if schema == MUSIC_SCHEMA:
            if key not in music or _sha(item.get('media_sha256'), 'media_sha256') != music[key]['media_sha256'].lower():
                raise ValueError('Music review item/media identity mismatch')
            duration = float(music[key]['duration_seconds'])
            coverage = _span(item.get('coverage_seconds'), 'coverage_seconds', duration)
            item['coverage_seconds'] = coverage
            tracks = _mapping(item.get('tracks'), 'tracks')
            if set(tracks) != set(TRACKS):
                raise ValueError('Beat/downbeat/accent/phrase tracks must remain separate')
            for kind, track in tracks.items():
                _mapping(track, kind+' track')
                status = track.get('status')
                if status not in STATUSES:
                    raise ValueError('Explicit per-kind coverage status required')
                events = _list(track.get('events'), kind+' events')
                if status == 'NO_CLEAR_EVENTS' and events:
                    raise ValueError('NO_CLEAR_EVENTS may not contain positive events')
                if status == 'COMPLETE' and not events:
                    raise ValueError('Explicit NO_CLEAR_EVENTS required for reviewed empty track')
                previous = None
                for event in events:
                    _mapping(event, 'event')
                    time = _number(event.get('seconds'), 'event seconds')
                    if not coverage[0] <= time < coverage[1] or previous is not None and time <= previous:
                        raise ValueError('Events must be unique increasing seconds within reviewed coverage')
                    if event.get('confidence') not in CONFIDENCES:
                        raise ValueError('Explicit event confidence required')
                    event['seconds'] = time
                    previous = time
            for interval in _list(item.get('unsure_intervals'), 'unsure_intervals'):
                _mapping(interval, 'unsure interval')
                a, b = _span([interval.get('start'), interval.get('end')], 'unsure interval', duration)
                if a < coverage[0] or b > coverage[1]:
                    raise ValueError('Unsure interval leaves actual reviewed coverage')
                _text(interval.get('reason'), 'unsure reason')
                interval.update(start=a, end=b)
        else:
            if key not in cases:
                raise ValueError('Unknown quality blind_item_id')
            expected = forms[out['form_id']]['panel_orders'][key]
            if item.get('variant_order') != expected:
                raise ValueError('Displayed variant order differs from the assigned sealed form')
            ratings = _list(item.get('ratings'), 'ratings')
            for rating in ratings:
                _mapping(rating, 'rating')
            ids = _unique([r.get('blind_variant_id') for r in ratings], 'rated variant IDs')
            if len(ids) != 4 or set(ids) != set(expected):
                raise ValueError('Every actual variant must retain its rating or explicit abstention')
            for rating in ratings:
                scores = _mapping(rating.get('scores'), 'scores')
                if set(scores) != set(DIMENSIONS):
                    raise ValueError('Exactly the four frozen rating dimensions required')
                for score in scores.values():
                    _mapping(score, 'dimension rating')
                    if 'value' not in score or 'reason' not in score:
                        raise ValueError('Every rating requires value and reason fields')
                    value, reason = score['value'], score['reason']
                    if value is None:
                        _text(reason, 'unrated reason')
                    elif type(value) is not int or not 1 <= value <= 5 or reason is not None:
                        raise ValueError('Ordinal score must be integer1..5 with null reason, or null with reason')
    return out


def _partial(doc, cases, music):
    if doc['schema'] == QUALITY_SCHEMA:
        return set(i['blind_item_id'] for i in doc['items']) != set(cases) or any(
            score['value'] is None for item in doc['items'] for rating in item['ratings'] for score in rating['scores'].values())
    return set(i['blind_item_id'] for i in doc['items']) != set(music) or any(
        item['coverage_seconds'] != [0., float(music[item['blind_item_id']]['duration_seconds'])]
        or item['unsure_intervals'] or any(t['status'] not in ('COMPLETE', 'NO_CLEAR_EVENTS') for t in item['tracks'].values())
        for item in doc['items'])


def qualify(doc, admin, roster=None):
    """Qualify a contract against an explicitly supplied, checked private roster.

    Authentication/independence booleans in a review itself are self-reports.
    A roster's evidence SHA is a reference to an external check, not a check
    performed here. CONTRACT_QUALIFIED does not grant semantic truth or routing.
    """
    doc = validate_document(doc, admin)
    cases, music, _ = admin_index(admin)
    partial = bool(_partial(doc, cases, music))
    result = dict(review_id=doc['review_id'], reviewer_id=doc['reviewer_id'], schema=doc['schema'],
        status='PARTIAL_REVIEW' if partial else 'IMPORTED_UNVERIFIED', person_uid=None,
        reasons=['MISSING_ITEMS_COVERAGE_OR_UNSURE_RATINGS'] if partial else [], contract_qualified=False,
        coverage_complete=not partial,
        semantic_truth='UNKNOWN', operational_qualified=False,
        authentication_scope='SUPPLIED_PRIVATE_ROSTER_ASSERTION; NOT_EXTERNAL_AUTHENTICATION_BY_THIS_MODULE')
    ind = doc['independence']
    exposed = [k for k in ('other_reviews_seen', 'method_identity_seen', 'prior_r64_exposure') if ind[k]]
    if not ind['own_judgment']:
        exposed.append('own_judgment_false')
    if exposed:
        result.update(status='EXCLUDED_EXPOSED', reasons=exposed)
        return result
    if roster is None:
        result['reasons'].append('NO_INDEPENDENTLY_VERIFIED_PRIVATE_ROSTER')
        return result
    entries = _list(_mapping(roster, 'roster').get('entries'), 'roster.entries')
    matches = [r for r in entries if isinstance(r, dict) and r.get('reviewer_id') == doc['reviewer_id']]
    if len(matches) != 1:
        result['reasons'].append('MISSING_OR_AMBIGUOUS_ROSTER_IDENTITY')
        return result
    entry = matches[0]
    evidence = entry.get('authentication_evidence_sha256')
    if (entry.get('reviewer_roster_ref') != doc['reviewer_roster_ref']
            or entry.get('assigned_form_id') != doc['form_id']
            or entry.get('verified') is not True or entry.get('independence_verified') is not True
            or not isinstance(entry.get('person_uid'), str) or not entry['person_uid'].strip()
            or entry['person_uid'] != entry['person_uid'].strip()
            or not isinstance(evidence, str) or not _SHA.fullmatch(evidence)):
        result['reasons'].append('PRIVATE_ROSTER_AUTHENTICATION_OR_ASSIGNMENT_NOT_QUALIFIED')
        return result
    result.update(status='PARTIAL_REVIEW' if partial else 'CONTRACT_QUALIFIED',
                  person_uid=entry['person_uid'], contract_qualified=True)
    return result
