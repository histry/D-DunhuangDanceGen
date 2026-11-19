"""Deterministic review diagnostics, with missing evidence kept unevaluable.

Event matching is one-to-one, maximum cardinality then minimum absolute timing
error. Human agreement is not accuracy, and proxy correlation is not a quality
or causal qualification. No training, threshold fitting or population selection.
"""
from itertools import combinations
from datetime import datetime
import math

import numpy as np

from .records import (METHODS, DIMENSIONS, TRACKS, PROXIES, MUSIC_SCHEMA,
                      QUALITY_SCHEMA, admin_index, validate_document, qualify)

DEFAULT_TOLERANCES = dict(beat=.07, downbeat=.10, accent=.10, phrase=.50)
_EPS = 1e-12


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
        raise ValueError(name+' must be a finite numeric value, not bool')
    return float(value)


def _span(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(name+' must contain start/end seconds')
    a, b = [_real(x, name) for x in value]
    if a < 0 or b <= a:
        raise ValueError(name+' must be positive half-open support')
    return a, b


def _events(values, name):
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(name+' must be a sequence of seconds/events')
    rows = []
    for index, event in enumerate(values):
        value = event.get('seconds') if isinstance(event, dict) else event
        value = _real(value, name)
        if value < 0:
            raise ValueError('Negative event timestamp')
        rows.append((value, index))
    return sorted(rows)


def event_match(ref, est, tolerance, interval, excluded=None):
    """Match points within known coverage; excluded intervals remain unscored.

    Inputs may contain duplicate times: duplicates remain separate detections,
    so they cannot inflate true positives. Input points outside the evaluation
    interval are explicitly counted but not scored. No pair crosses an excluded
    interval. Deterministic ordered matching attains the one-dimensional optimum.
    """
    tolerance = _real(tolerance, 'tolerance')
    if tolerance <= 0:
        raise ValueError('Strictly positive registered tolerance required')
    lo, hi = _span(interval, 'interval')
    spans = []
    for value in ([] if excluded is None else excluded):
        a, b = _span(value, 'excluded interval')
        if a < lo or b > hi:
            raise ValueError('Excluded interval leaves evaluation coverage')
        spans.append((a, b))
    merged = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    components, start = [], lo
    for a, b in merged:
        if start < a:
            components.append((start, a))
        start = b
    if start < hi:
        components.append((start, hi))

    def eligible(values, name):
        raw = _events(values, name)
        active, outside, unknown = [], 0, 0
        for time, index in raw:
            if not lo <= time < hi:
                outside += 1
                continue
            component = next((k for k, (a, b) in enumerate(components) if a <= time < b), None)
            if component is None:
                unknown += 1
            else:
                active.append((time, index, component))
        return raw, active, outside, unknown

    raw_ref, R, outside_ref, unknown_ref = eligible(ref, 'reference events')
    raw_est, E, outside_est, unknown_est = eligible(est, 'estimated events')
    n, m = len(R), len(E)
    count = np.zeros((n+1, m+1), dtype=np.int32)
    error = np.zeros((n+1, m+1), dtype=np.float64)
    back = np.zeros((n+1, m+1), dtype=np.uint8)
    back[1:, 0], back[0, 1:] = 1, 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [(int(count[i-1, j]), float(error[i-1, j]), 1),
                       (int(count[i, j-1]), float(error[i, j-1]), 2)]
            distance = abs(R[i-1][0]-E[j-1][0])
            if R[i-1][2] == E[j-1][2] and distance <= tolerance+_EPS:
                options.append((int(count[i-1, j-1])+1,
                                float(error[i-1, j-1])+distance, 3))
            # Prefer matches on an exact numerical tie, then stable skip order.
            best = min(options, key=lambda q: (-q[0], q[1], 0 if q[2] == 3 else q[2]))
            count[i, j], error[i, j], back[i, j] = best
    matched, i, j = [], n, m
    while i or j:
        step = back[i, j]
        if step == 3:
            a, b = R[i-1], E[j-1]
            matched.append(dict(reference_index=a[1], estimate_index=b[1],
                reference_seconds=a[0], estimate_seconds=b[0],
                signed_error_seconds=b[0]-a[0], absolute_error_seconds=abs(b[0]-a[0])))
            i, j = i-1, j-1
        elif step == 1:
            i -= 1
        elif step == 2:
            j -= 1
        else:
            raise AssertionError('Invalid deterministic matching traceback')
    matched.reverse()
    tp = len(matched)
    precision = tp/m if m else None
    recall = tp/n if n else None
    f1 = 2*tp/(n+m) if n+m else None
    return dict(status='EVALUABLE' if n+m else 'NOT_EVALUABLE_NO_EVENTS',
        TP=tp, FP=m-tp, FN=n-tp, precision=precision, recall=recall, F1=f1,
        matched_MAE_seconds=float(np.mean([x['absolute_error_seconds'] for x in matched])) if matched else None,
        matched_bias_seconds=float(np.mean([x['signed_error_seconds'] for x in matched])) if matched else None,
        matches=matched, tolerance_seconds=tolerance, interval_seconds=[lo, hi],
        excluded_seconds=[list(x) for x in merged], scored_components=[list(x) for x in components],
        reference_events=n, estimate_events=m, reference_input_events=len(raw_ref), estimate_input_events=len(raw_est),
        reference_outside_interval=outside_ref, estimate_outside_interval=outside_est,
        reference_in_unknown=unknown_ref, estimate_in_unknown=unknown_est,
        interpretation='ONE_TO_ONE_SAME_KIND_POINT_AGREEMENT_OR_REFERENCE_COMPARISON; NOT_SEMANTIC_TRUTH')


def tied_ranks(values):
    """Ascending average ranks, preserving exact ties."""
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError('A sequence of rank values required')
    x = np.asarray([_real(v, 'rank value') for v in values], dtype=float)
    if x.ndim != 1:
        raise ValueError('One-dimensional rank values required')
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i+1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i+1+j)/2
        i = j
    return ranks


def spearman_tied(x, y):
    """Pearson correlation of average ranks; constant/short inputs are UNKNOWN."""
    if not isinstance(x, (list, tuple, np.ndarray)) or not isinstance(y, (list, tuple, np.ndarray)):
        raise ValueError('Paired rank sequences required')
    if len(x) != len(y):
        raise ValueError('Paired rank values require equal lengths')
    a, b = tied_ranks(x), tied_ranks(y)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    a, b = a-a.mean(), b-b.mean()
    return float(np.clip(a@b/np.sqrt((a@a)*(b@b)), -1., 1.))


def _groups(scores):
    rated = [(name, value) for name, value in scores.items() if value is not None]
    return [[name for name, value in rated if value == level]
            for level in sorted({value for _, value in rated}, reverse=True)]


def _quality_rows(cases, documents, qualifications):
    raw, correlations, differences = [], [], []
    rated_people = {}
    case_people = {}
    for d in documents:
        q = qualifications[d['review_id']]
        if d['schema'] != QUALITY_SCHEMA or not q['contract_qualified'] or q.get('duplicate_person_document', False):
            continue
        for item in d['items']:
            cid = item['blind_item_id']
            case_people.setdefault(cid, set()).add(q['person_uid'])
            variants = {v['blind_variant_id']: v['method'] for v in cases[cid]['variants']}
            for rating in item['ratings']:
                for dimension, score in rating['scores'].items():
                    if score['value'] is not None:
                        rated_people.setdefault((cid, dimension, variants[rating['blind_variant_id']]), set()).add(q['person_uid'])
    people = {qualifications[d['review_id']]['person_uid'] for d in documents
              if d['schema'] == QUALITY_SCHEMA and qualifications[d['review_id']]['contract_qualified']
              and not qualifications[d['review_id']].get('duplicate_person_document', False)}
    for doc in documents:
        if doc['schema'] != QUALITY_SCHEMA:
            continue
        qualification = qualifications[doc['review_id']]
        qualified = qualification['contract_qualified'] and not qualification.get('duplicate_person_document', False)
        for item in doc['items']:
            case = cases[item['blind_item_id']]
            variants = {v['blind_variant_id']: v for v in case['variants']}
            methods = {variants[r['blind_variant_id']]['method']: r['scores'] for r in item['ratings']}
            common = dict(blind_item_id=item['blind_item_id'], case_key=case['case_key'], role=case['role'],
                music_id=case['music_id'], review_id=doc['review_id'], reviewer_id=doc['reviewer_id'],
                person_uid=qualification['person_uid'], qualified=qualified,
                qualification_status=qualification['status'],
                coverage_complete=qualification['coverage_complete'],
                independent_people_for_task=len(people), independent_people_for_case=len(case_people.get(item['blind_item_id'],set())))
            duplicate = {}
            for v in variants.values():
                duplicate.setdefault(v['trajectory_sha256'], []).append(v['method'])
            for dimension in DIMENSIONS:
                counts = {name:len(rated_people.get((item['blind_item_id'],dimension,name),set())) for name in METHODS}
                dim_common = dict(common, independent_rated_people_by_method=counts,
                                  two_person_requirement_met=all(v>=2 for v in counts.values()))
                scores = {name: methods[name][dimension]['value'] for name in METHODS}
                reasons = {name: methods[name][dimension]['reason'] for name in METHODS if scores[name] is None}
                raw.append(dict(dim_common, dimension=dimension, scores=scores, unsure_reasons=reasons,
                    rated_methods=sum(v is not None for v in scores.values()),
                    unrated_methods=sum(v is None for v in scores.values()),
                    rank_groups_descending=_groups(scores),
                    duplicate_trajectory_groups=[v for v in duplicate.values() if len(v) > 1],
                    independent_method_samples=False))
                for name in METHODS:
                    if name == 'Fixed48':
                        continue
                    available = scores[name] is not None and scores['Fixed48'] is not None
                    differences.append(dict(dim_common, dimension=dimension, method=name, baseline='Fixed48',
                        score=scores[name], baseline_score=scores['Fixed48'],
                        difference=(scores[name]-scores['Fixed48']) if available and qualified else None,
                        status='EVALUABLE' if available and qualified else 'NOT_EVALUABLE',
                        missing_reasons={n: methods[n][dimension]['reason'] for n in (name, 'Fixed48') if scores[n] is None},
                        scale='ORDINAL_SCORE_DIFFERENCE_DESCRIPTIVE_NOT_INTERVAL_SCALE_PROOF'))
                for proxy in PROXIES:
                    values = {v['method']: v['proxies'][proxy] for v in variants.values()}
                    names = [name for name in METHODS if scores[name] is not None and values[name] is not None]
                    sufficient = len(names) >= 3
                    rho = spearman_tied([values[n] for n in names], [scores[n] for n in names]) if qualified and sufficient else None
                    correlations.append(dict(dim_common, dimension=dimension, proxy=proxy,
                        methods=names, valid_method_count=len(names), spearman=rho,
                        status='EVALUABLE' if rho is not None else 'NOT_EVALUABLE',
                        reason=None if rho is not None else ('CONSTANT_RANKS' if qualified and sufficient else 'UNQUALIFIED_OR_FEWER_THAN_THREE_VALID_METHODS'),
                        proxy_ranks=tied_ranks([values[n] for n in names]).tolist() if sufficient else None,
                        rating_ranks=tied_ranks([scores[n] for n in names]).tolist() if sufficient else None,
                        interpretation='WITHIN_CASE_ASSOCIATION_ON_FOUR_FROZEN_SELECTED_OUTPUTS; NO_CAUSAL_OR_POPULATION_QUALITY_CLAIM'))
    return raw, correlations, differences


def summarize_reviews(admin, documents, roster=None, tolerances=None):
    """Analyze supplied records without generating people or missing labels.

    The caller must separately enforce packet sealing / audio-before-motion
    ordering and verify private authentication evidence files. This module only
    validates records and uses the explicitly supplied private roster assertion.
    """
    cases, music, _ = admin_index(admin)
    if not isinstance(documents, list):
        raise ValueError('documents must be an explicit list')
    docs = [validate_document(d, admin) for d in documents]
    review_ids = [d['review_id'] for d in docs]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError('Duplicate review_id across imported documents')
    unique = [(d['schema'], d['reviewer_id']) for d in docs]
    if len(unique) != len(set(unique)):
        raise ValueError('Duplicate primary reviewer document for the same task')
    limits = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        if not isinstance(tolerances, dict) or set(tolerances) != set(TRACKS):
            raise ValueError('Explicit tolerance for every registered music kind required')
        limits = {k: _real(v, 'tolerance') for k, v in tolerances.items()}
        if min(limits.values()) <= 0:
            raise ValueError('Positive registered event tolerances required')
    qualifications = {d['review_id']: qualify(d, admin, roster) for d in docs}
    # An alias ID cannot manufacture a second independent person. First by
    # timestamp/review_id is retained deterministically, never by review scores.
    persons = {}
    for d in sorted(docs, key=lambda x: (datetime.fromisoformat(x['reviewed_at']).timestamp(), x['review_id'])):
        q = qualifications[d['review_id']]
        if not q['contract_qualified']:
            continue
        key = (d['schema'], q['person_uid'])
        if key in persons:
            q.update(duplicate_person_document=True, counted_as_independent=False,
                     duplicate_of_review_id=persons[key])
        else:
            persons[key] = d['review_id']
            q.update(duplicate_person_document=False, counted_as_independent=True)
    active = [d for d in docs if qualifications[d['review_id']]['contract_qualified']
              and not qualifications[d['review_id']].get('duplicate_person_document', False)]
    music_docs = [d for d in active if d['schema'] == MUSIC_SCHEMA]
    music_coverage = [dict(review_id=d['review_id'], reviewer_id=d['reviewer_id'],
        blind_item_id=item['blind_item_id'], coverage_seconds=item['coverage_seconds'], kind=kind,
        track_status=track['status'], recorded_event_count=len(track['events']),
        unsure_interval_count=len(item['unsure_intervals']),
        contract_qualified=qualifications[d['review_id']]['contract_qualified'],
        person_uid=qualifications[d['review_id']]['person_uid'],
        counted_as_independent=qualifications[d['review_id']].get('counted_as_independent',False))
        for d in docs if d['schema']==MUSIC_SCHEMA for item in d['items'] for kind,track in item['tracks'].items()]
    event_agreement, music_not_evaluable = [], []
    for first, second in combinations(music_docs, 2):
        f = {r['blind_item_id']: r for r in first['items']}
        s = {r['blind_item_id']: r for r in second['items']}
        for item in sorted(set(f) & set(s)):
            for kind in TRACKS:
                a, b = f[item], s[item]
                statuses = [d['tracks'][kind]['status'] for d in (a, b)]
                if any(v not in ('COMPLETE', 'NO_CLEAR_EVENTS') for v in statuses):
                    music_not_evaluable.append(dict(blind_item_id=item,kind=kind,
                        review_ids=[first['review_id'],second['review_id']],
                        status='NOT_EVALUABLE',reason='KIND_COVERAGE_UNSURE_OR_PARTIAL',track_statuses=statuses))
                    continue
                lo = max(a['coverage_seconds'][0], b['coverage_seconds'][0])
                hi = min(a['coverage_seconds'][1], b['coverage_seconds'][1])
                if hi <= lo:
                    continue
                excluded = [[max(lo,r['start']), min(hi,r['end'])]
                            for r in a['unsure_intervals']+b['unsure_intervals']
                            if max(lo,r['start']) < min(hi,r['end'])]
                stats = event_match(a['tracks'][kind]['events'], b['tracks'][kind]['events'], limits[kind], [lo, hi], excluded)
                event_agreement.append(dict(stats, blind_item_id=item, raw_audio_sha256=music[item]['raw_audio_sha256'],
                    kind=kind, review_ids=[first['review_id'], second['review_id']],
                    person_uids=[qualifications[d['review_id']]['person_uid'] for d in (first, second)],
                    interpretation='INTER_REVIEWER_AGREEMENT_NOT_ACCURACY_OR_ADJUDICATED_TRUTH'))
    raw, correlations, differences = _quality_rows(cases, docs, qualifications)
    quality_docs = [d for d in active if d['schema'] == QUALITY_SCHEMA]
    rating_agreement = []
    for first, second in combinations(quality_docs, 2):
        f = {r['blind_item_id']: r for r in first['items']}
        s = {r['blind_item_id']: r for r in second['items']}
        for item in sorted(set(f) & set(s)):
            case = cases[item]
            vmap = {v['blind_variant_id']: v['method'] for v in case['variants']}
            sides = [{vmap[r['blind_variant_id']]: r['scores'] for r in d[item]['ratings']} for d in (f, s)]
            for dimension in DIMENSIONS:
                common_methods = [name for name in METHODS if all(d[name][dimension]['value'] is not None for d in sides)]
                a, b = [[d[name][dimension]['value'] for name in common_methods] for d in sides]
                rating_agreement.append(dict(blind_item_id=item, case_key=case['case_key'], music_id=case['music_id'],
                    dimension=dimension, review_ids=[first['review_id'], second['review_id']],
                    person_uids=[qualifications[d['review_id']]['person_uid'] for d in (first, second)],
                    rated_method_count=len(common_methods), common_methods=common_methods,
                    exact_agreement_count=sum(x == y for x, y in zip(a, b)),
                    mean_absolute_ordinal_difference=float(np.mean(np.abs(np.asarray(a)-np.asarray(b)))) if a else None,
                    spearman=spearman_tied(a, b), scale='ORDINAL_AGREEMENT_DESCRIPTIVE_NOT_ACCURACY'))
    qualified_people = sorted({qualifications[d['review_id']]['person_uid'] for d in active})
    music_people = {qualifications[d['review_id']]['person_uid'] for d in music_docs}
    quality_people = {qualifications[d['review_id']]['person_uid'] for d in quality_docs}
    source_groups = sorted({uid for c in cases.values() for uid in c.get('source_uids', [])})
    return dict(schema='R65_REVIEW_DIAGNOSTICS_V1',
        status='NOT_EVALUABLE' if not active else ('TWO_PERSON_RECORDS_PRESENT' if max(len(music_people),len(quality_people)) >= 2 else 'DESCRIPTIVE_SINGLE_PERSON_PER_TASK'),
        documents_registered=len(docs), qualified_documents=len(active), independent_person_count=len(qualified_people),
        qualifications=list(qualifications.values()), tolerances_seconds=limits,
        music_coverage=music_coverage,
        music_independent_person_count=len(music_people), quality_independent_person_count=len(quality_people),
        two_person_music_requirement_met=len(music_people)>=2, two_person_quality_requirement_met=len(quality_people)>=2,
        two_person_requirement_scope='TASK_ROSTER_COUNT_ONLY; PER_CASE_DIMENSION_VALID_PERSON_COUNTS_REPORTED_SEPARATELY',
        music_agreement_status='EVALUABLE' if any(r['F1'] is not None for r in event_agreement) else 'NOT_EVALUABLE',
        music_agreement=event_agreement, music_not_evaluable=music_not_evaluable,
        quality_agreement_status='EVALUABLE' if any(r['rated_method_count'] for r in rating_agreement) else 'NOT_EVALUABLE', quality_agreement=rating_agreement,
        quality_reviews=raw, proxy_correlations=correlations, fixed48_differences=differences,
        music_groups=len({c['music_id'] for c in cases.values()}),
        source_groups=source_groups or None, source_group_status='AVAILABLE' if source_groups else 'NOT_IN_FIXED_ADMIN_SCHEMA',
        confidence_intervals=None, independent_edge_bootstrap_performed=False,
        population_scope='FIXED_HISTORICALLY_EXPOSED_MUSIC_AND_SHARED_SOURCE_CASES; FOUR_METHODS_AND_REVIEWERS_NOT_INDEPENDENT_CASES',
        human_semantic_truth='UNKNOWN_NOT_GRANTED_BY_CONTRACT_OR_CORRELATION', operational_qualified=False,
        training_steps=0, planner_runs=0, can_route=False)
