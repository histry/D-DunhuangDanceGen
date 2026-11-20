"""Frozen R66 source-clock diagnostics; no fitting or operational promotion.

Positive experimental delta reads later audio: audio_frame = motion_frame +
calibrated_offset + delta. The target in this direct comparison is delta=0.
It is not R59's separate injected-motion-shift recovery target (-injection).
"""
from collections import Counter, defaultdict
import math
import re

import numpy as np

OFFSETS = (-15, -8, 0, 8, 15)
TRUTH = 0
SCORE_TOLERANCE = 1e-12
SEED = 66066
DRAWS = 3000
MINIMUM_MUSIC_GROUPS = 8
METHODS = ('Beat', 'Accent', 'Primary', 'Energy', 'Density')
METHOD_SCOPE = dict(Beat='FROZEN_MOTION_SPEED_PEAK_TO_AUDIO_BEAT_SOURCE_PROXY',
    Accent='FROZEN_MOTION_SPEED_PEAK_TO_ACOUSTIC_ACCENT_SOURCE_PROXY',
    Primary='FROZEN_HALF_BEAT_HALF_ACCENT_SOURCE_PROXY',
    Energy='CONTINUOUS_AUDIO_MOTION_ENERGY_BASELINE_NOT_EVENT_TRUTH',
    Density='AUDIO_EVENT_DENSITY_BASELINE_NOT_EVENT_CORRESPONDENCE')
FIELDS = ('top1_accuracy', 'mae_frames', 'pairwise_win_rate',
          'pairwise_strict_win_rate', 'pairwise_tie_rate', 'mean_correct_minus_shift')
PAIRED_FIELDS = tuple(f for f in FIELDS if f != 'mean_correct_minus_shift')


def _text(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError('Nonempty unpadded '+name+' required')
    return value


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int,float,np.integer,np.floating)) or not math.isfinite(value):
        raise ValueError('Finite numeric '+name+' required; bool is not numeric evidence')
    return float(value)


def _score(value):
    return None if value is None else _real(value, 'score')


def _vector(values, length=None):
    if not isinstance(values, (list,tuple,np.ndarray)) or np.ndim(values) != 1:
        raise ValueError('One-dimensional score list required')
    if length is not None and len(values) != length:
        raise ValueError('Exactly five registered temporal scores required')
    return [_score(x) for x in values]


def _win(correct, other, tolerance):
    delta = correct-other
    return .5 if abs(delta) <= tolerance else (1. if delta > 0 else 0.)


def pair_metrics(scores, offsets=OFFSETS, truth=TRUTH, tolerance=SCORE_TOLERANCE):
    """Exact tie expectation, with whole-five-condition and partial coverage split."""
    if (any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in offsets)
            or tuple(offsets) != OFFSETS or type(truth) is not int or truth != 0):
        raise ValueError('R66 direct audio-shift grid and zero target are frozen')
    tolerance = _real(tolerance, 'score tolerance')
    if tolerance != SCORE_TOLERANCE:
        raise ValueError('Frozen 1e-12 tie tolerance required')
    values = _vector(scores, 5)
    correct = values[2]
    complete = all(x is not None for x in values)
    pairs = []
    for offset, value in zip(offsets, values):
        if offset == truth:
            continue
        available = correct is not None and value is not None
        win = _win(correct,value,tolerance) if available else None
        pairs.append(dict(audio_shift_frames=offset, available=available,
            correct_score=correct, shifted_score=value,
            correct_minus_shift=correct-value if available else None,
            win=win, tied=(win == .5) if available else None,
            label_scope='NONZERO_CLOCK_CONTROL_NOT_ARTISTICALLY_WRONG_DANCE'))
    best, accuracy, mae = [], None, None
    if complete:
        maximum = max(values)
        best = [offset for offset,value in zip(offsets,values) if abs(value-maximum) <= tolerance]
        accuracy = (1./len(best)) if truth in best else 0.
        mae = float(np.mean([abs(offset-truth) for offset in best]))
    available_pairs = [x for x in pairs if x['available']]
    return dict(status='EVALUABLE_COMPLETE_FIVE_CONDITIONS' if complete else 'NOT_EVALUABLE_INCOMPLETE_FIVE_CONDITIONS',
        complete=complete, finite_conditions=sum(x is not None for x in values),
        missing_offsets=[o for o,v in zip(offsets,values) if v is None], correct_score=correct,
        top1_accuracy=accuracy, mae_frames=mae, best_offsets=best, tie_count=len(best) if complete else None,
        pairwise_win_rate=float(np.mean([x['win'] for x in pairs])) if complete else None,
        pairwise_strict_win_rate=float(np.mean([x['win']==1 for x in pairs])) if complete else None,
        pairwise_tie_rate=float(np.mean([x['tied'] for x in pairs])) if complete else None,
        mean_correct_minus_shift=float(np.mean([x['correct_minus_shift'] for x in pairs])) if complete else None,
        pairwise=pairs, available_pair_count=len(available_pairs),
        available_pairs_win_rate=float(np.mean([x['win'] for x in available_pairs])) if available_pairs else None,
        primary_population='SAME_QUERY_ALL_FIVE_FINITE; PARTIAL_PAIRS_DIAGNOSTIC_ONLY',
        truth_audio_shift_frames=truth, score_tolerance=tolerance)


clock_metrics = pair_metrics


def wrong_song_metrics(correct, wrong_scores, tolerance=SCORE_TOLERANCE):
    """Association control only: no semantic-negative labels or donor-IID CI."""
    if _real(tolerance,'score tolerance') != SCORE_TOLERANCE:
        raise ValueError('Frozen 1e-12 tie tolerance required')
    correct = _score(correct)
    values = _vector(wrong_scores)
    available = [value for value in values if correct is not None and value is not None]
    return dict(status='DIAGNOSTIC_ONLY' if available else 'NOT_EVALUABLE',
        registered_donors=len(values), finite_donors=sum(x is not None for x in values),
        evaluable_donors=len(available),
        correct_vs_wrong_win_rate=float(np.mean([_win(correct,x,tolerance) for x in available])) if available else None,
        correct_minus_wrong_mean=float(np.mean([correct-x for x in available])) if available else None,
        tie_rate=float(np.mean([abs(correct-x)<=tolerance for x in available])) if available else None,
        scope='OTHER_RECORDED_AUDIO_ASSOCIATION_NOT_ARTISTIC_OR_SEMANTIC_TRUTH',
        independent_donor_bootstrap_performed=False)


def _source_unit(row):
    value = row.get('raw_motion_sha256')
    if value is not None:
        if not isinstance(value,str) or not re.fullmatch(r'[a-fA-F0-9]{64}',value):
            raise ValueError('Valid raw motion SHA256 required when supplied')
        return 'raw_motion:'+value.lower()
    value = row.get('recording_uid')
    if value is not None:
        return 'recording:'+_text(value,'recording_uid')
    return 'source_uid:'+row['source_uid']


def _motion_status(row):
    count = row.get('motion_event_count')
    if count is not None and (isinstance(count,(bool,np.bool_)) or not isinstance(count,(int,np.integer)) or count < 0):
        raise ValueError('motion_event_count must be a nonnegative integer or absent')
    flag = row.get('no_motion_events')
    if flag is not None and type(flag) is not bool:
        raise ValueError('no_motion_events must be an explicit bool')
    state = row.get('evidence_status',row.get('status'))
    explicit = True if state == 'NO_MOTION_EVENTS' else flag
    if count is not None:
        if explicit is not None and explicit != (count == 0):
            raise ValueError('Contradictory motion-event coverage observations')
        return count == 0
    return explicit


def population_audit(queries):
    """Register duplicate evidence; aliases never become separate recordings."""
    if not isinstance(queries,list):
        raise ValueError('All registered query rows must be passed as a list')
    ids, source_context, source_names, units, duplicate_windows = set(),{},{},defaultdict(list),defaultdict(list)
    audio_groups=defaultdict(set)
    groups = defaultdict(list)
    for row in queries:
        if not isinstance(row,dict):
            raise ValueError('Query row must be an object')
        for key in ('uid','group','source_uid','dataset','role'):
            _text(row.get(key),key)
        if row['uid'] in ids:
            raise ValueError('Duplicate registered query UID')
        ids.add(row['uid'])
        _motion_status(row)
        for name in ('scores','wrong_scores'):
            if name in row and not isinstance(row[name],dict):
                raise ValueError(name+' must be a method mapping')
            if set(row.get(name,{}))-set(METHODS):
                raise ValueError('Only frozen Beat/Accent/Primary/Energy/Density methods may be supplied')
        context=(row['dataset'],row['role'],row['group'])
        unit=_source_unit(row)
        name=(row['dataset'],row['source_uid'])
        if name in source_names and source_names[name] != (unit,context):
            raise ValueError('One source UID changes its recording identity or group/role')
        source_names[name]=(unit,context)
        if unit in source_context and source_context[unit] != context:
            raise ValueError('One source/duplicate cluster crosses registered role or music-group strata')
        source_context[unit]=context
        units[unit].append(row)
        groups[context].append(row)
        audio=row.get('audio_sha256')
        if audio is not None:
            if not isinstance(audio,str) or not re.fullmatch(r'[a-fA-F0-9]{64}',audio):
                raise ValueError('Valid raw audio SHA256 required when supplied')
            audio_groups[(row['dataset'],row['role'],audio.lower())].add(row['group'])
        interval=row.get('source_interval')
        if interval is not None:
            if not isinstance(interval,(list,tuple)) or len(interval)!=2 or any(type(x) is not int for x in interval) or not 0<=interval[0]<interval[1]:
                raise ValueError('Actual source_interval must be an integer half-open pair')
            duplicate_windows[(unit,tuple(interval))].append(row['uid'])
    if any(len(v)>1 for v in audio_groups.values()):
        raise ValueError('Identical raw audio is split across music bootstrap groups')
    source_clusters=[]
    for unit,rows in sorted(units.items()):
        names=sorted({r['source_uid'] for r in rows})
        source_clusters.append(dict(source_balance_unit=unit,source_uids=names,
            source_uid_count=len(names),registered_queries=len(rows),query_uids=[r['uid'] for r in rows],
            dataset=rows[0]['dataset'],role=rows[0]['role'],group=rows[0]['group'],
            duplicate_source_aliases=len(names)>1))
    group_evidence=[]
    for (dataset,role,group),rows in sorted(groups.items()):
        music_ids=sorted({str(r.get('music_id',r.get('audio_sha256'))) for r in rows
                          if r.get('music_id',r.get('audio_sha256')) is not None})
        audio_hashes=sorted({r['audio_sha256'] for r in rows if r.get('audio_sha256') is not None})
        group_evidence.append(dict(dataset=dataset,role=role,group=group,
            registered_queries=len(rows),source_uids=sorted({r['source_uid'] for r in rows}),
            source_uid_count=len({r['source_uid'] for r in rows}),source_balance_unit_count=len({_source_unit(r) for r in rows}),
            music_ids=music_ids,music_id_count=len(music_ids),audio_sha256=audio_hashes,
            audio_sha256_count=len(audio_hashes),music_identity_missing_queries=sum(r.get('music_id',r.get('audio_sha256')) is None for r in rows)))
    return dict(registered_queries=len(queries),registered_source_uids=len({r['source_uid'] for r in queries}),
        recording_balance_units=len(units),music_group_strata=len(groups),
        source_duplicate_clusters=source_clusters,music_group_evidence=group_evidence,
        raw_audio_group_clusters=[dict(dataset=k[0],role=k[1],audio_sha256=k[2],music_groups=sorted(v))
                                 for k,v in sorted(audio_groups.items())],
        metadata_duplicate_window_clusters=[dict(source_balance_unit=k[0],source_interval=list(k[1]),query_uids=v)
            for k,v in sorted(duplicate_windows.items()) if len(v)>1],
        duplicate_query_rows_removed=False,
        unit_definition='RAW_MOTION_SHA_IF_PRESENT_ELSE_RECORDING_UID_ELSE_SOURCE_UID; THEN_REGISTERED_MUSIC_GROUP',
        independence='REGISTERED_GROUPS_NOT_REPROVEN_AS_UNSEEN_MUSIC_OR_INDEPENDENT_CHOREOGRAPHIES')


def balanced_summary(rows, field, seed=SEED, draws=DRAWS, bootstrap=True):
    """Mean windows within recording, recordings within group, then equal groups."""
    if isinstance(seed,bool) or not isinstance(seed,(int,np.integer)) or isinstance(draws,bool) or not isinstance(draws,(int,np.integer)) or draws<1:
        raise ValueError('Integer bootstrap seed and positive draws required')
    cells=defaultdict(lambda:defaultdict(list))
    for row in rows:
        value=row.get(field)
        if value is not None:
            cells[row['group']][row['source_unit']].append(_real(value,field))
    groups={g:dict(source_means={u:float(np.mean(v)) for u,v in sorted(sources.items())},
        source_count=len(sources),window_count=sum(len(v) for v in sources.values()),
        mean=float(np.mean([np.mean(v) for v in sources.values()]))) for g,sources in sorted(cells.items())}
    values=np.array([row['mean'] for row in groups.values()],float)
    ci=None
    if bootstrap and len(values)>=2:
        rng=np.random.default_rng(seed)
        samples=values[rng.integers(len(values),size=(int(draws),len(values)))].mean(1)
        ci=np.quantile(samples,[.025,.975]).tolist()
    return dict(evaluable_queries=sum(r['window_count'] for r in groups.values()),
        sources=sum(r['source_count'] for r in groups.values()),groups=len(groups),group_means=groups,
        mean=float(values.mean()) if len(values) else None,ci95=ci,
        bootstrap_draws=int(draws) if bootstrap and len(values)>=2 else 0,bootstrap_seed=int(seed),
        below_eight_registered_music_groups=len(groups)<MINIMUM_MUSIC_GROUPS,
        unit='MUSIC_GROUP_MEAN_AFTER_RECORDING_AND_WINDOW_BALANCE',
        interval_scope='EXPOSED_REGISTERED_GROUP_CLUSTER_DIAGNOSTIC_NOT_NEW_MUSIC_GENERALIZATION' if ci is not None else
                       ('DESCRIPTIVE_ONLY_SHARED_WRONG_SONG_DONORS' if not bootstrap else 'NO_CI_FEWER_THAN_TWO_EVALUABLE_MUSIC_GROUPS'))


def _meta(row):
    return {k:row[k] for k in ('uid','source_uid','group','dataset','role')} | {'source_unit':_source_unit(row)}


def _coverage(rows,method,metrics):
    return dict(registered_queries=len(rows),complete_queries=sum(q['complete'] for q in metrics),
        partial_queries=sum(0<q['finite_conditions']<5 for q in metrics),
        no_score_queries=sum(q['finite_conditions']==0 for q in metrics),
        finite_correct_queries=sum(q['correct_score'] is not None for q in metrics),
        missing_method_queries=sum(method not in r.get('scores',{}) for r in rows),
        no_motion_events_explicit=sum(_motion_status(r) is True for r in rows),
        motion_event_status_unreported=sum(_motion_status(r) is None for r in rows),
        all_null_not_assumed_no_motion=True,
        registered_groups=len({r['group'] for r in rows}),registered_recording_units=len({_source_unit(r) for r in rows}),
        missing_offsets={str(o):sum(o in q['missing_offsets'] for q in metrics) for o in OFFSETS},
        evidence_status_counts=dict(Counter(str(r.get('evidence_status',r.get('status','UNREPORTED'))) for r in rows)))


def summarize(queries, seed=SEED, draws=DRAWS):
    """Return all strata and paired method effects; never choose a winning rule."""
    audit=population_audit(queries)
    methods=list(METHODS)
    parsed={};query_results=[]
    for row in queries:
        by={};wrong={}
        for method in methods:
            by[method]=pair_metrics(row.get('scores',{}).get(method,[None]*5))
            wrong[method]=wrong_song_metrics(by[method]['correct_score'],row.get('wrong_scores',{}).get(method,[]))
        parsed[row['uid']]=(by,wrong)
        query_results.append(dict(_meta(row),motion_event_count=row.get('motion_event_count'),
            no_motion_events=_motion_status(row),metrics=by,wrong_song=wrong))
    strata={}
    for dataset,role in sorted({(r['dataset'],r['role']) for r in queries}):
        rows=[r for r in queries if (r['dataset'],r['role'])==(dataset,role)]
        method_summaries={}
        for method in methods:
            qs=[parsed[r['uid']][0][method] for r in rows]
            flat=[dict(_meta(r),**{f:q[f] for f in FIELDS}) for r,q in zip(rows,qs)]
            main={f:balanced_summary(flat,f,seed,draws) for f in FIELDS}
            by_offset={}
            for index,offset in enumerate(o for o in OFFSETS if o!=0):
                complete,available=[],[]
                for r,q in zip(rows,qs):
                    pair=q['pairwise'][index]
                    item=dict(_meta(r),win=pair['win'],correct_minus_shift=pair['correct_minus_shift'])
                    if pair['available']:available.append(item)
                    if q['complete']:complete.append(item)
                by_offset[str(offset)]=dict(registered_queries=len(rows),complete_population_queries=len(complete),
                    available_pair_queries=len(available),complete_population={f:balanced_summary(complete,f,seed,draws) for f in ('win','correct_minus_shift')},
                    available_pairs_diagnostic={f:balanced_summary(available,f,seed,draws) for f in ('win','correct_minus_shift')})
            wrong_rows=[]
            for row in rows:
                q=parsed[row['uid']][1][method]
                wrong_rows.append(dict(_meta(row),**{f:q[f] for f in ('correct_vs_wrong_win_rate','correct_minus_wrong_mean','tie_rate')}))
            method_summaries[method]=dict(method_scope=METHOD_SCOPE[method],coverage=_coverage(rows,method,qs),metrics=main,by_offset=by_offset,
                wrong_song=dict(metrics={f:balanced_summary(wrong_rows,f,seed,draws,bootstrap=False)
                    for f in ('correct_vs_wrong_win_rate','correct_minus_wrong_mean','tie_rate')},
                    registered_queries=len(rows),registered_donors=sum(parsed[r['uid']][1][method]['registered_donors'] for r in rows),
                    evaluable_donors=sum(parsed[r['uid']][1][method]['evaluable_donors'] for r in rows),
                    diagnostic_only=True,independent_donor_bootstrap_performed=False))
        comparisons={}
        for first,second in [('Primary','Energy')]:
            shared=[]
            for row in rows:
                a,b=(parsed[row['uid']][0][name] for name in (first,second))
                if a['complete'] and b['complete']:
                    shared.append(dict(_meta(row),**{f:a[f]-b[f] for f in PAIRED_FIELDS}))
            comparisons[first+'_minus_'+second]=dict(registered_queries=len(rows),common_complete_queries=len(shared),
                metrics={f:balanced_summary(shared,f,seed,draws) for f in PAIRED_FIELDS},
                raw_score_margin_not_subtracted=True,
                scope='SAME_QUERY_PAIRED_DIFFERENCES_THEN_RECORDING_MUSIC_GROUP_BALANCE; RAW_METRIC_SIGN_NOT_QUALITY_VERDICT')
        strata[dataset+'|'+role]=dict(dataset=dataset,role=role,registered_queries=len(rows),
            methods=method_summaries,paired_method_differences=comparisons,
            music_group_evidence=[r for r in audit['music_group_evidence'] if (r['dataset'],r['role'])==(dataset,role)])
    return dict(schema='R66_PAIRED_SOURCE_CLOCK_METRICS_V1',registered_queries=len(queries),methods=methods,
        population=audit,strata=strata,queries=query_results,
        status='REGISTERED_CLOCK_DIAGNOSTIC' if queries else 'NOT_EVALUABLE_NO_REGISTERED_QUERIES',
        offsets=list(OFFSETS),truth=0,tie_tolerance=SCORE_TOLERANCE,
        tie_policy='PAIRWISE_HALF_CREDIT; TOP1_UNIFORM_AMONG_ALL_MAXIMA; EXPECTED_MAE_ACROSS_MAXIMA',
        no_information_tied_baselines=dict(pairwise_win_rate=.5,top1_accuracy=.2,mae_frames=9.2),
        cross_dataset_or_role_pooling_performed=False,operational_qualified=False,
        semantic_alignment_truth='UNKNOWN; CORRECT_REFERS_TO_REGISTERED_SOURCE_CLOCK_ONLY',
        wrong_song_scope='DIAGNOSTIC_ASSOCIATION_NOT_ARTISTIC_NEGATIVE; SHARED_DONORS_NOT_IID',
        training_steps=0,planner_runs=0,can_route=False)


summarize_queries=summarize
