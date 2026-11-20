"""A common, score-blind original-E1W candidate bank and synthetic clock audit.

An injected correspondence is mathematical supervision, not artistic truth.
Candidate generation never receives SDMR scores, and oracle paths are never
inserted into the bank used by rankers.
"""
import hashlib
import json

import numpy as np

from dunhuang.research.complete_r36.core import multi_warp
from dunhuang.research.warp_coupling_r53 import warp_lattice as frozen
from dunhuang.research.warp_coupling_r53.common import array_hash


KS = (1, 4, 8, 16)
Q_ATOL = 1e-9


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf8')).hexdigest()


def _interval(value, name):
    if len(value) != 2 or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) for t in value):
        raise ValueError(name+' must be a half-open integer frame interval')
    a, b = map(int, value)
    if a < 0 or b-a < 2:
        raise ValueError(name+' must retain two actual frame centers')
    return [a, b]


def _case(case):
    source = _interval(case['source_interval'], 'source')
    target = _interval(case['target_interval'], 'target')
    if target[1]-target[0] != 240:
        raise ValueError('The fixed Core must contain exactly240 target frames')
    support = _interval(case.get('support_component', case.get('source_support', source)), 'support')
    if not support[0] <= source[0] < source[1] <= support[1]:
        raise ValueError('Original full source interval leaves accepted support')
    phases = case.get('phase_inputs', case.get('phases', []))
    beats = case['beats']
    return source, target, support, phases, beats


def _candidate(case, source, target, support, value, origin, cost_rank=None):
    warp = frozen.validate(value['time_warp'], source, target)
    ticks = np.arange(*target)
    q = np.interp(ticks, warp[:, 1], warp[:, 0])
    if (len(q) != 240 or not np.isfinite(q).all() or not (np.diff(q) > 0).all()
            or q[0] != source[0] or q[-1] != source[1]-1
            or np.floor(q).min() < support[0] or np.ceil(q).max() >= support[1]):
        raise ValueError('Candidate actual source reads leave the accepted component')
    rates = np.diff(warp[:, 1])/np.diff(warp[:, 0])
    qhash = array_hash(q)
    identity = dict(source_uid=case.get('source_uid'), event_uid=case.get('event_uid'),
                    source_interval=source, target_interval=target,
                    support_component=support, q_sha256=qhash)
    return dict(uid=_digest(identity), time_warp=warp.tolist(), q=q.tolist(), cost=float(value['cost']),
        q_sha256=qhash, source_interval=list(source), target_interval=list(target),
        support_component=list(support), origin=origin,
        original_cost_ranks=[cost_rank] if cost_rank is not None else [],
        q_merged_duplicate_count=0, merged_q_max_abs_difference_frames=0.,
        source_to_target_rate_min=float(rates.min()), source_to_target_rate_max=float(rates.max()),
        target_to_source_rate_min=float((1/rates).min()), target_to_source_rate_max=float((1/rates).max()),
        source_to_target_rate_units='TARGET_FRAMES_PER_SOURCE_FRAME',
        inverse_rate_units='SOURCE_FRAMES_PER_TARGET_FRAME_NOT_THE_SAME_RATE_BAND',
        interpolation_read_support=sorted(set(np.floor(q).astype(int).tolist()+np.ceil(q).astype(int).tolist())),
        interpolation='FROZEN_R53_PIECEWISE_LINEAR_ACTUAL_FRAME_CENTERS',
        actual_body_hash='CALLER_MUST_COMPILE_FROM_ITS_OWN_FULL_SOURCE_R',
        SDMR_scores_used=False, semantic_truth='UNKNOWN')


def bank(case, max_k=16, seed=67067):
    """One fixed bank; all K prefixes and all rankers share these actual q arrays.

    Only the first max_k original-cost paths are scanned. Deduplication may make
    the bank smaller, even when more legal paths exist. No replacement is drawn.
    `seed` is bound as design provenance; best-cost enumeration is deterministic.
    """
    if isinstance(max_k, (bool, np.bool_)) or not isinstance(max_k, (int, np.integer)) or not 1 <= max_k <= 16:
        raise ValueError('Registered candidate scan budget must be1..16')
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError('Registered integer design seed required')
    source, target, support, phases, beats = _case(case)
    baseline = multi_warp(source, target, phases, beats)
    if baseline is None:
        raise ValueError('No original legal E1-W baseline; no source or boundary replacement')
    if baseline['time_warp'] != case['time_warp']:
        raise ValueError('Frozen baseline does not equal original multi_warp exactly')
    graph = frozen.lattice(source, target, phases, beats)
    count = int(frozen.suffix(graph)[0][0])
    if count < 1:
        raise ValueError('Original baseline absent from the original E1-W lattice')
    first = _candidate(case, source, target, support, baseline, 'FROZEN_BASELINE')
    output = [first]
    scanned = frozen.best(graph, int(max_k))
    duplicate_records = []
    for rank, value in enumerate(scanned, 1):
        item = _candidate(case, source, target, support, value, 'ORIGINAL_R53_BEST_COST', rank)
        duplicate = next((row for row in output if np.allclose(row['q'], item['q'], rtol=0, atol=Q_ATOL)), None)
        if duplicate is not None:
            diff = float(np.max(np.abs(np.asarray(duplicate['q'])-item['q'])))
            duplicate['original_cost_ranks'].append(rank)
            duplicate['q_merged_duplicate_count'] += 1
            duplicate['merged_q_max_abs_difference_frames'] = max(duplicate['merged_q_max_abs_difference_frames'], diff)
            duplicate_records.append(dict(original_cost_rank=rank, representative_uid=duplicate['uid'], max_q_difference_frames=diff))
        else:
            output.append(item)
    output = output[:int(max_k)]
    q = np.asarray([row['q'] for row in output])
    maximum = np.max(np.abs(q[:, None]-q[None, :]), axis=-1)
    mean = np.mean(np.abs(q[:, None]-q[None, :]), axis=-1)
    synthetic = bool(phases) and all(p['type'] == 'synthetic_grid_anchor' for p in phases)
    result = dict(schema='R67_COMMON_ORIGINAL_E1W_BANK_V1', status='BUILT',
        source_interval=source, target_interval=target, support_component=support,
        candidates=output, prefixes={str(k): [r['uid'] for r in output[:k]] for k in KS},
        requested_scan_budget=int(max_k), scanned_original_cost_paths=len(scanned),
        legal_path_count=count, actual_unique_bank_q_count=len(output),
        full_domain_unique_q_count=None,
        unique_count_scope='ONLY_THE_SCANNED_ORIGINAL_COST_PREFIX_NOT_FULL_DOMAIN_CAPACITY',
        shortfall={str(k): max(0, k-len(output)) for k in KS},
        baseline_uid=first['uid'], baseline_time_warp_exact=True,
        q_equivalence_atol_source_frames=Q_ATOL, q_equivalence_rtol=0.,
        qspread=dict(units='ORIGINAL_SOURCE_FRAMES', pairwise_max_abs=maximum.tolist(),
                     pairwise_mean_abs=mean.tolist(), max_abs=float(maximum.max())),
        duplicate_paths=duplicate_records, graph=graph, graph_sha256=_digest(graph),
        source_R_sha256=case.get('source_R_sha256'), design_seed=int(seed),
        order='FROZEN_BASELINE_THEN_ORIGINAL_R53_BEST_COST_LEXICOGRAPHIC_PATH_TIE',
        design_anchor_scope='SYNTHETIC_GRID_NOT_MOTION_PEAKS_OR_MUSIC_BEATS' if synthetic else 'FROZEN_CASE_PHASES_AND_BEATS',
        actual_rotation_hashes_verified=False, scores_consulted=False, ranker_calls=0,
        candidate_replacements=0, artificial_duplicate_fill=0,
        oracle_candidates_appended=False, training_steps=0)
    result['bank_sha256'] = _digest({k: v for k, v in result.items() if k != 'bank_sha256'})
    return result


def _map(value):
    x = np.asarray(value, float)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2 or not np.isfinite(x).all() or not (np.diff(x, axis=0) > 0).all():
        raise ValueError('Finite strictly monotone piecewise-linear map required')
    return x


def _compose(outer, inner):
    """All composition breakpoints, including preimages of outer-map knots."""
    f, g = _map(outer), _map(inner)
    if g[0, 1] < f[0, 0] or g[-1, 1] > f[-1, 0]:
        raise ValueError('Composed source map leaves true original frame support')
    knots = f[(f[:, 0] >= g[0, 1])&(f[:, 0] <= g[-1, 1]), 0]
    x = np.unique(np.r_[g[:, 0], np.interp(knots, g[:, 1], g[:, 0])])
    y = np.interp(np.interp(x, g[:, 0], g[:, 1]), f[:, 0], f[:, 1])
    points = list(map(list, np.c_[x, y]))
    k = 1
    while k < len(points)-1:
        a, p, b = np.asarray(points[k-1]), np.asarray(points[k]), np.asarray(points[k+1])
        if abs(p[1]-(a[1]+(p[0]-a[0])*(b[1]-a[1])/(b[0]-a[0]))) <= 1e-12:
            points.pop(k)
        else:
            k += 1
    return np.asarray(points)


def _rates(value):
    value = _map(value)
    r = np.diff(value[:, 1])/np.diff(value[:, 0])
    return dict(minimum=float(r.min()), maximum=float(r.max()),
                in_075_133=bool(np.all((r >= .75-1e-10)&(r <= 1.33+1e-10))))


def inject(case, bank_value, key):
    """Choose an existing nonbaseline ID first; never add/retry a true candidate.

    C(u)=R(h(u)), h=tau0^-1 o tau_j. Consequently the correct resampling is
    q_truth=tau_j^-1. Requiring h^-1 inside the same numerical band is an extra
    constructor qualification, not the E1-W inverse-q rate definition.
    """
    source, target, _, _, _ = _case(case)
    if source != [0, 240] or target != [0, 240]:
        raise ValueError('Controlled injection requires the declared common0..239 grid')
    if bank_value['source_interval'] != source or bank_value['target_interval'] != target:
        raise ValueError('Previously frozen candidate domain differs from this case')
    rows = bank_value['candidates']
    if not rows or rows[0]['uid'] != bank_value['baseline_uid']:
        raise ValueError('Frozen baseline-first candidate bank required')
    base = rows[0]
    w0 = frozen.validate(case['time_warp'], source, target)
    if not np.array_equal(w0, base['time_warp']):
        raise ValueError('Injection baseline differs from the previously fixed bank')
    digest = hashlib.sha256(str(key).encode('utf8')).hexdigest()
    proof = dict(constructor='C(U)=R(H(U)); H=TAU0_INVERSE_COMPOSE_TAU_J',
        noncircular=True, endpoints_preserved=True, candidates_added=0, selected_before_legality_check=True,
        candidate_selection='SHA256_KEY_MOD_EXISTING_NONBASELINE_BANK_ORDER', key_sha256=digest,
        inverse_injection_band_is_extra_constructor_qualification=True,
        correspondence_truth='KNOWN_SYNTHETIC_CLOCK_NOT_ARTISTIC_OR_SEMANTIC_TRUTH',
        fallback_used=False, interpolation_floor='DISCRETE_BODY_DOUBLE_RESAMPLING_MUST_BE_MEASURED_BY_CALLER')
    if len(rows) < 2:
        return dict(status='NOT_CONSTRUCTIBLE', reason='NO_DISTINCT_NONBASELINE_CANDIDATE', known_candidate_uid=None, legality=proof)
    chosen = rows[1+int(digest, 16) % (len(rows)-1)]
    tau = frozen.validate(chosen['time_warp'], source, target)
    h = _compose(w0[:, ::-1], tau)
    recovery_tau = _compose(w0, h)
    rates = dict(injection_h=_rates(h), inverse_h=_rates(h[:, ::-1]), recovery_tau=_rates(recovery_tau))
    proof.update(rates=rates, selected_candidate_remains_legal=True)
    result = dict(known_candidate_uid=chosen['uid'], h=h.tolist(), truth_tau=recovery_tau.tolist(), legality=proof)
    if not all(v['in_075_133'] for v in rates.values()):
        return dict(result, status='NOT_CONSTRUCTIBLE', reason='PRESELECTED_INJECTION_OR_INVERSE_OUTSIDE_CONSTRUCTOR_RATE_BAND')
    q0 = np.asarray(base['q'], float)
    qtruth = np.interp(np.arange(240), recovery_tau[:, 1], recovery_tau[:, 0])
    if not np.allclose(qtruth, chosen['q'], rtol=0, atol=Q_ATOL):
        raise ValueError('Known recovery map is not the previously registered actual candidate q')
    qinject = np.interp(np.arange(240), h[:, 0], h[:, 1])
    restored_clock = np.interp(qtruth, h[:, 0], h[:, 1])
    if (qinject[0] != 0 or qinject[-1] != 239 or not (np.diff(qinject) > 0).all()
            or np.floor(qinject).min() < 0 or np.ceil(qinject).max() >= 240
            or not np.allclose(restored_clock, q0, rtol=0, atol=Q_ATOL)):
        raise ValueError('Known source support or composed recovery identity failed')
    proof.update(known_candidate_in_bank=True,
                 composed_clock_max_error_frames=float(np.max(abs(restored_clock-q0))))
    return dict(result, status='CONSTRUCTED', q_injected=qinject.tolist(), truth_q=qtruth.tolist(),
                q_injected_sha256=array_hash(qinject), truth_q_sha256=array_hash(qtruth))


def _support(support, frames):
    if support is None:
        return np.ones(frames, bool)
    s = np.asarray(support)
    if s.dtype == bool and s.shape == (frames,):
        mask = s.copy()
    else:
        if s.shape != (2,) or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, np.integer)) for t in support):
            raise ValueError('Support must be local target index[a,b) or a boolean mask')
        a, b = map(int, support)
        if not 0 <= a < b <= frames:
            raise ValueError('Actual local target support required')
        mask = (np.arange(frames) >= a)&(np.arange(frames) < b)
    if not mask.any():
        raise ValueError('No clock-error target support')
    return mask


def _evaluation(bank_value, h, baselineq, support):
    h, baseline = _map(h), np.asarray(baselineq, float)
    frames = bank_value['target_interval'][1]-bank_value['target_interval'][0]
    a, b = bank_value['source_interval']
    if (baseline.shape != (frames,) or not np.isfinite(baseline).all()
            or not (np.diff(baseline) > 0).all()
            or not np.allclose(baseline, bank_value['candidates'][0]['q'], rtol=0, atol=Q_ATOL)):
        raise ValueError('Actual full frozen baseline q required')
    if not np.array_equal(h[[0, -1]], [[a, a], [b-1, b-1]]):
        raise ValueError('Injected map must preserve full original source endpoints')
    return h, baseline, _support(support, frames)


def clock_errors(bank_value, h, baselineq, support=None):
    """Evaluate mean_support|h(q_k(t))-q0(t)|; never select candidates here."""
    h, baseline, mask = _evaluation(bank_value, h, baselineq, support)
    truth = np.interp(baseline, h[:, 1], h[:, 0])
    rows = []
    for candidate in bank_value['candidates']:
        q = np.asarray(candidate['q'], float)
        if q.shape != baseline.shape or q.min() < h[0, 0] or q.max() > h[-1, 0]:
            raise ValueError('Candidate source reads leave the injected source domain')
        error = np.interp(q, h[:, 0], h[:, 1])-baseline
        inverse_error = q-truth
        rows.append(dict(candidate_uid=candidate['uid'], mae_frames=float(abs(error[mask]).mean()),
            rmse_frames=float(np.sqrt(np.mean(error[mask]**2))), max_error_frames=float(abs(error[mask]).max()),
            inverse_q_mae_frames=float(abs(inverse_error[mask]).mean()), support_frames=int(mask.sum()),
            units='ORIGINAL_SOURCE_FRAMES', scope='SYNTHETIC_COMPOSED_CLOCK_NOT_ARTISTIC_TRUTH'))
    return rows


def domain_oracle(bank_value, h, baselineq, support=None):
    """Exact additive clock-error oracle on the full original DAG, not just16.

    It is an evaluation lower bound. Its path is never appended to the bank.
    Each target frame belongs to exactly one half-open edge interval; the fixed
    final center is included separately. No full-path enumeration is necessary.
    """
    graph = bank_value['graph']
    h, baseline, mask = _evaluation(bank_value, h, baselineq, support)
    A, B = bank_value['target_interval']; ticks = np.arange(A, B)
    states = {0: (0., [float(A)])}
    for k in range(1, len(graph['source'])):
        current = {}
        for j, y in enumerate(graph['options'][k]):
            choices = []
            for i, (loss, path) in states.items():
                if not frozen.legal(graph, k, i, j):
                    continue
                y0 = graph['options'][k-1][i]
                selected = mask&(ticks >= y0)&(ticks < y)
                q = graph['source'][k-1]+(ticks[selected]-y0)/(y-y0)*(graph['source'][k]-graph['source'][k-1])
                edge_loss = float(abs(np.interp(q, h[:, 0], h[:, 1])-baseline[selected]).sum())
                choices.append((loss+edge_loss, path+[float(y)]))
            if choices:
                current[j] = min(choices, key=lambda item: (item[0], item[1]))
        states = current
    if 0 not in states:
        return dict(status='FULL_DOMAIN_ORACLE_NO_LEGAL_PATH', minimum_mae_frames=None,
                    evaluation_only=True, appended_to_bank=False)
    loss, ys = states[0]
    if mask[-1]:
        loss += abs(float(np.interp(graph['source'][-1], h[:, 0], h[:, 1]))-baseline[-1])
    warp = np.c_[graph['source'], ys]
    frozen.validate(warp, bank_value['source_interval'], bank_value['target_interval'])
    q = np.interp(ticks, warp[:, 1], warp[:, 0])
    measured = float(abs(np.interp(q, h[:, 0], h[:, 1])-baseline)[mask].mean())
    if not np.isclose(measured, loss/mask.sum(), rtol=0, atol=1e-10):
        raise ValueError('Additive full-domain clock oracle did not reproduce its dense q loss')
    bank_rows = clock_errors(bank_value, h, baseline, support)
    bank_best = min(bank_rows, key=lambda row: (row['mae_frames'], row['candidate_uid']))
    return dict(status='EXACT_FULL_ORIGINAL_DAG_CLOCK_ORACLE', minimum_mae_frames=measured,
        time_warp=warp.tolist(), q=q.tolist(), support_frames=int(mask.sum()),
        bank_minimum_mae_frames=bank_best['mae_frames'], bank_oracle_uid=bank_best['candidate_uid'],
        legal_path_count=bank_value['legal_path_count'], evaluation_only=True,
        appended_to_bank=False, known_truth_used_only_for_diagnostic=True,
        scope='ADDITIVE_SYNTHETIC_CLOCK_LOWER_BOUND_NOT_ARTISTIC_OR_SO3_INTERPOLATION_ERROR')
