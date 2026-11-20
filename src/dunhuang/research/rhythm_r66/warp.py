"""Frozen finite search for a smooth internal rhythm map on one fixed240 Core.

The motion source, order, endpoint frame centers and target length are fixed.
Only the internal source-to-target clock changes. Holds/turns are kinematic
proxies, not semantic labels. No Bridge is built or requalified here.
"""
import hashlib
import json

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from dunhuang.research.timing_r64.music import _reward, SCORE_PROTOCOL
from dunhuang.research.warp_coupling_r53.common import array_hash


CONFIG = dict(version='R66B_FIXED240_SMOOTH_FINITE_WARP_V1', fps=30,
    target_frames=240, source_to_target_rate_limits=[.75, 1.33],
    max_internal_phase_anchors=8, local_candidate_band_frames=15,
    linear_grid_offsets_frames=[-6, -3, 0, 3, 6], maximum_event_options=12,
    beam_width=128, speed_penalty_weight=.05, curvature_penalty_weight=.05,
    curvature_source_second_frames=30, minimum_dense_checks_per_segment=16,
    dense_source_step_frames=.25, numerical_roundoff_tolerance=1e-10,
    weights_selected_from_results=False, training_steps=0)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf8')).hexdigest()


def _interval(value, name):
    if len(value) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in value):
        raise ValueError(name+' must contain integer half-open frame boundaries')
    a, b = map(int, value)
    if not 0 <= a < b or b-a < 2:
        raise ValueError(name+' must retain at least two real frame centers')
    return a, b


def _kind(p):
    name = p['type']
    if name.endswith('speed_peak'):
        return 'speed_peak'
    if name.endswith('pose_hold') or name.endswith('_hold') or name == 'hold':
        return 'hold'
    if name.endswith('direction_turn') or name.endswith('root_turn_span') or name == 'turn':
        return 'turn'
    return None


def _phases(phases, interval):
    a, b = interval
    valid = []
    for p in phases:
        if _kind(p) is None:
            continue
        frame = float(p['frame'])
        if not np.isfinite(frame):
            raise ValueError('Finite original phase frame required')
        if a <= frame <= b-1:
            valid.append(dict(p))
    return valid


def _anchors(phases, interval):
    """Metadata-only kind coverage then time quantiles; phase frames never move."""
    a, b = interval
    pool = sorted({float(p['frame']) for p in phases if a < p['frame'] < b-1})
    if len(pool) <= CONFIG['max_internal_phase_anchors']:
        return pool
    selected = set()
    mid = (a+b-1)/2
    for kind in ('speed_peak', 'hold', 'turn'):
        candidates = sorted({float(p['frame']) for p in phases
                             if _kind(p) == kind and a < p['frame'] < b-1})
        if candidates:
            selected.add(min(candidates, key=lambda t: (abs(t-mid), t)))
    remaining = [t for t in pool if t not in selected]
    take = CONFIG['max_internal_phase_anchors']-len(selected)
    if take:
        selected.update(remaining[int(i)] for i in np.linspace(0, len(remaining)-1, take).round().astype(int))
    return sorted(selected)


def _clock(clock, target, control):
    A, B = target
    n = clock['frames']
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 0 <= A < B <= n:
        raise ValueError('Core must remain inside the actual full music frame clock')
    def events(name):
        x = np.asarray(clock[name], float)
        if (x.ndim != 1 or not np.isfinite(x).all() or len(x) and
                (x[0] < 0 or x[-1] > n-1 or (np.diff(x) <= 0).any())):
            raise ValueError('Sorted unique actual audio frame observations required')
        return x
    beats, accents = events('beat_frames'), events('accent_frames')
    curve = np.asarray(clock['accent_strength'], float)
    if curve.shape != (n,) or not np.isfinite(curve).all() or (curve < 0).any():
        raise ValueError('Frozen finite full-clock nonnegative accent strengths required')
    strengths = np.interp(accents, np.arange(n), curve)
    original = dict(beats=beats.tolist(), accents=accents.tolist(), strengths=strengths.tolist())
    if control:
        def shifted(x, values=None):
            x = x.copy(); inside = (x >= A)&(x < B)
            x[inside] = A+(x[inside]-A+(B-A)//2) % (B-A)
            order = np.argsort(x, kind='stable')
            return (x[order], values[order]) if values is not None else x[order]
        beats = shifted(beats)
        accents, strengths = shifted(accents, strengths)
        if len(beats) and beats[-1] > n-1 or len(accents) and accents[-1] > n-1:
            raise ValueError('Controlled event clock leaves actual frame-center support')
    used = dict(beats=beats.tolist(), accents=accents.tolist(), strengths=strengths.tolist())
    proof = dict(original_evidence_sha256=_digest(original), used_evidence_sha256=_digest(used),
        full_music_clock_sha256=clock.get('clock_sha256'),
        audio_sha256=clock.get('audio_sha256'), control=bool(control),
        local_event_shift_frames=(B-A)//2 if control else 0,
        local_event_shift_interval=[A, B] if control else None,
        audio_waveform_changed=False, outside_core_events_changed=False,
        control_scope='LOCAL_PROXY_CLOCK_SHIFT_NOT_ARTISTIC_INCOMPATIBILITY' if control else 'CORRECT_AUDIO_PROXY_CLOCK',
        score_protocol=SCORE_PROTOCOL)
    return beats, accents, strengths, proof


def _reward_at(target_peaks, beats, accents, strengths):
    sigma = SCORE_PROTOCOL['phase_tolerance_frames']
    if sigma != 3 or SCORE_PROTOCOL['primary_weights'] != dict(phase_beat_score=.5, phase_accent_score=.5):
        raise ValueError('Frozen R64 reward protocol changed')
    b = _reward(target_peaks, beats, sigma)
    a = _reward(target_peaks, accents, sigma, strengths)
    primary = .5*b['score']+.5*a['score'] if b['score'] is not None and a['score'] is not None else None
    return dict(phase_beat_score=b['score'], phase_accent_score=a['score'],
                primary_music_score=primary, objective_terms=dict(Beat=b, Accent=a))


def _curve(x, y):
    """Analytic derivative extrema and independent dense checks of every piece."""
    if not (np.diff(x) > 0).all() or not (np.diff(y) > 0).all():
        return None, dict(legal=False, reason='NON_MONOTONE_ANCHORS')
    tau = PchipInterpolator(x, y, extrapolate=False)
    low, high = CONFIG['source_to_target_rate_limits']
    tol = CONFIG['numerical_roundoff_tolerance']
    extrema, curvature, dense_x = [], [], []
    pieces = []
    for k, h in enumerate(np.diff(x)):
        c0, c1, c2, _ = tau.c[:, k]
        sites = [0., float(h)]
        if abs(c0) > 1e-16:
            critical = -c1/(3*c0)
            if 0 < critical < h:
                sites.append(float(critical))
        values = [float(3*c0*u*u+2*c1*u+c2) for u in sites]
        curv = [float(2*c1), float(6*c0*h+2*c1)]
        extrema.extend(values); curvature.extend(curv)
        dense_x.extend(np.linspace(x[k], x[k+1], max(CONFIG['minimum_dense_checks_per_segment'],
                         int(np.ceil(h/CONFIG['dense_source_step_frames']))+1)).tolist())
        pieces.append(dict(source_interval=[float(x[k]), float(x[k+1])],
            derivative_extremum_source_frames=[float(x[k]+u) for u in sites],
            rate_values=values, curvature_endpoint_values=curv))
    dense_x = np.unique(dense_x)
    dense_rate = tau.derivative()(dense_x)
    proof = dict(legal=True, analytic_rate_min=float(min(extrema)), analytic_rate_max=float(max(extrema)),
        dense_rate_min=float(dense_rate.min()), dense_rate_max=float(dense_rate.max()),
        dense_checks=len(dense_x), curvature_abs_max=float(max(abs(v) for v in curvature)),
        rate_units='TARGET_FRAMES_PER_SOURCE_FRAME',
        curvature_units='TARGET_FRAMES_PER_SOURCE_FRAME_SQUARED',
        roundoff_tolerance=tol, pieces=pieces)
    if min(extrema) < low-tol or max(extrema) > high+tol:
        proof.update(legal=False, reason='ANALYTIC_PCHIP_RATE_OUTSIDE_FROZEN_LIMITS')
        return None, proof
    if dense_rate.min() < low-tol or dense_rate.max() > high+tol:
        proof.update(legal=False, reason='DENSE_PCHIP_RATE_OUTSIDE_FROZEN_LIMITS')
        return None, proof
    return tau, proof


def _penalties(tau, rho):
    speed = curvature = 0.
    for k, h in enumerate(np.diff(tau.x)):
        c0, c1, c2, _ = tau.c[:, k]
        rate_error = Polynomial([c2/rho-1, 2*c1/rho, 3*c0/rho])
        rate_change = Polynomial([2*c1/rho, 6*c0/rho])*CONFIG['curvature_source_second_frames']
        v, a = (rate_error*rate_error).integ(), (rate_change*rate_change).integ()
        speed += v(h)-v(0); curvature += a(h)-a(0)
    span = tau.x[-1]-tau.x[0]
    return max(0., float(speed/span)), max(0., float(curvature/span))


def _score(x, y, phase_peaks, evidence, rho):
    tau, proof = _curve(np.asarray(x, float), np.asarray(y, float))
    if tau is None:
        return None, proof
    peaks = tau(phase_peaks)
    reward = _reward_at(peaks, *evidence)
    speed, smooth = _penalties(tau, rho)
    primary = reward['primary_music_score']
    objective = (primary-CONFIG['speed_penalty_weight']*speed-CONFIG['curvature_penalty_weight']*smooth
                 if primary is not None else None)
    return dict(tau=tau, proof=proof, anchors=np.c_[x, y].tolist(), reward=reward,
                speed_penalty=speed, curvature_penalty=smooth, objective=objective,
                target_peak_frames=peaks.tolist()), proof


def _materialize(candidate, source, target, support):
    a, b = source; A, B = target; lo, hi = support
    tau = candidate['tau']
    q = np.asarray([float(a) if t == A else float(b-1) if t == B-1 else
                    brentq(lambda s: float(tau(s))-t, a, b-1, xtol=1e-11, rtol=1e-14)
                    for t in range(A, B)])
    if (len(q) != 240 or not (np.diff(q) > 0).all()
            or q[0] != a or q[-1] != b-1
            or np.floor(q).min() < lo or np.ceil(q).max() >= hi):
        raise ValueError('Inverse clock leaves full real source support')
    source_grid = np.unique(np.r_[np.arange(a, b), tau.x])
    rate = tau.derivative()(source_grid)
    curvature = tau.derivative(2)(source_grid)
    endpoints = tau.derivative()([a, b-1])
    rho = (B-A-1)/(b-a-1)
    result = {k: v for k, v in candidate.items() if k not in ('tau',)}
    result.update(status='LEGAL', source_positions=q.tolist(), source_positions_sha256=array_hash(q),
        source_to_target_anchors=candidate['anchors'], time_warp=candidate['anchors'],
        map_interpolation='PCHIP_SOURCE_TO_TARGET; BRENTQ_INVERSE_AT_ACTUAL_TARGET_FRAME_CENTERS',
        source_positions_authoritative=True, map_smoothness='C1_NOT_C2_GUARANTEED',
        source_grid=source_grid.tolist(), target_grid=tau(source_grid).tolist(),
        source_to_target_rate=rate.tolist(), source_to_target_curvature=curvature.tolist(),
        inverse_source_rate_per_target_frame=(1/tau.derivative()(q)).tolist(),
        uniform_source_to_target_rate=rho, endpoint_source_to_target_rates=endpoints.tolist(),
        endpoint_source_per_target_rates=(1/endpoints).tolist(),
        endpoint_source_per_target_rate_ratio_vs_uniform=(rho/endpoints).tolist(),
        endpoint_velocity_changed_vs_uniform=bool(np.max(abs(endpoints-rho)) > 1e-10),
        endpoint_clock_rate_changed_vs_uniform=bool(np.max(abs(endpoints-rho)) > 1e-10),
        endpoint_velocity_change_scope='CLOCK_DERIVATIVE_PROXY_NOT_MEASURED_BODY_JET',
        endpoint_body_jets_evaluated=False,
        interpolation_read_support=sorted(set(np.floor(q).astype(int).tolist()+np.ceil(q).astype(int).tolist())),
        interpolation_support_inside_component=True,
        source_positions_units='ABSOLUTE_ORIGINAL_SOURCE_FRAME_CENTERS',
        phase_semantic_truth='UNKNOWN', postfilter_applied=False,
        transition48='RESERVED_NOT_REGENERATED', adjacent_bridge_revalidated=False,
        boundary_note='CORE_ENDPOINT_POSES_FIXED_BUT_ENDPOINT_JETS_MAY_CHANGE; NO_BRIDGE_QUALIFICATION')
    return result


def _setup(source_interval, target_interval, phases, clock, source_support, control):
    source, target = _interval(source_interval, 'source'), _interval(target_interval, 'target')
    if target[1]-target[0] != CONFIG['target_frames']:
        raise ValueError('Exactly240 target frames required')
    support = source if source_support is None else _interval(source_support, 'source_support')
    if not support[0] <= source[0] < source[1] <= support[1]:
        raise ValueError('Full source interval must stay inside accepted support')
    phase = _phases(phases, source)
    peaks = np.asarray([float(p['frame']) for p in phase if _kind(p) == 'speed_peak'])
    beats, accents, strengths, clock_proof = _clock(clock, target, control)
    rho = (target[1]-target[0]-1)/(source[1]-source[0]-1)
    return source, target, support, phase, peaks, (beats, accents, strengths), clock_proof, rho


def uniform(source_interval, target_interval, phases, clock, source_support=None, control=False):
    """Independent explicit linear baseline, never a hidden optimized fallback."""
    source, target, support, phase, peaks, evidence, clock_proof, rho = _setup(
        source_interval, target_interval, phases, clock, source_support, control)
    x, y = [source[0], source[1]-1], [target[0], target[1]-1]
    candidate, guard = _score(x, y, peaks, evidence, rho)
    base = dict(method='UNIFORM_LINEAR_EXPLICIT', source_interval=list(source),
        target_interval=list(target), source_support=list(support), clock=clock_proof,
        config=dict(CONFIG), config_sha256=_digest(CONFIG), fallback_used=False)
    if candidate is None:
        return dict(base, status='ILLEGAL', reason=guard['reason'], proof=guard)
    return dict(base, **_materialize(candidate, source, target, support))


def optimize(source_interval, target_interval, phases, clock, source_support=None, control=False):
    """Choose a legal PCHIP from a fixed finite beam, including explicit linear."""
    source, target, support, phase, peaks, evidence, clock_proof, rho = _setup(
        source_interval, target_interval, phases, clock, source_support, control)
    a, b = source; A, B = target
    internal = _anchors(phase, source)
    x = np.asarray([a]+internal+[b-1], float)
    linear = A+(x-a)*rho
    base = dict(method='R66B3_LOCAL_PROXY_CLOCK_SHIFT_PCHIP' if control else 'R66B2_SMOOTH_PCHIP',
        source_interval=list(source), target_interval=list(target), source_support=list(support),
        config=dict(CONFIG), config_sha256=_digest(CONFIG), clock=clock_proof,
        source_phase_anchors=internal, source_phase_frames_moved=False,
        anchor_selection='ONE_NEAREST_MIDPOINT_PER_KIND_THEN_TIME_QUANTILES_METADATA_ONLY',
        observed_speed_peak_count=len(peaks), total_eligible_phase_count=len(phase),
        phase_evidence_truncated_count=sum(1 for p in phase if p.get('evidence_span') and
            (p['evidence_span'][0] < a or p['evidence_span'][1] > b)),
        phase_semantic_truth='UNKNOWN', fallback_used=False,
        finite_search='DETERMINISTIC_BOUNDED_BEAM_NOT_GLOBAL_OPTIMALITY_PROOF',
        transition48='RESERVED_NOT_REGENERATED', adjacent_bridge_revalidated=False)
    low, high = CONFIG['source_to_target_rate_limits']
    if not low <= rho <= high:
        return dict(base, status='ILLEGAL', reason='FIXED_ENDPOINT_RATE_OUTSIDE_FROZEN_LIMITS',
                    endpoint_rate=rho, candidate_counts=dict(complete_paths_screened=0))
    if not len(peaks) or not len(evidence[0]) or not len(evidence[1]):
        return dict(base, status='NOT_EVALUABLE', reason='SPEED_PEAK_OR_AUDIO_PROXY_EVIDENCE_MISSING',
                    candidate_counts=dict(complete_paths_screened=0))
    options = [[float(A)]]
    event_grid = np.unique(np.r_[evidence[0], evidence[1]])
    for center in linear[1:-1]:
        near = sorted([float(t) for t in event_grid if A < t < B-1 and abs(t-center) <= CONFIG['local_candidate_band_frames']],
                      key=lambda t: (abs(t-center), t))[:CONFIG['maximum_event_options']]
        grid = [float(center+d) for d in CONFIG['linear_grid_offsets_frames'] if A < center+d < B-1]
        options.append(sorted(set(near+grid+[float(center)])))
    options.append([float(B-1)])
    counts = dict(anchor_options=[len(v) for v in options], beam_expansions=0,
        secant_rate_rejected=0, beam_pruned=0, complete_paths_screened=0,
        analytic_rate_rejected=0, dense_rate_rejected=0, legal_paths_scored=0,
        uniform_candidate_registered=True)
    states = [(0., [float(A)])]
    peak_count = max(len(peaks), 1)
    for k in range(1, len(x)):
        trials = []
        h = x[k]-x[k-1]
        for old_cost, path in states:
            for y in options[k]:
                counts['beam_expansions'] += 1
                rate = (y-path[-1])/h
                if not low <= rate <= high:
                    counts['secant_rate_rejected'] += 1
                    continue
                fraction = h/(b-a-1)
                cost = old_cost+CONFIG['speed_penalty_weight']*fraction*(rate/rho-1)**2
                if len(path) >= 2:
                    previous_h = x[k-1]-x[k-2]
                    previous_rate = (path[-1]-path[-2])/previous_h
                    approximate_curvature = 2*(rate-previous_rate)/(h+previous_h)
                    cost += CONFIG['curvature_penalty_weight']*fraction*(30*approximate_curvature/rho)**2
                multiplicity = int(np.sum(peaks == x[k]))
                if multiplicity:
                    reward = _reward_at(np.asarray([y]), *evidence)['primary_music_score']
                    cost -= reward*multiplicity/peak_count
                trials.append((float(cost), path+[float(y)]))
        trials.sort(key=lambda v: (v[0], v[1]))
        counts['beam_pruned'] += max(0, len(trials)-CONFIG['beam_width'])
        states = trials[:CONFIG['beam_width']]
    paths = {tuple(path) for _, path in states}
    paths.add(tuple(linear.tolist()))
    candidates = []
    for path in sorted(paths):
        counts['complete_paths_screened'] += 1
        candidate, guard = _score(x, np.asarray(path), peaks, evidence, rho)
        if candidate is None:
            if guard.get('reason', '').startswith('ANALYTIC'):
                counts['analytic_rate_rejected'] += 1
            else:
                counts['dense_rate_rejected'] += 1
            continue
        if candidate['objective'] is None:
            continue
        counts['legal_paths_scored'] += 1
        candidates.append(candidate)
    base['candidate_counts'] = counts
    if not candidates:
        return dict(base, status='ILLEGAL', reason='NO_LEGAL_CANDIDATE_IN_REGISTERED_FINITE_SEARCH')
    chosen = min(candidates, key=lambda v: (-v['objective'], v['speed_penalty'],
                 v['curvature_penalty'], [pair[1] for pair in v['anchors']]))
    result = _materialize(chosen, source, target, support)
    result.update(selected_uniform_candidate=bool(np.array_equal(np.asarray(chosen['anchors'])[:, 1], linear)),
        objective_definition='MAX(0.5*BEAT+0.5*ACCENT-0.05*MEAN((TAU_PRIME/RHO-1)^2)-0.05*MEAN((30*TAU_SECOND/RHO)^2))',
        objective_penalty_units='DIMENSIONLESS; CURVATURE_NORMALIZES_RATE_CHANGE_PER_SOURCE_SECOND',
        derivative_guard='ANALYTIC_ALL_SEGMENT_EXTREMA_PLUS_DENSE_CHECKS',
        source_order_preserved=True)
    return dict(base, **result)
