"""Local target-clock interventions on a frozen piecewise-linear source clock.

The source path, source order and 240-frame budget stay fixed.  The compact
C3 bump is a post-map of the existing E1-W clock; it does not smooth existing
E1-W corners or establish music quality, semantic phase truth or physics.
"""
from collections.abc import Mapping
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq

FRAMES = 240
RATE_LIMITS = (.75, 1.33)
_BUMP = Polynomial([1., 0., -4., 0., 6., 0., -4., 0., 1.])


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
        raise ValueError('Finite numeric '+name+' required')
    return float(value)


def _integer(value, name):
    value = _real(value, name)
    if value != int(value):
        raise ValueError('Integer '+name+' required')
    return int(value)


def _interval(value, name, expected=None):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError('Half-open '+name+' interval required')
    a, b = [_integer(v, name) for v in value]
    if a < 0 or b-a < 2 or expected is not None and b-a != expected:
        raise ValueError('Actual frame centers with '+str(expected or 'at least two')+' frames required: '+name)
    return a, b


def _setup(time_warp, target_interval):
    A, B = _interval(target_interval, 'target', FRAMES)
    try:
        raw = np.asarray(time_warp, dtype=object)
        if any(isinstance(v, (bool, np.bool_, str)) or not isinstance(v, (int, float, np.integer, np.floating)) for v in raw.flat):
            raise ValueError('Numeric non-boolean frame knots required')
        w = np.asarray(time_warp, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('Finite numeric source-to-target knots required') from error
    if w.ndim != 2 or w.shape[1:] != (2,) or len(w) < 2 or not np.isfinite(w).all() or not (np.diff(w, axis=0) > 0).all():
        raise ValueError('Strictly increasing finite source-to-target knot pairs required')
    if np.asarray(time_warp).dtype.kind == 'b':
        raise ValueError('Boolean knots are not frame positions')
    a, last = [_integer(v, 'source endpoint frame') for v in w[[0, -1], 0]]
    if a < 0 or last <= a or w[0, 1] != A or w[-1, 1] != B-1:
        raise ValueError('Map endpoints must be the actual source and target frame centers')
    return w, (a, last+1), (A, B)


def _bump(z, order=0):
    x = np.asarray(z, dtype=float)
    answer = np.zeros_like(x)
    inside = abs(x) < 1
    # Factored value avoids cancellation near the support boundary.  Higher
    # derivatives are evaluated only inside the open support; C3 jets are zero.
    if order == 0:
        answer[inside] = (1-x[inside]**2)**4
    else:
        answer[inside] = _BUMP.deriv(order)(x[inside])
    return float(answer) if answer.ndim == 0 else answer


def _map(w, source):
    return np.interp(source, w[:, 0], w[:, 1])


def _source(w, target):
    return np.interp(target, w[:, 1], w[:, 0])


def _guard(w, centre=None, delta=0., half_width=60.):
    """Continuous extrema on every original E1-W piece, including one-sided jets."""
    pieces = []
    sites = []
    if centre is not None:
        sites = [centre-half_width, centre+half_width,
                 centre-half_width/math.sqrt(7), centre+half_width/math.sqrt(7)]
    for left, right in zip(w[:-1], w[1:]):
        rate = float((right[1]-left[1])/(right[0]-left[0]))
        target_sites = sorted(set([float(left[1]), float(right[1])]+
                                 [float(u) for u in sites if left[1] <= u <= right[1]]))
        values = [rate*(1+delta/half_width*_bump((u-centre)/half_width, 1))
                  if centre is not None else rate for u in target_sites]
        pieces.append(dict(source_interval=left[[0]].tolist()+right[[0]].tolist(),
            baseline_target_interval=[float(left[1]), float(right[1])],
            baseline_rate=rate, derivative_extrema_baseline_target_frames=target_sites,
            source_to_target_rates=list(map(float, values))))
    values = [v for p in pieces for v in p['source_to_target_rates']]
    lo, hi = min(values), max(values)
    # Same frozen physical limits; no feasibility repair, clipping or shrinking.
    legal = RATE_LIMITS[0] <= lo and hi <= RATE_LIMITS[1]
    return dict(legal=bool(legal), source_to_target_rate_min=float(lo),
                source_to_target_rate_max=float(hi), rate_limits=list(RATE_LIMITS),
                inverse_source_rate_min=float(1/hi) if hi > 0 else None,
                inverse_source_rate_max=float(1/lo) if lo > 0 else None,
                rate_units='TARGET_FRAMES_PER_SOURCE_FRAME',
                inverse_rate_units='SOURCE_FRAMES_PER_TARGET_FRAME',
                analytic_scope='EVERY_PIECE_CONTINUOUS_ONE_SIDED_RATE_EXTREMA',
                global_C1_claim=False, global_C2_claim=False,
                pieces=pieces, threshold_relaxed=False, feasibility_repair=False)


def _common(w, source, target, method):
    return dict(method=method, source_interval=list(source), target_interval=list(target),
        frames=FRAMES, time_warp=w.tolist(), map_direction='SOURCE_TO_TARGET_FRAME_CENTERS',
        map_smoothness='ORIGINAL_PIECEWISE_LINEAR_BASELINE_CORNERS_RETAINED',
        semantic_event_truth='UNKNOWN', human_reviewed=False, artistic_quality='NOT_EVALUABLE',
        Native24='NOT_EVALUABLE', can_route=False, fallback_used=False,
        target_interval_origin_frame=target[0])


def _plot(w, centre=None, delta=0., half_width=60.):
    source = np.unique(np.r_[np.linspace(w[0, 0], w[-1, 0], max(241, int(w[-1, 0]-w[0, 0])*4+1)), w[:, 0]])
    target = _map(w, source)
    if centre is not None:
        target = target+delta*_bump((target-centre)/half_width)
    return source, target


def baseline(time_warp, target_interval):
    """The exact E1-W inverse at the same 240 actual target frame centers."""
    w, source, target = _setup(time_warp, target_interval)
    result = _common(w, source, target, 'E1W_FROZEN')
    proof = _guard(w)
    result['proof'] = proof
    if not proof['legal']:
        return dict(result, status='NOT_FEASIBLE', reason='BASELINE_OUTSIDE_FROZEN_RATE_LIMITS')
    q = _source(w, np.arange(*target, dtype=float))
    x, y = _plot(w)
    return dict(result, status='LEGAL', reason=None, source_positions=q, q=q,
                source_grid=x, target_grid=y,
                dose=dict(max_abs_source_clock_shift_frames=0., max_abs_target_clock_offset_frames=0.))


def uniform(source_interval, target_interval):
    """A separate linear baseline, never a fallback for a rejected intervention."""
    a, b = _interval(source_interval, 'source')
    A, B = _interval(target_interval, 'target', FRAMES)
    result = baseline([[a, A], [b-1, B-1]], target_interval)
    result['method'] = 'UNIFORM_EXPLICIT'
    return result


def _event(event, source):
    if event is None:
        return None
    if not isinstance(event, Mapping):
        raise ValueError('Original source event record required')
    f = _integer(event.get('frame'), 'event frame')
    span = event.get('evidence_span')
    if not isinstance(span, (list, tuple, np.ndarray)) or len(span) != 2:
        raise ValueError('Original half-open event evidence required')
    a, b = [_integer(v, 'event evidence frame') for v in span]
    if not 0 <= a <= f < b:
        raise ValueError('Event representative must lie in its actual evidence span')
    return dict(event, frame=f, evidence_span=[a, b])


def intervene(time_warp, target_interval, event, delta=6., half_width=60., guard_frames=4):
    """τδ=F∘τ0, F(u)=u+δ(1-((u-c)/H)^2)^4 on the open support.

    Positive delta delays the registered event on the target clock.  Both the
    event and compensation regions change: this is not an isolated semantic
    stage edit.  Source/target endpoints are frame centers, not half-open ends.
    """
    delta, half_width = _real(delta, 'delta'), _real(half_width, 'half_width')
    guard_frames = _integer(guard_frames, 'guard_frames')
    if half_width <= 0 or not 1 <= guard_frames < FRAMES//2:
        raise ValueError('Positive support half-width and nonempty protected edge frames required')
    w, source, target = _setup(time_warp, target_interval)
    base = baseline(time_warp, target_interval)
    result = _common(w, source, target, 'COMPACT_C3_TARGET_POSTMAP_ON_FROZEN_E1W')
    result.update(delta_target_frames=delta, support_half_width_target_frames=half_width,
                  guard_frames_each_side=guard_frames, baseline_proof=base['proof'])
    if base['status'] != 'LEGAL':
        return dict(result, status='NOT_FEASIBLE', reason='BASELINE_OUTSIDE_FROZEN_RATE_LIMITS')
    event = _event(event, source)
    if event is None:
        return dict(result, status='NO_EVENT', reason='REGISTERED_TYPE_HAS_NO_ELIGIBLE_EVENT')
    result['event'] = event
    a, b = source; A, B = target
    f = event['frame']; ea, eb = event['evidence_span']
    if not a <= ea <= f < eb <= b:
        return dict(result, status='NOT_FEASIBLE', reason='EVENT_EVIDENCE_NOT_WHOLE_INSIDE_SOURCE_CORE')
    centre = float(_map(w, f)); left, right = centre-half_width, centre+half_width
    result.update(event_target_frame_before=centre-A, event_target_frame_after=centre+delta-A,
        event_target_frame_before_absolute=centre, event_target_frame_after_absolute=centre+delta,
        bump_support_target_interval=[left-A, right-A],
        bump_support_target_interval_absolute=[left, right],
        support_scope='OPEN_CONTINUOUS_TARGET_SUPPORT_ZERO_VALUE_AND_C3_JETS_AT_BOUNDS')
    if left < A+guard_frames-1 or right > B-guard_frames:
        return dict(result, status='NOT_FEASIBLE', reason='BUMP_SUPPORT_INVADES_PROTECTED_EDGE_FRAMES')
    proof = _guard(w, centre, delta, half_width)
    result['proof'] = proof
    if not proof['legal']:
        return dict(result, status='NOT_FEASIBLE', reason='CONTINUOUS_SOURCE_TO_TARGET_RATE_OUTSIDE_FROZEN_LIMITS')
    targets = np.arange(A, B, dtype=float)
    edited = (targets > left)&(targets < right)
    # Exact copied values outside the support, not approximately recomputed zeros.
    q0 = base['source_positions']; q = q0.copy(); u = targets.copy()
    for i in np.flatnonzero(edited):
        u[i] = brentq(lambda v: v+delta*_bump((v-centre)/half_width)-targets[i],
                      left, right, xtol=1e-11, rtol=1e-14)
    q[edited] = _source(w, u[edited])
    if not np.isfinite(q).all() or not (np.diff(q) > 0).all() or q[0] != a or q[-1] != b-1:
        raise ValueError('Actual inverse map lost source support or monotonicity')
    protected = np.r_[np.arange(guard_frames), np.arange(FRAMES-guard_frames, FRAMES)]
    if not np.array_equal(q[~edited], q0[~edited]) or not np.array_equal(q[protected], q0[protected]):
        raise AssertionError('Internal intervention modified protected source-clock reads')
    original_centres = _map(w, q)
    recovered = original_centres+delta*_bump((original_centres-centre)/half_width)
    residual = float(np.max(abs(recovered-targets)))
    if residual > 1e-8:
        raise ValueError('Inverse-clock residual exceeds numerical contract')
    evidence_before = [int(np.searchsorted(q0, ea)), int(np.searchsorted(q0, eb))]
    evidence_after = [int(np.searchsorted(q, ea)), int(np.searchsorted(q, eb))]
    ids = np.flatnonzero(edited)
    edit_interval = [int(ids[0]), int(ids[-1]+1)] if len(ids) else [0, 0]
    focus = [max(edit_interval[0], min(evidence_before[0], evidence_after[0])),
             min(edit_interval[1], max(evidence_before[1], evidence_after[1]))]
    x, y = _plot(w, centre, delta, half_width)
    dose = dict(max_abs_source_clock_shift_frames=float(np.max(abs(q-q0))),
        rms_source_clock_shift_frames=float(np.sqrt(np.mean((q-q0)**2))),
        max_abs_target_clock_offset_frames=abs(delta), support_target_width_frames=2*half_width,
        baseline_rate_margin_lower=float(base['proof']['source_to_target_rate_min']-RATE_LIMITS[0]),
        baseline_rate_margin_upper=float(RATE_LIMITS[1]-base['proof']['source_to_target_rate_max']),
        postmap_perturbation_derivative_abs_max={}, curvature_scope='POSTMAP_ONLY; ORIGINAL_E1W_CORNERS_RETAINED_NO_GLOBAL_CURVATURE_CERTIFICATE')
    for order in range(1, 4):
        polynomial = _BUMP.deriv(order)
        roots = [float(r.real) for r in polynomial.deriv().roots() if abs(r.imag) < 1e-10 and -1 < r.real < 1]
        dose['postmap_perturbation_derivative_abs_max'][str(order)] = float(max(abs(polynomial(v)) for v in [-1., 1.]+roots)*abs(delta)/half_width**order)
    result.update(status='LEGAL', reason=None, source_positions=q, q=q, source_grid=x, target_grid=y,
        dose=dose, edited_frame_mask=edited, edited_target_interval=edit_interval,
        target_evidence_interval=evidence_after, baseline_target_evidence_interval=evidence_before,
        target_focus_interval=focus, region_interval_scope='LOCAL_OUTPUT_HALF_OPEN_INTEGER_FRAME_INDICES',
        compensation_intervals=[[edit_interval[0], focus[0]], [focus[1], edit_interval[1]]],
        protected_frame_indices=protected.tolist(), outside_support_q_exact=True,
        protected_q_exact=True, inverse_maximum_residual_target_frames=residual,
        interpolation_read_support=sorted(set(np.floor(q).astype(int).tolist()+np.ceil(q).astype(int).tolist())))
    return result
