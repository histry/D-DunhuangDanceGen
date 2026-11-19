"""Fixed-budget R63 recompilation; frozen geometry receives edge observations only.

Changing a bridge length repays time to the two complete source Cores. Every
allocation recompiles its E1-W map and boundary jets before geometric screening.
There is no hidden cross-source target and no replacement for a rejected map.
"""
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from .common import array_hash, relative_fk
from dunhuang.research.complete_r36.core import multi_warp, rotation_sample
from dunhuang.research.warp_coupling_r53.warp_lattice import validate
from dunhuang.research.duration_r62.cross_source import (
    make_observed, solve_boundary, edge_jets, measure)


FPS = 30
DISPLAY_CONTEXT = 30
DURATIONS = (24, 36, 48, 60, 72)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _interval(value, name):
    if len(value) != 2:
        raise ValueError(name + ' must be a half-open pair')
    a, b = [_integer(v, name) for v in value]
    if a < 0 or b <= a:
        raise ValueError(name + ' must be positive half-open support')
    return a, b


def allocation(g, total=528, left=240, right=240, baseline_gap=48):
    """Repay bridge time without changing the declared pair music budget."""
    g, total, left, right, baseline_gap = [
        _integer(v, name) for v, name in
        zip((g, total, left, right, baseline_gap),
            ('gap', 'total', 'left', 'right', 'baseline_gap'))]
    if min(g, total, left, right, baseline_gap) <= 0:
        raise ValueError('All baseline lengths and gap must be positive')
    if left + baseline_gap + right != total:
        raise ValueError('Declared baseline does not equal the fixed budget')
    dleft = left + (baseline_gap-g)//2
    dright = total-g-dleft
    if min(dleft, dright) < DISPLAY_CONTEXT:
        raise ValueError('Allocation removes genuine boundary context')
    return dict(total_frames=total, left_frames=dleft, gap_frames=g,
                right_frames=dright, baseline_gap_frames=baseline_gap,
                left_core_debt_frames=dleft-left, right_core_debt_frames=dright-right,
                budget_exact=dleft+g+dright == total)


def _phases(phases):
    """R47 raw types require their measured body-region namespace for R36."""
    result = []
    for p in phases:
        item = dict(p)
        body = item.get('body_region', '')
        kind = item['type']
        if body and not kind.startswith(body+'_'):
            kind = body+'_'+kind
        item['type'] = kind
        result.append(item)
    return result


def dynamics(R, J, parents):
    """Root-relative body22 diagnostics, never a world-contact qualification."""
    R = np.asarray(R)
    P = relative_fk(R, J, parents)
    out = dict(frames=len(R), FK_path_length_mean_m=float(
        np.linalg.norm(np.diff(P, axis=0), axis=-1).sum(0).mean()),
        FK_motion_range_rms_m=float(np.sqrt(np.mean(np.var(P, axis=0)))))
    for n in (1, 2, 3):
        values = np.sqrt(np.mean((np.diff(P, n=n, axis=0)*FPS**n)**2, axis=(1, 2)))
        out['D'+str(n)+'_Q95'] = float(np.quantile(values, .95)) if len(values) else None
        out['D'+str(n)+'_max'] = float(values.max()) if len(values) else None
    tangent = Rotation.from_matrix(
        (R[:-1].swapaxes(-1, -2)@R[1:]).reshape(-1, 3, 3)).as_rotvec().reshape(-1, 22, 3)
    speed = np.linalg.norm(tangent, axis=-1)*FPS
    out['mean_joint_speed_radps'] = float(speed.mean()) if speed.size else 0.
    out['joint_speed_Q95_radps'] = float(np.quantile(speed, .95)) if speed.size else 0.
    return out


def compile_core(side, target_interval, beats, lib, J, parents):
    """Re-run frozen E1-W on a full original interval and report all reads."""
    a, b = _interval(side['source_interval'], 'source_interval')
    A, B = _interval(target_interval, 'target_interval')
    lo, hi = _interval(side['support_component'], 'support_component')
    uid = side['source_uid']
    meta = dict(event_uid=side['event_uid'], source_uid=uid,
                source_interval=[a, b], support_component=[lo, hi],
                target_interval=[A, B], target_frames=B-A,
                status='UNCOMPILED', semantic_truth='UNKNOWN',
                phase_proxy_only=True, source_interval_unchanged=True,
                warp_algorithm='FROZEN_R36_MULTI_WARP_ACTUAL_FRAME_CENTERS')
    if uid not in lib.rotations:
        meta.update(status='SOURCE_MISSING', reason='Original source UID not loaded')
        return meta, None
    source = np.asarray(lib.rotations[uid])
    if source.ndim != 4 or source.shape[1:] != (22, 3, 3):
        raise ValueError('Full original body22 rotations required')
    if not (0 <= lo <= a < b <= hi <= len(source)) or b-a < 2:
        meta.update(status='SOURCE_SUPPORT_REJECTED', reason='Full interval outside accepted component')
        return meta, None
    meta['source_R_sha256'] = array_hash(source)
    phase_input = _phases(side.get('phases', []))
    meta['phase_inputs'] = phase_input
    warp = multi_warp([a, b], [A, B], phase_input, beats)
    if warp is None:
        meta.update(status='E1W_ILLEGAL', reason='No frozen E1-W path for allocated duration')
        return meta, None
    meta['warp'] = warp
    try:
        w = validate(warp['time_warp'], [a, b], [A, B])
    except ValueError as error:
        meta.update(status='E1W_INVALID', reason=str(error))
        return meta, None
    q = np.interp(np.arange(A, B), w[:, 1], w[:, 0])
    floor = np.floor(q).astype(int)
    ceil = np.ceil(q).astype(int)
    if (not np.isfinite(q).all() or not (np.diff(q) > 0).all()
            or floor.min() < lo or ceil.max() >= hi
            or q[0] != a or q[-1] != b-1):
        meta.update(status='INTERPOLATION_SUPPORT_REJECTED', reason='Actual interpolation reads leave source component')
        return meta, None
    R = rotation_sample(source, q)
    if not np.isfinite(R).all():
        meta.update(status='SOURCE_NONFINITE', reason='Original source contains non-finite sampled observations')
        return meta, None
    # R36's sampler also reads floor+1 for integer positions (zero interpolation
    # weight). Preserve this implementation readset independently of ceil support.
    implementation_hi = np.minimum(floor+1, len(source)-1)
    if implementation_hi.max() >= hi:
        # A zero-weight endpoint read outside the component carries no pose
        # evidence, but record it rather than silently calling it an accepted read.
        outside = sorted(set(implementation_hi[implementation_hi >= hi].tolist()))
    else:
        outside = []
    rates = np.diff(w[:, 1])/np.diff(w[:, 0])
    meta.update(status='LEGAL_CORE', time_warp=w.tolist(), source_positions=q.tolist(),
                interpolation_read_support=sorted(set(floor.tolist()+ceil.tolist())),
                sampler_implementation_read_support=sorted(set(floor.tolist()+implementation_hi.tolist())),
                sampler_zero_weight_reads_outside_component=outside,
                positive_weight_interpolation_support_inside_component=True,
                time_warp_sha256=array_hash(w), source_positions_sha256=array_hash(q),
                full_core_R_sha256=array_hash(R), full_core_frames=len(R),
                rate_min=float(rates.min()), rate_max=float(rates.max()),
                boundary_source_positions_left=q[:4].tolist(),
                boundary_source_positions_right=q[-4:].tolist(),
                core_dynamics=dynamics(R, J, parents))
    return meta, R


def _beat_proxy(R, beats, origin, J, parents):
    P = relative_fk(R, J, parents)
    speed = np.linalg.norm(np.diff(P, axis=0), axis=-1).mean(1)*FPS
    peaks = find_peaks(speed, distance=8, prominence=.1*np.ptp(speed))[0]+1
    b = np.asarray([float(t)-origin for t in beats if origin <= t < origin+len(R)])
    distance = np.min(np.abs(peaks[:, None]-b[None, :]), axis=1) if len(peaks) and len(b) else np.array([])
    return dict(scope='EXISTING_BEATS_TO_FK_SPEED_PEAKS_PROXY_NOT_SEMANTIC_ALIGNMENT',
                peak_frames_local=peaks.tolist(), beat_frames_local=b.tolist(),
                mean_nearest_beat_error_seconds=float(distance.mean()/FPS) if len(distance) else None,
                peak_within_3_frames_fraction=float(np.mean(distance <= 3)) if len(distance) else None)


def evaluate_pair(case, g, lib, J, parents, reference, controls=False):
    """Compile both Cores anew, then splice frozen boundary-only geometry."""
    A, B = _interval(case['music_interval'], 'music_interval')
    if g not in DURATIONS or isinstance(g, (bool, np.bool_)):
        raise ValueError('Use a preregistered R63 gap grid value')
    alloc = allocation(g, total=B-A, left=case.get('baseline_left_frames', 240),
                       right=case.get('baseline_right_frames', 240))
    nl, nr = alloc['left_frames'], alloc['right_frames']
    gap = (nl, nl+g)
    key = case['key']+'_g'+str(g)
    rec = dict(key=key, case_key=case['key'], gap_frames=g, allocation=alloc,
               music_interval=[A, B], gap=list(gap), target_gap_interval=[A+nl, A+nl+g],
               baseline_duration_frames=48, status='CORE_RECOMPILE_PENDING', methods={},
               music_duration_unchanged=True, controls=bool(controls),
               full_source_intervals_retained=True, training_steps=0, planner_runs=0,
               hidden_truth_available=False, human_reviewed=False, verified_phase='UNKNOWN',
               Native24='NOT_EVALUABLE', physical_qualified=False,
               discarded_or_fallback_warps=0)
    names = ('G1', 'G0') if controls else ('G1',)
    if case.get('left') is None or case.get('right') is None:
        # The preregistered population retains missing units. Do not replace the
        # source, independently compile half a pair, or manufacture source IDs.
        for name in ('left', 'right'):
            side = case.get(name)
            rec[name] = dict(
                event_uid=side.get('event_uid') if side else None,
                source_uid=side.get('source_uid') if side else case.get(name+'_source_uid'),
                source_interval=side.get('source_interval') if side else None,
                support_component=side.get('support_component') if side else None,
                status='SOURCE_MISSING' if side is None else 'PAIR_SOURCE_MISSING',
                reason='Registered pair has no matchable original source unit',
                core_dynamics=None, semantic_truth='UNKNOWN', phase_proxy_only=True)
        rec.update(status='CORE_RECOMPILE_REJECTED',
                   case_status=case.get('status', 'NOT_MATCHABLE'),
                   missing_source_uids=case.get('missing_source_uids', []))
        for name in names:
            rec['methods'][name] = dict(constructible=False, passes_original_screen=False,
                reason='SOURCE_MISSING_NOT_MATCHABLE', arrays_key=None, fallback_used=False)
        return rec, {}
    left, L = compile_core(case['left'], [A, A+nl], case['beats'], lib, J, parents)
    right, R = compile_core(case['right'], [A+nl+g, B], case['beats'], lib, J, parents)
    rec.update(left=left, right=right)
    arrays = {}
    for name, data in (('left', L), ('right', R)):
        if data is not None:
            arrays[key+'_'+name] = data
    if L is None or R is None:
        rec['status'] = 'CORE_RECOMPILE_REJECTED'
        for name in names:
            rec['methods'][name] = dict(constructible=False, passes_original_screen=False,
                                       reason='E1W_OR_SOURCE_SUPPORT_REJECTED', arrays_key=None)
        return rec, arrays
    observed, local_gap = make_observed(L, R, g, context=DISPLAY_CONTEXT)
    curves, proof = solve_boundary(observed, local_gap, controls=controls)
    local_reads = proof['available_boundary_frame_indices']
    global_reads = (np.asarray(local_reads)+nl-DISPLAY_CONTEXT).tolist()
    edge = np.concatenate((L[-4:], R[:4]))
    jets = edge_jets(observed, local_gap)
    rec.update(status='GEOMETRY_EVALUATED', boundary_readset_sha256=array_hash(edge),
               boundary_jets_sha256=array_hash(jets), boundary_jets=jets.tolist(),
               boundary_available_pair_frame_indices=global_reads,
               boundary_proof=proof, solver_context_frames_per_side=DISPLAY_CONTEXT,
               solver_receives_only_four_edge_poses=True,
               source_boundary_readset=dict(left=left['source_positions'][-4:],
                                           right=right['source_positions'][:4]))
    full_observed = np.concatenate((L, np.full((g, 22, 3, 3), np.nan), R))
    for name in names:
        curve = curves[name]
        if curve is None:
            rec['methods'][name] = dict(constructible=False, passes_original_screen=False,
                reason='FROZEN_CHART_OR_CONSTRUCTION_REJECTED', arrays_key=None,
                guard=proof['guards'][name], fallback_used=False)
            continue
        middle = curve[local_gap[0]:local_gap[1]]
        Q = np.concatenate((L, middle, R))
        if Q.shape != (B-A, 22, 3, 3) or not np.isfinite(Q).all():
            raise AssertionError('Complete finite fixed-budget body22 trajectory required')
        exact = np.array_equal(Q[:nl], L) and np.array_equal(Q[nl+g:], R)
        if not exact:
            raise AssertionError('Geometry edited a source Core')
        m = measure(Q, full_observed, J, parents, gap, reference)
        m.update(arrays_key=key+'_'+name, full_R_sha256=array_hash(Q),
                 guard=proof['guards'][name], fallback_used=False,
                 core_exact=True, full_dynamics=dynamics(Q, J, parents),
                 beat_proxy=_beat_proxy(Q, case['beats'], A, J, parents))
        rec['methods'][name] = m
        arrays[key+'_'+name] = Q
    return rec, arrays


def _ratio(value, reference):
    return float(value/reference) if value is not None and reference is not None and abs(reference) > 1e-9 else None


def attach_retention(record, baseline):
    """Compare re-timed Core activity with g48; this is not hidden-truth error."""
    if record['case_key'] != baseline['case_key'] or baseline['gap_frames'] != 48:
        raise ValueError('Retention requires the same frozen pair at gap48')
    fields = ('FK_path_length_mean_m', 'FK_motion_range_rms_m',
              'mean_joint_speed_radps', 'D1_Q95', 'D2_Q95', 'D3_Q95')
    record['retention_reference_key'] = baseline['key']
    record['retention_scope'] = 'SAME_FULL_SOURCE_INTERVAL_RELATIVE_TO_RECOMPILED_FIXED48_NOT_HIDDEN_TRUTH'
    for side in ('left', 'right'):
        now, old = record[side].get('core_dynamics'), baseline[side].get('core_dynamics')
        record[side]['retention_vs_fixed48'] = {
            field+'_ratio': _ratio(now[field], old[field]) for field in fields
        } if now is not None and old is not None else None
    record['boundary_readset_changed_vs_fixed48'] = (
        record.get('boundary_readset_sha256') != baseline.get('boundary_readset_sha256')) if (
            record.get('boundary_readset_sha256') and baseline.get('boundary_readset_sha256')) else None
    record['boundary_jets_changed_vs_fixed48'] = (
        record.get('boundary_jets_sha256') != baseline.get('boundary_jets_sha256')) if (
            record.get('boundary_jets_sha256') and baseline.get('boundary_jets_sha256')) else None
    return record
