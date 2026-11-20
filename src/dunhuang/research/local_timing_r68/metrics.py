"""R68 same-clock, full-stencil diagnostics for local source-clock edits.

Regions describe the registered edit support; they are not semantic phase
truth. Derivatives are computed on the complete Core before masks are applied.
No statistic here is a new physical or perceptual acceptance threshold.
"""
import numpy as np

from dunhuang.research.rhythm_r66 import metrics as original
from dunhuang.research.warp_coupling_r53.common import array_hash

FRAMES, FPS = 240, 30
NUMERICAL_DIFFERENCE_TOLERANCE = 1e-10


def _mask(value, name):
    value = np.asarray(value)
    if value.shape != (FRAMES,) or value.dtype.kind != 'b':
        raise ValueError(name+' must be an actual240-frame boolean mask')
    return value.copy()


def _clock(value, name):
    value = np.asarray(value)
    if (value.shape != (FRAMES,) or value.dtype.kind not in 'fiu'
            or not np.isfinite(value).all() or value[0] < 0
            or not (np.diff(value.astype(float)) > 0).all()):
        raise ValueError(name+' must be240 finite strictly increasing source positions')
    return value


def _frame_byte_changed(a, b):
    """A dtype change is a representation change, not silently cast equality."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError('Matching actual and baseline array shapes required')
    if a.dtype != b.dtype:
        return np.ones(len(a), dtype=bool)
    av = np.ascontiguousarray(a).view(np.uint8).reshape(len(a), -1)
    bv = np.ascontiguousarray(b).view(np.uint8).reshape(len(b), -1)
    return np.any(av != bv, axis=1)


def _ratio(value, reference):
    return float(value/reference) if abs(reference) > 1e-12 else None


def _frame_ranges(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1))]


def _regions(event, support, protected):
    # These four frame masks are mutually exclusive and exhaustive.
    regions = dict(target_event=event & ~protected,
                   compensation=support & ~event & ~protected,
                   protected=protected.copy(),
                   untouched=~(event | support | protected))
    if not np.all(np.stack(list(regions.values())).sum(axis=0) == 1):
        raise AssertionError('Every frame must have exactly one region')
    return regions


def _stencils(regions, actual_edit, declared_edit, order):
    """Index j addresses original frames[j:j+order+1], right endpoint j+order."""
    windows = {k: np.lib.stride_tricks.sliding_window_view(v, order+1)
               for k, v in regions.items()}
    active = np.lib.stride_tricks.sliding_window_view(actual_edit, order+1)
    declared = np.lib.stride_tricks.sliding_window_view(declared_edit, order+1)
    touching = {k: v.any(axis=1) for k, v in windows.items()}
    groups = dict(touch_target=touching['target_event'],
                  touch_compensation=touching['compensation'],
                  touch_edit=active.any(axis=1),
                  touch_declared_edit=declared.any(axis=1),
                  cross_edited_boundary=active.any(axis=1) & ~active.all(axis=1),
                  cross_declared_edit_boundary=declared.any(axis=1) & ~declared.all(axis=1),
                  fully_untouched=~active.any(axis=1),
                  fully_outside_edit_domain=~declared.any(axis=1))
    # Separate from the overlapping questions above, use a deterministic
    # endpoint-independent priority partition so no stencil disappears.
    remaining = np.ones(FRAMES-order, bool); partition = {}
    for name in ('target_event', 'compensation', 'protected', 'untouched'):
        partition[name] = remaining & touching[name]
        remaining &= ~partition[name]
    if remaining.any() or not np.all(np.stack(list(partition.values())).sum(axis=0) == 1):
        raise AssertionError('Every full derivative stencil must have one partition')
    groups.update({'partition_'+name: value for name, value in partition.items()})
    membership = np.stack(list(touching.values())).sum(axis=0)
    groups['cross_frame_region_boundary'] = membership > 1
    return groups


def _activity(P, motion):
    velocity = np.diff(P, axis=0)*FPS
    centred = velocity-velocity.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centred, axis=0))**2
    frequencies = np.fft.rfftfreq(len(velocity), 1/FPS)
    # One-sided real FFT Parseval weights. The fixed6Hz band is a descriptive
    # instrument, not a tuned or physical quality gate.
    weights = np.full(len(frequencies), 2.); weights[0] = 1.
    if len(velocity) % 2 == 0:
        weights[-1] = 1.
    energy = power.sum(axis=(1, 2))*weights
    total, high = float(energy.sum()), float(energy[frequencies >= 6.].sum())
    fk = motion['FK']; so3 = motion['SO3']
    d1 = np.asarray(fk['D1']['curve'])
    rms1 = float(np.sqrt(np.mean(d1*d1)))
    result = dict(FK_velocity_RMS_mps=rms1,
        SO3_speed_mean_radps=so3['D1']['mean'],
        FK_velocity_6Hz_to_Nyquist_power_fraction=(high/total if total > 1e-18 else None),
        velocity_power_sum=total, spectral_band_hz=[6., FPS/2],
        spectral_scope='MEAN_REMOVED_FK_VELOCITY_RECTANGULAR_CORE_WINDOW_WITH_SPECTRAL_LEAKAGE',
        near_static='UNDEFINED_NORMALIZED_RATIOS_ARE_NULL_NOT_QUALITY_PASS',
        quality_pass=None)
    for n in (2, 3):
        values = np.asarray(fk['D'+str(n)]['curve'])
        rms = float(np.sqrt(np.mean(values*values)))
        result['FK_D'+str(n)+'_RMS_over_FPS_power_D1_RMS'] = (
            rms/(FPS**(n-1)*rms1) if rms1 > 1e-12 else None)
    return result


def measure(R, baseline_R, J, parents, q, q0, event_mask, edit_mask,
            support_mask=None, guard_frames=4, include_curves=True):
    """Compare one actual Core to the same240-frame E1-W reference.

    ``edit_mask`` is the declared set of frames the caller may change;
    ``support_mask`` is its registered compensation envelope (defaults to the
    declared set). Actual q/body byte changes are measured independently.
    guard_frames=0 is available only as an explicit unprotected control; it
    never receives the strict-internal label. An explicit Uniform control may
    use this setting, whereas every local intervention uses the default4.
    """
    raw_R, raw_baseline = np.asarray(R), np.asarray(baseline_R)
    R = original._rotations(R, 'actual R', FRAMES)
    baseline = original._rotations(baseline_R, 'baseline R', FRAMES)
    q, q0 = _clock(q, 'q'), _clock(q0, 'q0')
    if q[0] != q0[0] or q[-1] != q0[-1] or q.min() < q0[0] or q.max() > q0[-1]:
        raise ValueError('Same real source interval and endpoints required')
    if (isinstance(guard_frames, bool) or not isinstance(guard_frames, (int, np.integer))
            or not 0 <= guard_frames < FRAMES//2):
        raise ValueError('Integer guard_frames in[0,120) required')
    event = _mask(event_mask, 'event_mask'); declared = _mask(edit_mask, 'edit_mask')
    support = declared.copy() if support_mask is None else _mask(support_mask, 'support_mask')
    protected = np.zeros(FRAMES, bool)
    if guard_frames:
        protected[:guard_frames] = True; protected[-guard_frames:] = True
    if (declared & ~support).any() or ((event | support) & protected).any():
        raise ValueError('Declared edits must be inside support and outside protected edges')
    J, parents = np.asarray(J, float), np.asarray(parents)
    if (J.shape != (22,3) or not np.isfinite(J).all() or parents.shape != (22,)
            or parents.dtype.kind not in 'iu' or parents[0] != -1
            or any(not 0 <= parents[j] < j for j in range(1,22))):
        raise ValueError('Actual common22 skeleton and topological parent contract required')
    clock_bytes = _frame_byte_changed(q,q0)
    body_bytes = _frame_byte_changed(raw_R,raw_baseline)
    q_delta = q.astype(float)-q0.astype(float)
    clock_tol = abs(q_delta) > NUMERICAL_DIFFERENCE_TOLERANCE
    body_max_component = abs(R-baseline).reshape(FRAMES,-1).max(axis=1)
    body_tol = body_max_component > NUMERICAL_DIFFERENCE_TOLERANCE
    actual_edit = clock_bytes | body_bytes
    if (actual_edit & ~declared).any():
        raise ValueError('Actual clock/body edit escaped its declared frame domain')
    if (actual_edit & protected).any():
        raise ValueError('Protected clock and rotation edge bytes must remain exact')
    regions = _regions(event,support,protected)
    P, base_P = original._fk(R,J,parents), original._fk(baseline,J,parents)
    actual, reference = original._motion(R,P,True), original._motion(baseline,base_P,True)
    out = dict(schema='R68_FULL_CORE_STENCIL_ATTRIBUTION_V1',frames=FRAMES,fps=FPS,
        scope='ROOT_RELATIVE_BODY22_SOURCE_RESAMPLING_DIAGNOSTIC_NOT_PHYSICAL_OR_ARTISTIC_QUALIFICATION',
        Native24='NOT_EVALUABLE',artistic_quality='NOT_EVALUABLE',human_reviewed=False,
        semantic_phase_truth='UNKNOWN',can_route=False,quality_pass=None,
        boundary_contract='STRICT_INTERNAL_EXACT_GUARD' if guard_frames else 'EXPLICIT_UNPROTECTED_CONTROL',
        guard_frames=int(guard_frames),
        whole=dict(actual=actual,baseline=reference),
        footprint=dict(numerical_tolerance=NUMERICAL_DIFFERENCE_TOLERANCE,
            tolerance_is_diagnostic_not_a_permission_relaxation=True,
            clock_byte_changed_frames=int(clock_bytes.sum()),body_byte_changed_frames=int(body_bytes.sum()),
            clock_numerical_changed_frames=int(clock_tol.sum()),body_numerical_changed_frames=int(body_tol.sum()),
            actual_union_changed_frames=int(actual_edit.sum()),declared_edit_frames=int(declared.sum()),
            byte_change_ranges=_frame_ranges(actual_edit),
            unchanged_outside_declared_edit_exact=True,protected_q_and_R_bytes_exact=True,
            actual_q_sha256=array_hash(q),baseline_q_sha256=array_hash(q0),
            actual_R_sha256=array_hash(raw_R),baseline_R_sha256=array_hash(raw_baseline),
            guard_q_sha256=array_hash(q[protected]),baseline_guard_q_sha256=array_hash(q0[protected]),
            guard_R_sha256=array_hash(raw_R[protected]),baseline_guard_R_sha256=array_hash(raw_baseline[protected])),
        dose=dict(maximum_abs_source_position_change_frames=float(abs(q_delta).max()),
            mean_abs_source_position_change_frames=float(abs(q_delta).mean()),
            inverse_source_rate_per_target_frame_actual=original._summary(np.diff(q.astype(float))),
            inverse_source_rate_per_target_frame_baseline=original._summary(np.diff(q0.astype(float))),
            maximum_abs_inverse_rate_change=float(abs(np.diff(q.astype(float))-np.diff(q0.astype(float))).max()),
            clock_discrete_second_difference_change=original._summary(np.diff(q_delta,n=2)),
            clock_discrete_third_difference_change=original._summary(np.diff(q_delta,n=3)),
            derivative_scope='ACTUAL_SAMPLED_INVERSE_CLOCK; NOT_CONTINUOUS_TAU_RATE_LEGALITY_CERTIFICATE'),
        regions={name:dict(frame_count=int(mask.sum()),half_open_frame_ranges=_frame_ranges(mask)) for name,mask in regions.items()},
        stencils={},
        derivative_clock='CURVE_INDEX_J_READS_ALL_ORIGINAL_FRAMES[J,J+N+1); OUTPUT_RIGHT_ENDPOINT_J+N',
        partition_definition='TARGET_TOUCH_FIRST_THEN_COMPENSATION_THEN_PROTECTED_THEN_UNTOUCHED; OVERLAP_QUERIES_SEPARATE',
        delta_definition='ACTUAL_DERIVATIVE_RMS_MAGNITUDE_MINUS_BASELINE_AT_SAME_FULL_STENCIL; NOT_DIFFERENCE_OF_P95S')
    for n in (1,2,3):
        name='D'+str(n);groups=_stencils(regions,actual_edit,declared,n)
        result=dict(order=n,full_stencil_count=FRAMES-n,
                    partition_total=sum(int(v.sum()) for k,v in groups.items() if k.startswith('partition_')),
                    groups={})
        for space in ('FK','SO3'):
            av=np.asarray(actual[space][name]['curve']);bv=np.asarray(reference[space][name]['curve'])
            if av.shape != (FRAMES-n,) or bv.shape != av.shape:
                raise AssertionError('Derivative curve has the wrong original frame clock')
            differences=av-bv
            for kind,mask in groups.items():
                summary=result['groups'].setdefault(kind,dict(stencil_count=int(mask.sum()),spaces={}))
                summary['spaces'][space]=dict(actual=original._summary(av[mask]),
                    baseline=original._summary(bv[mask]),signed_magnitude_delta=original._summary(differences[mask]))
                if include_curves:
                    summary['right_endpoint_frames']=np.flatnonzero(mask).astype(int).__add__(n).tolist()
            if include_curves:
                result[space+'_signed_magnitude_delta_curve']=differences.tolist()
        residual=np.sqrt(np.mean((np.diff(P-base_P,n=n,axis=0)*FPS**n)**2,axis=(1,2)))
        result['FK_derivative_residual_RMS']=original._summary(residual)
        if include_curves:
            result['FK_derivative_residual_RMS_curve']=residual.tolist()
        out['stencils'][name]=result
    ga,gb=original._geometry(R,P),original._geometry(baseline,base_P)
    out['geometry_retention']=original._retention(ga,gb)
    out['geometry']=dict(actual=ga,baseline=gb)
    aa,ab=_activity(P,actual),_activity(base_P,reference)
    out['activity_normalized_oscillation']=dict(actual=aa,baseline=ab,
        FK_mean_speed_ratio=_ratio(actual['FK']['D1']['mean'],reference['FK']['D1']['mean']),
        SO3_mean_speed_ratio=_ratio(actual['SO3']['D1']['mean'],reference['SO3']['D1']['mean']),
        interpretation='ACTIVITY_AND_OSCILLATION_DESCRIPTORS; D3_OR_HIGH_BAND_INCREASE_IS_NOT_A_QUALITY_FAILURE')
    if not include_curves:
        for block in (actual,reference):
            for space in ('FK','SO3'):
                for result in block[space].values():result.pop('curve',None)
    return out
