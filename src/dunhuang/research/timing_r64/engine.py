"""Asymmetric local timing candidates using the frozen R63 geometry engine.

The adapter changes only where a fixed528-frame music budget is allocated. It
does not introduce a geometry solver, loss, threshold, source replacement or
cached edge. New candidates call R63 on full original intervals; identical
offset0 candidates can be imported from sealed R63 outputs with exact checks.
"""
from copy import deepcopy
import hashlib
import json

import numpy as np

from dunhuang.research.budget_r63 import engine as frozen


DURATIONS = (24, 36, 48, 60, 72)
OFFSETS = (-24, -12, 0, 12, 24)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+' must be an integer')
    return int(value)


def allocation(g, offset, total=528):
    """Return debts relative to the actual declared240/48/240 baseline."""
    g, offset, total = (_integer(g, 'gap'), _integer(offset, 'offset'),
                        _integer(total, 'total'))
    if g not in DURATIONS or offset not in OFFSETS:
        raise ValueError('Use only the preregistered R64 duration/offset grid')
    if total != 528:
        raise ValueError('R64 local task requires the fixed528-frame budget')
    # Reuse the original positivity/context/budget checks. The shifted baseline
    # is an adapter input only; scientific debt remains relative to240 each side.
    out = frozen.allocation(g, total=total, left=240+offset, right=240-offset,
                            baseline_gap=48)
    out.update(baseline_left_frames=240, baseline_right_frames=240,
               baseline_gap_frames=48, timing_offset_frames=offset,
               left_core_debt_frames=out['left_frames']-240,
               right_core_debt_frames=out['right_frames']-240,
               left_offset_from_symmetric_frames=offset,
               right_offset_from_symmetric_frames=-offset,
               symmetric_left_frames=240+(48-g)//2,
               total_frames=528, budget_exact=True)
    return out


def _cache_payload(record, case):
    sides = {}
    for side in ('left', 'right'):
        meta = record[side]
        sides[side] = {name: meta.get(name) for name in (
            'event_uid', 'source_uid', 'source_interval', 'support_component',
            'target_interval', 'time_warp', 'time_warp_sha256',
            'source_R_sha256', 'source_positions_sha256', 'full_core_R_sha256',
            'status')}
    return dict(schema='R64_FULL_EXECUTION_IDENTITY_V1', sides=sides,
        music_interval=record['music_interval'],
        audio_sha256=case.get('audio_sha256'),
        gap_frames=record['gap_frames'], timing_offset_frames=record['timing_offset_frames'],
        boundary_readset_sha256=record.get('boundary_readset_sha256'),
        boundary_jets_sha256=record.get('boundary_jets_sha256'),
        geometry='FROZEN_R62_G1_VIA_R63_NO_FALLBACK',
        observed_boundary_contract='FOUR_ACTUAL_POSES_PER_SIDE')


def _validate_case(case, g, offset):
    declared = allocation(g, offset)
    if list(case['music_interval'])[1]-list(case['music_interval'])[0] != 528:
        raise ValueError('Case music interval must cover exactly528 frames')
    if (case.get('baseline_left_frames', 240) != 240
            or case.get('baseline_right_frames', 240) != 240
            or case.get('baseline_gap_frames', 48) != 48):
        raise ValueError('R64 declaration must preserve the actual240/48/240 baseline')
    return declared


def _decorate(case, g, offset, rec, reused=False):
    declared = _validate_case(case, g, offset)
    if (rec['allocation']['left_frames'] != declared['left_frames']
            or rec['allocation']['right_frames'] != declared['right_frames']):
        raise AssertionError('Frozen engine and declared allocation disagree')
    rec.update(case_key=case['key'], allocation=declared, timing_offset_frames=offset,
        schema='R64_ASYMMETRIC_LOCAL_CANDIDATE_V1',
        parent_geometry='FROZEN_R63_ENGINE_G1_ONLY',
        declared_baseline=dict(left_frames=240, gap_frames=48, right_frames=240,
                               total_frames=528),
        adapter_baseline=dict(left_frames=240+offset, right_frames=240-offset,
                              purpose='IMPLEMENTATION_ONLY_NOT_DEBT_REFERENCE'),
        source_identity_unchanged=True, source_order_unchanged=True,
        prior_bridge_or_core_reused=bool(reused), execution_cache_lookup=bool(reused),
        independent_e1w_recompile=(not reused and case.get('left') is not None
                                  and case.get('right') is not None))
    if set(rec['methods']) != {'G1'}:
        raise AssertionError('R64 engine may execute only frozen G1')
    for name in ('left', 'right'):
        side = case.get(name)
        if side is not None and (rec[name]['source_uid'] != side['source_uid']
                                or rec[name]['event_uid'] != side['event_uid']
                                or rec[name]['source_interval'] != side['source_interval']):
            raise AssertionError('A timing candidate changed the frozen source identity')
    payload = _cache_payload(rec, case)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode('utf8')).hexdigest()
    rec.update(execution_cache_payload=payload, execution_cache_key=digest,
               execution_cache_key_includes_full_warp_and_jets=True)
    return rec


def evaluate(case, g, offset, lib, J, parents, reference):
    """Independently rebuild one candidate; successful arrays retain exact Cores."""
    g, offset = _integer(g, 'gap'), _integer(offset, 'offset')
    _validate_case(case, g, offset)
    adapted = dict(case, key=case['key']+'_o'+str(offset),
                   baseline_left_frames=240+offset, baseline_right_frames=240-offset)
    # The original implementation recompiles source observations, E1-W, endpoint
    # jets and G1. No previously solved R63/R64 bridge is provided to this call.
    rec, arrays = frozen.evaluate_pair(adapted, g, lib, J, parents, reference,
                                       controls=False)
    return _decorate(case, g, offset, rec), arrays


# Only fields returned by the frozen engine are imported. R63 study-level
# retention annotations and policy labels are recomputed by the R64 study.
_FROZEN_FIELDS = (
    'key', 'case_key', 'gap_frames', 'allocation', 'music_interval', 'gap',
    'target_gap_interval', 'baseline_duration_frames', 'status', 'methods',
    'music_duration_unchanged', 'controls', 'full_source_intervals_retained',
    'training_steps', 'planner_runs', 'hidden_truth_available', 'human_reviewed',
    'verified_phase', 'Native24', 'physical_qualified', 'discarded_or_fallback_warps',
    'left', 'right', 'case_status', 'missing_source_uids',
    'boundary_readset_sha256', 'boundary_jets_sha256', 'boundary_jets',
    'boundary_available_pair_frame_indices', 'boundary_proof',
    'solver_context_frames_per_side', 'solver_receives_only_four_edge_poses',
    'source_boundary_readset')


def from_frozen(case, g, record, arrays):
    """Import only an identical offset0 R63 execution, without geometry calls.

    The caller seals the source JSON and NPZ first. This checks case, source,
    target, interpolation and all available motion hashes. R64 independent
    replay still rebuilds every candidate rather than using this import path.
    """
    g = _integer(g, 'gap')
    declared = _validate_case(case, g, 0)
    if (record.get('case_key') != case['key'] or record.get('gap_frames') != g
            or record.get('music_interval') != case['music_interval']):
        raise ValueError('Frozen case, gap or music interval differs')
    old_key, new_key = record['key'], case['key']+'_o0_g'+str(g)
    if old_key != case['key']+'_g'+str(g):
        raise ValueError('Frozen R63 record key differs from declared case/gap')
    for key in ('left_frames', 'right_frames', 'gap_frames', 'total_frames'):
        if record.get('allocation', {}).get(key) != declared[key]:
            raise ValueError('Frozen symmetric allocation differs')
    A, B = case['music_interval']
    nl, nr = declared['left_frames'], declared['right_frames']
    if record.get('gap') != [nl, nl+g] or record.get('target_gap_interval') != [A+nl, A+nl+g]:
        raise ValueError('Frozen actual gap target differs')
    rec = {key: deepcopy(record[key]) for key in _FROZEN_FIELDS if key in record}
    rec.update(key=new_key, controls=False)
    if 'G1' not in rec.get('methods', {}):
        raise ValueError('Frozen G1 execution missing')
    rec['methods'] = {'G1': rec['methods']['G1']}
    if 'boundary_proof' in rec:
        proof = rec['boundary_proof']
        proof['G0_actual_boundary_read_indices'] = None
        proof['guards'] = {'G1': proof['guards']['G1']}
    output, cores = {}, {}
    targets = dict(left=[A, A+nl], right=[A+nl+g, B])
    for side in ('left', 'right'):
        meta, source = rec[side], case.get(side)
        meta.pop('retention_vs_fixed48', None)
        if source is None:
            if meta.get('status') != 'SOURCE_MISSING' or meta.get('source_interval') is not None:
                raise ValueError('Frozen missing-source status differs')
            expected_uid = case.get(side+'_source_uid')
            if expected_uid is not None and meta.get('source_uid') != expected_uid:
                raise ValueError('Frozen missing-source UID differs')
        else:
            for field in ('source_uid', 'event_uid', 'source_interval', 'support_component'):
                if meta.get(field) != source.get(field):
                    raise ValueError('Frozen '+side+' source identity/support differs')
            if 'phase_inputs' in meta and meta['phase_inputs'] != frozen._phases(source.get('phases', [])):
                raise ValueError('Frozen source phase proxy inputs differ')
            if meta.get('status') != 'PAIR_SOURCE_MISSING' and meta.get('target_interval') != targets[side]:
                raise ValueError('Frozen '+side+' target interval differs')
        old_array = old_key+'_'+side
        if meta.get('status') == 'LEGAL_CORE':
            if old_array not in arrays:
                raise ValueError('Frozen legal Core array missing')
            core = np.asarray(arrays[old_array])
            expected_frames = nl if side == 'left' else nr
            if core.shape != (expected_frames, 22, 3, 3) or not np.isfinite(core).all():
                raise ValueError('Frozen Core shape/finite contract differs')
            if frozen.array_hash(core) != meta['full_core_R_sha256']:
                raise ValueError('Frozen Core array hash mismatch')
            w = frozen.validate(meta['time_warp'], meta['source_interval'], targets[side])
            q = np.interp(np.arange(*targets[side]), w[:, 1], w[:, 0])
            if (not np.array_equal(q, meta['source_positions'])
                    or frozen.array_hash(q) != meta['source_positions_sha256']
                    or frozen.array_hash(w) != meta['time_warp_sha256']):
                raise ValueError('Frozen actual source clock/hash differs')
            lo, hi = meta['support_component']
            if np.floor(q).min() < lo or np.ceil(q).max() >= hi:
                raise ValueError('Frozen interpolation leaves accepted support')
            output[new_key+'_'+side] = core.copy()
            cores[side] = core
        elif old_array in arrays:
            raise ValueError('Frozen rejected Core unexpectedly has an array')
    if len(cores) == 2:
        edge = np.concatenate((cores['left'][-4:], cores['right'][:4]))
        observed, gap = frozen.make_observed(cores['left'], cores['right'], g,
                                             context=frozen.DISPLAY_CONTEXT)
        jets = frozen.edge_jets(observed, gap)
        if (frozen.array_hash(edge) != rec.get('boundary_readset_sha256')
                or frozen.array_hash(jets) != rec.get('boundary_jets_sha256')
                or not np.array_equal(jets, rec.get('boundary_jets'))):
            raise ValueError('Frozen four-boundary readset or jets hash mismatch')
    method = rec['methods']['G1']
    if method.get('constructible', False):
        old_array = method.get('arrays_key')
        if old_array != old_key+'_G1' or old_array not in arrays or len(cores) != 2:
            raise ValueError('Frozen full G1 array missing or keyed incorrectly')
        curve = np.asarray(arrays[old_array])
        if (curve.shape != (528, 22, 3, 3) or not np.isfinite(curve).all()
                or frozen.array_hash(curve) != method['full_R_sha256']
                or not np.array_equal(curve[:nl], cores['left'])
                or not np.array_equal(curve[nl+g:], cores['right'])):
            raise ValueError('Frozen full trajectory hash or exact-Core splice mismatch')
        method['arrays_key'] = new_key+'_G1'
        output[new_key+'_G1'] = curve.copy()
    elif method.get('arrays_key') is not None or old_key+'_G1' in arrays:
        raise ValueError('Frozen rejected G1 must not contain a trajectory')
    return _decorate(case, g, 0, rec, reused=True), output
