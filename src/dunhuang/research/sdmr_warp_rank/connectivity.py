"""Recompute frozen48-frame G1 beside each actual re-timed ChangE Core.

The neighboring source Core is fixed by the saved R64 Fixed48 pair. No R41
score, new neighbor search, budget change or inherited Bridge qualification
enters this calculation. The result is a body22 numerical screen, not Native24
or a statement about artistic quality.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from dunhuang.research.complete_r36.core import read, sha
from dunhuang.research.budget_r63.engine import compile_core
from dunhuang.research.budget_r63.population import build as asset_build, native
from dunhuang.research.duration_r62.cross_source import (
    make_observed, solve_boundary, edge_jets, measure)
from dunhuang.research.warp_coupling_r53.common import array_hash


PARENT = Path('outputs/timing_r64_01')
PARENT_SHA = '48122dea0886791e5fa7ba946d9715cfc2a788314d6d5bc004baa0912bce30ee'
GAP, CORE, CONTEXT = 48, 240, 30


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
        allow_nan=False).encode('utf8')).hexdigest()


def prepare(parentroot=PARENT, *, population=None):
    """Load verified source assets and original R64 pairs without solving G1.

    Pass the already-built R66B population to avoid rebuilding its metadata.
    The R64 scientific JSON is read once here and its bytes must match the
    original seal. No previously saved Bridge is used as a numerical result.
    """
    if population is None:
        from dunhuang.research.rhythm_r66.population_b import build
        population = build(parentroot)
    if population['parent_sha'] != PARENT_SHA:
        raise ValueError('R66B population is not bound to the frozen R64 main')
    path = Path(parentroot)/'main/result.json'
    if sha(path) != PARENT_SHA:
        raise ValueError('Frozen R64 scientific pair identities changed')
    saved = read(path)
    _,lib,J,parents,reference,manifest = asset_build()
    if saved['population']['original_numerical_gate'] != reference:
        raise ValueError('Frozen R64 and actual R63 numerical screens differ')
    original_cases = {c['key']:c for c in saved['cases']}
    records = {r['key']:r for r in saved['records']}
    selections = {(s['case_key'],s['policy']):s for s in saved['selections']}
    registered = {c['key']:deepcopy(c) for c in population['cases']}
    if len(registered) != len(population['cases']) or len(registered) != 12:
        raise ValueError('The twelve frozen R66B Core cases must remain unique')
    pairs = {}
    for case in population['cases']:
        key,side = case['parent_case_key'],case['parent_side']
        if side not in ('left','right'):
            raise ValueError('Original left/right Core identity required')
        parent = original_cases[key]
        selection = selections[(key,'Fixed48')]
        record = records[selection['record_key']]
        if (record['key'] != case['parent_record_key'] or record['gap_frames'] != GAP
                or record['timing_offset_frames'] != 0):
            raise ValueError('Neighbor/budget may not change from original Fixed48')
        for field in ('source_uid','source_interval','target_interval','time_warp'):
            if case[field] != record[side][field]:
                raise ValueError('R66B selected Core differs from frozen pair: '+field)
        if case['baseline_core_R_sha256'] != record[side]['full_core_R_sha256']:
            raise ValueError('Selected Core rotation identity differs from pair')
        if key in pairs:
            continue
        a,b = parent['music_interval']
        if b-a != 528 or record['music_interval'] != [a,b]:
            raise ValueError('Original complete528-frame pair music budget required')
        cores,meta = {},{}
        for name,target in [('left',[a,a+CORE]),('right',[a+CORE+GAP,b])]:
            compiled,R = compile_core(parent[name],target,parent['beats'],lib,J,parents)
            old = record[name]
            if R is None or compiled['status'] != 'LEGAL_CORE' or R.shape != (CORE,22,3,3):
                raise ValueError('Frozen original neighboring Core cannot be recompiled')
            for field in ('source_uid','source_interval','target_interval','time_warp',
                          'full_core_R_sha256','source_R_sha256'):
                if compiled[field] != old[field]:
                    raise ValueError('Original E1-W neighboring Core changed: '+field)
            cores[name],meta[name] = R.copy(),deepcopy(compiled)
        pairs[key] = dict(case=deepcopy(parent),parent_record_key=record['key'],
            left=cores['left'],right=cores['right'],left_meta=meta['left'],right_meta=meta['right'])
    config = Path('src/dunhuang/research/duration_r62/protocol.json')
    bindings = dict(parent_main_sha256=PARENT_SHA, reference_sha256=manifest['reference_sha256'],
        original_numerical_gate=deepcopy(reference),
        duration_protocol_sha256=sha(config), skeleton_J_sha256=array_hash(J),
        skeleton_parents_sha256=array_hash(np.asarray(parents)),
        geometry='ORIGINAL_R62_MAKE_OBSERVED_SOLVE_BOUNDARY_G1_MEASURE_NO_FALLBACK',
        source_catalog_sha256=manifest['catalog_sha256'])
    return dict(lib=lib,J=J,parents=parents,reference=reference,
        change_cases=deepcopy(population['cases']),registered=registered,pairs=pairs,bindings=bindings,
        cache={},counters=dict(check_requests=0,actual_geometry_calls=0,cached_geometry_reuses=0,
            actual_constructed=0,actual_screen_passed=0),preparation_bridge_solves=0)


def check(case_r66b, candidate_R240, ctx):
    """Return ``(receipt, arrays)`` for the fixed original pair's actual G1.

    Arrays use local pair coordinates: ``full_R``528, ``bridge_R``48,
    ``boundary_jets``. Each returned array is a copy, including cache hits.
    Missing construction is a rejection, never a fallback to the saved Bridge.
    """
    registered = ctx['registered'].get(case_r66b['key'])
    if registered is None:
        raise ValueError('Candidate case was not registered before scoring')
    for field in ('parent_case_key','parent_record_key','parent_side','source_uid','event_uid',
                  'source_interval','target_interval','time_warp','baseline_core_R_sha256'):
        if case_r66b[field] != registered[field]:
            raise ValueError('Candidate changed frozen parent Core identity: '+field)
    candidate = np.asarray(candidate_R240)
    if candidate.shape != (CORE,22,3,3) or not np.isfinite(candidate).all():
        raise ValueError('Complete finite240-frame body22 candidate required')
    if not np.allclose(candidate.swapaxes(-1,-2)@candidate,np.eye(3),atol=1e-5,rtol=0) or not np.allclose(np.linalg.det(candidate),1,atol=1e-5,rtol=0):
        raise ValueError('Actual proper SO(3) candidate rotations required')
    side = registered['parent_side']
    pair = ctx['pairs'][registered['parent_case_key']]
    if not np.array_equal(candidate[[0,-1]],pair[side][[0,-1]]):
        raise ValueError('Legal internal Warp must retain the same source endpoints')
    cores = dict(left=pair['left'],right=pair['right'])
    cores[side] = candidate
    L,R = cores['left'],cores['right']
    hashes = {name:array_hash(cores[name]) for name in ('left','right')}
    identity = dict(parent_case_key=registered['parent_case_key'],parent_record_key=registered['parent_record_key'],
        left_R_sha256=hashes['left'],right_R_sha256=hashes['right'],gap_frames=GAP,
        bindings=ctx['bindings'])
    cache_key = _digest(identity)
    ctx['counters']['check_requests'] += 1
    cache_hit = cache_key in ctx['cache']
    if cache_hit:
        ctx['counters']['cached_geometry_reuses'] += 1
        result,arrays = ctx['cache'][cache_key]
    else:
        observed,gap = make_observed(L,R,GAP,context=CONTEXT)
        jets = edge_jets(observed,gap)
        curves,proof = solve_boundary(observed,gap,controls=False)
        ctx['counters']['actual_geometry_calls'] += 1
        curve = curves['G1']
        edge = np.concatenate((L[-4:],R[:4]))
        result = dict(method='G1',gap_frames=GAP,total_pair_frames=528,
            gap_interval_local=[CORE,CORE+GAP],pair_music_interval=pair['case']['music_interval'],
            boundary_readset_sha256=array_hash(edge),boundary_jets_sha256=array_hash(jets),
            boundary_jets=jets.tolist(),boundary_proof=proof,
            actual_boundary_core_frame_indices=dict(left=list(range(236,240)),right=list(range(4))),
            solver_context_frames_per_side=CONTEXT,solver_receives_only_four_edge_poses=True,
            left_core_R_sha256=hashes['left'],right_core_R_sha256=hashes['right'],
            frozen_neighbor_recompiled_exact=True,previous_bridge_qualification_inherited=False,
            fallback_used=False,hidden_motion_truth='UNOBSERVED_DIFFERENT_SOURCE_GAP',
            construction_and_original_screen_separate_from_warp_legality=True,
            Native24='NOT_EVALUABLE',human_reviewed=False,artistic_quality='NOT_EVALUATED',can_route=False)
        arrays = dict(boundary_jets=jets.copy())
        if curve is None:
            result.update(constructible=False,passes_original_screen=False,
                reason='FROZEN_G1_CONSTRUCTION_REJECTED',full_R_sha256=None,core_exact=None)
        else:
            middle = curve[gap[0]:gap[1]]
            Q = np.concatenate((L,middle,R))
            if Q.shape != (528,22,3,3) or not np.isfinite(Q).all():
                raise ValueError('G1 did not produce the complete finite fixed-budget pair')
            if not np.array_equal(Q[:CORE],L) or not np.array_equal(Q[CORE+GAP:],R):
                raise AssertionError('G1 may not edit either complete source Core')
            full_observed = np.concatenate((L,np.full((GAP,22,3,3),np.nan),R))
            measured = measure(Q,full_observed,ctx['J'],ctx['parents'],(CORE,CORE+GAP),ctx['reference'])
            result.update(measured,full_R_sha256=array_hash(Q),bridge_R_sha256=array_hash(middle),core_exact=True)
            arrays.update(full_R=Q.copy(),bridge_R=middle.copy())
            ctx['counters']['actual_constructed'] += 1
            ctx['counters']['actual_screen_passed'] += int(result['passes_original_screen'])
        result = native(result)
        ctx['cache'][cache_key] = (deepcopy(result),{k:v.copy() for k,v in arrays.items()})
    result = deepcopy(result)
    result.update(case_key=case_r66b['key'],parent_case_key=registered['parent_case_key'],
        parent_record_key=registered['parent_record_key'],candidate_side=side,
        candidate_R_sha256=array_hash(candidate),neighbor_R_sha256=hashes['right' if side=='left' else 'left'],
        cache_key=cache_key,cache_hit=cache_hit,actual_geometry_calls_this_request=int(not cache_hit),
        cached_geometry_reuses_this_request=int(cache_hit),cumulative_counts=deepcopy(ctx['counters']),
        numerical_screen_reference=deepcopy(ctx['reference']))
    return result,{k:v.copy() for k,v in arrays.items()}
