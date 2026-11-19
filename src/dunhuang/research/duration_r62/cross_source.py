"""Frozen R52 Core identities and warps, with independently varied bridge time.

These are different-source transitions, not held-out true motion recovery. Core
motion remains byte-identical; additional gap frames add time rather than borrow
it from a music budget. No automatic duration selector is qualified here.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .common import ROOT, CFG, binding, fresh, read, write, sha, relative_fk, save_arrays, array_hash
from dunhuang.research.complete_r36.core import Library, rotation_sample, identity
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.bridge_audit_r46.admission import certificate
from dunhuang.research.manifold_bridge_r47.geometry import segmented, endpoint_jets
from dunhuang.research.bridge_dynamics_r48.study import metrics as original_metrics
from dunhuang.research.edit_support_r61.fixed_gap import derivatives

R51 = Path('outputs/validation_r51_01/controls/plans/result.json')
R52 = Path('outputs/strong_baseline_r52_01/plans')
R47 = Path('outputs/event_coverage_r47_01/index/result.json')
DISPLAY_CONTEXT = int(CFG['display_context'])
DURATIONS = tuple(CFG['durations'])


def core_slice(R, interval, music_interval):
    """Never confuse absolute music frames with local concatenated R52 frames."""
    start, end = map(int, music_interval)
    a, b = map(int, interval)
    if not (len(R) == end-start and start <= a < b <= end):
        raise ValueError('R52 music origin or saved concatenated length mismatch')
    return np.asarray(R[a-start:b-start]).copy()


def make_observed(left, right, frames, context=DISPLAY_CONTEXT):
    if not isinstance(frames, (int, np.integer)) or isinstance(frames, (bool, np.bool_)) or frames <= 0:
        raise ValueError('Positive integer transition duration required')
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 4 or right.ndim != 4 or left.shape[1:] != (22,3,3) or right.shape[1:] != (22,3,3):
        raise ValueError('Observed body22 core rotations required')
    if len(left) < context or len(right) < context or context < 4:
        raise ValueError('Both genuine core boundary contexts must exist')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('A fixed Core may not contain invented or unknown poses')
    R = np.full((2*context+frames,22,3,3), np.nan, dtype=np.float64)
    R[:context] = left[-context:]
    R[context+frames:] = right[:context]
    return R, (context,context+frames)


def solve_boundary(observed, gap, controls=False):
    """Only four poses per boundary are available to either frozen solver."""
    raw = np.asarray(observed)
    a,b = gap
    if raw.ndim != 4 or raw.shape[1:] != (22,3,3) or not 4 <= a < b <= len(raw)-4:
        raise ValueError('Four actual edge observations per side required')
    ids = np.r_[a-4:a,b:b+4]
    if not np.isfinite(raw[ids]).all():
        raise ValueError('Missing boundary observations')
    restricted = np.full(raw.shape,np.nan,dtype=np.float64)
    restricted[ids] = raw[ids]
    curves = {}
    Q, guard = segmented(restricted,gap,tuple(CFG['G1_counts']),CFG['chart_radius_rad'])
    curves['G1'] = Q
    guards = dict(G1=guard)
    if controls:
        cert = certificate(restricted,gap)
        guards['G0'] = cert
        curves['G0'] = observed_bridge(restricted,gap) if cert['chart_eligible'] else None
    # Restore every genuine visible context frame, including those the solver
    # never received. A poison/hidden value is never copied into the gap.
    visible = np.r_[0:a,b:len(raw)]
    for name,Q in curves.items():
        if Q is None:
            continue
        Q[visible] = raw[visible]
        assert np.array_equal(Q[visible],raw[visible])
        assert np.isfinite(Q).all()
    proof = dict(available_boundary_frame_indices=ids.tolist(),
                 G1_actual_boundary_read_indices=np.r_[a-3:a,b:b+3].tolist(),
                 G0_actual_boundary_read_indices=np.r_[a-3:a,b:b+3].tolist() if controls else None,
                 hidden_read=False,phase_labels_used=False,rejection_fallback=False,
                 guards=guards)
    return curves,proof


def edge_jets(R,gap):
    return np.stack([np.stack(endpoint_jets(R[:,j],gap)) for j in range(22)])


def boundary_evidence(positions,source):
    """Record observed source evidence near edges without extending it into gap."""
    positions=np.asarray(positions,float)
    support=sorted({int(t) for t in np.r_[np.floor(positions),np.ceil(positions)]})
    result=dict(source_positions=positions.tolist(),interpolation_read_support=support,
                verified_phase='UNKNOWN',transition_source_evidence='UNOBSERVED_DIFFERENT_SOURCE_GAP')
    for name,spans in [('raw',[p['evidence_span'] for p in source['phase_evidence']]),
                       ('band',source['protected_spans'])]:
        observed=[t for t in support if any(a<=t<b for a,b in spans)]
        distances=[0. if a<=t<=b-1 else min(abs(t-a),abs(t-(b-1))) for t in positions for a,b in spans]
        result[name+'_observed_read_frames']=observed
        result[name+'_minimum_distance_source_frames']=float(min(distances)) if distances else None
    return result


def measure(Q,observed,J,parents,gap,reference):
    """Numerical motion diagnostics, without fictitious hidden truth errors."""
    a,b = gap
    m = original_metrics(Q,J,parents,gap)
    m.pop('jerk_curve',None)
    P = relative_fk(Q,J,parents)
    delta = Q[:-1].swapaxes(-1,-2)@Q[1:]
    omega,alpha = derivatives(Q)
    rate_increment = (delta[:-1]@omega[1:,...,None])[...,0]-omega[:-1]
    accel_increment = (delta[:-2]@alpha[1:,...,None])[...,0]-alpha[:-1]
    # Time indices refer to observed/generated velocity stencils, not a new
    # claim of C1/C2 equality on sampled FK positions.
    speed_change = np.linalg.norm(rate_increment[[a-2,b-1]],axis=-1).mean()
    acceleration_change = np.linalg.norm(accel_increment[[a-3,b-1]],axis=-1).mean()
    path = np.linalg.norm(np.diff(P[a-1:b+1],axis=0),axis=-1).sum(0)
    chord = np.linalg.norm(P[b]-P[a-1],axis=-1)
    gap_speed = np.linalg.norm(omega[a-1:b],axis=-1).mean()
    side_rates = np.concatenate((omega[a-4:a-1],omega[b:b+3]))
    side_speed = np.linalg.norm(side_rates,axis=-1).mean()
    passed = (m['gap_D2'] <= reference['D2'] and m['gap_D3'] <= reference['D3']
              and m['shoulder_wrist_min_m'] >= reference['shoulder_wrist_lower_m'])
    return dict(constructible=True,passes_original_screen=bool(passed),metrics=m,
                transported_boundary_velocity_increment_radps=float(speed_change),
                transported_boundary_acceleration_increment_radps2=float(acceleration_change),
                transported_rate_increment_indices=[a-2,b-1],
                transported_acceleration_increment_indices=[a-3,b-1],
                FK_path_length_mean_m=float(path.mean()),FK_endpoint_chord_mean_m=float(chord.mean()),
                FK_path_to_endpoint_chord_ratio=float(path.mean()/chord.mean()) if chord.mean()>1e-9 else None,
                FK_motion_range_rms_m=float(np.sqrt(np.mean(np.var(P[a:b],axis=0)))),
                gap_mean_joint_speed_radps=float(gap_speed),adjacent_core_mean_joint_speed_radps=float(side_speed),
                gap_to_adjacent_core_speed_ratio=float(gap_speed/side_speed) if side_speed>1e-9 else None,
                hidden_SO3_error_rad=None,hidden_FK_error_m=None,
                evaluation_scope='NO_HIDDEN_TRUTH_CROSS_SOURCE_NUMERICAL_DIAGNOSTICS_NOT_NATIVE24')


def build_roster(limit=12):
    """First frozen 48-frame-plan pairs by identity, never by R62 outcomes."""
    old = read(R51)
    r52 = read(R52/'result.json')
    r52_index = {r['key']:r for r in r52['records']}
    config = read('src/dunhuang/research/complete_r36/config.json')
    assert sha(config['catalog']) == config['catalog_sha256']
    lib = Library(config['catalog'])
    units = {u['unit_uid']:u for u in read(R47)['units']}
    sources = {s['source_uid']:s for s in read('outputs/phase_integrity_r60_02/population/result.json')['sources']}
    roster,seen = [],set()
    with np.load(R52/'arrays.npz') as z:
        J,parents = z['J'].copy(),z['parents'].copy()
        for row in sorted(old['records'],key=lambda r:r['key']):
            if row['gap'] != CFG['cross_original_gap']:
                continue
            chosen = row['selected']['HeuristicFullPool']
            frozen = r52_index[row['key']]['selected']['Heuristic_repeat0_same0_balanced0']
            assert chosen['events'] == frozen['events']
            prefix = row['key']+'_Heuristic_repeat0_same0_balanced0_R'
            compiled = z[prefix]
            spans = [r['interval'] for r in row['regions'] if r['kind']=='core']
            assert len(spans) == len(chosen['events'])
            cores,meta = [],[]
            for event,target in zip(chosen['events'],spans):
                R = core_slice(compiled,target,row['music_interval'])
                uid = event['source_uid'];w = np.asarray(event['warp']['time_warp'],float)
                q = np.interp(np.arange(*target),w[:,1],w[:,0])
                assert np.all(np.diff(q)>0)
                unit = units[event['event_uid']]
                lo,hi = unit['support_component']
                assert lo <= q.min() and q.max() <= hi-1
                rebuilt = rotation_sample(lib.rotations[uid],q)
                assert np.array_equal(R,rebuilt), (row['key'],event['event_uid'])
                cores.append(R)
                meta.append(dict(event_uid=event['event_uid'],source_uid=uid,
                    source_interval=event['source_interval'],target_interval=target,
                    time_warp=event['warp']['time_warp'],warp=event['warp'],
                    support_component=unit['support_component'],native_reference=lib.rows[uid][0]['reference'],
                    source_R_sha256=array_hash(lib.rotations[uid]),full_core_R_sha256=array_hash(R),
                    full_core_frames=len(R),independent_source_recompile_exact=True,
                    boundary_source_positions_left=q[:4].tolist(),boundary_source_positions_right=q[-4:].tolist(),
                    boundary_evidence_left=boundary_evidence(q[:4],sources[uid]),
                    boundary_evidence_right=boundary_evidence(q[-4:],sources[uid])))
            for edge,(left,right) in enumerate(zip(meta,meta[1:])):
                if left['source_uid'] == right['source_uid']:
                    continue
                # Full identities and the actual four-pose read set both enter
                # uniqueness. A different target warp is not a UID cache hit.
                pair_identity = dict(left=left,right=right,
                    readset_hash=array_hash(np.concatenate((cores[edge][-4:],cores[edge+1][:4]))))
                key = identity(pair_identity)
                if key in seen:
                    continue
                seen.add(key)
                roster.append(dict(case=len(roster),pair_uid=key,frozen_plan_key=row['key'],
                    frozen_edge_index=edge,frozen_plan_music_interval=row['music_interval'],
                    frozen_gap_frames=48,left=left,right=right,
                    left_core=cores[edge],right_core=cores[edge+1]))
                if len(roster)>=limit:
                    break
            if len(roster)>=limit:
                break
    evidence = dict(requested_maximum_pairs=limit,selected_pairs=len(roster),
        restricted_to_frozen_gap_frames=48,shortfall_retained=len(roster)<limit,
        selection='FIRST_UNIQUE_FULL_EVENT_WARP_AND_EDGE_READSET_BY_SORTED_FROZEN_R51_KEY_EDGE; NO_R62_OUTCOMES',
        frozen_R51_sha256=sha(R51),frozen_R52_result_sha256=sha(R52/'result.json'),
        frozen_R52_arrays_sha256=sha(R52/'arrays.npz'),R47_units_sha256=sha(R47),
        catalog_sha256=config['catalog_sha256'])
    return roster,J,parents,evidence


def duration_summary(records):
    """Descriptive fixed-pair response; oracle choice is never a new policy."""
    by_pair = {}
    for row in records:
        by_pair.setdefault(row['pair_case'],[]).append(row)
    response = []
    for case,rows in sorted(by_pair.items()):
        rows = sorted(rows,key=lambda r:r['gap_frames'])
        successful = [r['gap_frames'] for r in rows
                      if r['methods'].get('G1',{}).get('passes_original_screen',False)]
        response.append(dict(pair_case=case,passing_G1_durations=successful,
            minimum_passing_duration=min(successful) if successful else None,
            duration_pass_pattern={str(r['gap_frames']):r['methods'].get('G1',{}).get('passes_original_screen',False) for r in rows},
            posthoc_scope='OBSERVED_ORACLE_RESPONSE_NOT_TRAINED_ADAPTIVE_DECISION_OR_FIXED_MUSIC_BUDGET'))
    return dict(by_duration={str(g):dict(tasks=sum(r['gap_frames']==g for r in records),
        methods={method:dict(constructed=sum(r['gap_frames']==g and r['methods'].get(method,{}).get('constructible',False) for r in records),
                            passed_original_screen=sum(r['gap_frames']==g and r['methods'].get(method,{}).get('passes_original_screen',False) for r in records))
                 for method in sorted({k for r in records for k in r['methods']})}) for g in DURATIONS},
        pair_responses=response,pass_patterns_differ_across_pairs=len({tuple(x['passing_G1_durations']) for x in response})>1,
        adaptive_policy_qualified=False,P2='NOT_ADMITTED_NO_OUT_OF_SAMPLE_DURATION_DECISION_NET_GAIN')


def run(out,controls=False,main=None):
    out = fresh(out)
    roster,J,parents,selection = build_roster(int(CFG['cross_max_pairs']))
    reference = read('outputs/bridge_dynamics_r48_01/main/result.json')['reference']
    arrays = dict(J=J,parents=parents)
    records = []
    for pair in roster:
        pair_case = pair['case'];left,right = pair['left_core'],pair['right_core']
        arrays['pair%d_left_core'%pair_case] = left
        arrays['pair%d_right_core'%pair_case] = right
        fixed_edges = np.concatenate((left[-4:],right[:4]))
        invariant_jets = None
        for frames in DURATIONS:
            raw,gap = make_observed(left,right,frames)
            curves,proof = solve_boundary(raw,gap,controls=controls)
            current_jets = edge_jets(raw,gap)
            if invariant_jets is None:
                invariant_jets = current_jets
            assert np.array_equal(current_jets,invariant_jets)
            assert np.array_equal(np.concatenate((raw[gap[0]-4:gap[0]],raw[gap[1]:gap[1]+4])),fixed_edges)
            key='cross%d_d%d'%(pair_case,frames)
            methods = {}
            for name,Q in curves.items():
                if Q is None:
                    methods[name]=dict(constructible=False,passes_original_screen=False,arrays_key=None,
                        reason='FROZEN_METHOD_CHART_OR_JET_REJECTION_NO_FALLBACK')
                else:
                    methods[name]=measure(Q,raw,J,parents,gap,reference)
                    methods[name]['arrays_key']=key+'_'+name
                    arrays[key+'_'+name]=Q
                    assert np.array_equal(Q[:DISPLAY_CONTEXT],left[-DISPLAY_CONTEXT:])
                    assert np.array_equal(Q[-DISPLAY_CONTEXT:],right[:DISPLAY_CONTEXT])
            record=dict(case=pair_case,pair_case=pair_case,key=key,gap_frames=frames,
                source_uid=pair['left']['source_uid']+' -> '+pair['right']['source_uid'],dataset='ChangE',
                pair_uid=pair['pair_uid'],frozen_plan_key=pair['frozen_plan_key'],frozen_edge_index=pair['frozen_edge_index'],
                left=pair['left'],right=pair['right'],display_context=DISPLAY_CONTEXT,frames=len(raw),gap=list(gap),
                edit_target_interval=list(gap),retained_left_interval=[0,DISPLAY_CONTEXT],
                retained_right_interval=[DISPLAY_CONTEXT+frames,len(raw)],retained_core_edit_overlap_frames=0,
                full_core_hashes_unchanged=True,boundary_poses_and_jets_identical_across_durations=True,
                available_boundary_readset_sha256=array_hash(fixed_edges),observed_jets_sha256=array_hash(current_jets),
                source_clock='DIFFERENT_SOURCE_CLOCKS_NO_SINGLE_HIDDEN_TRUTH',
                transition_source_support=None,transition_phase_truth='UNKNOWN',
                reference_Warp='FROZEN_R52_SAME_CORE_AND_TIME_MAP_FOR_ALL_GAPS',
                local_total_frames=len(raw),fixed_full_pair_total_frames=len(left)+frames+len(right),
                fixed_music_budget=False,total_time_policy='ADDITIONAL_TRANSITION_TIME_NOT_BORROWED_FROM_FIXED_CORES',
                methods=methods,solver=proof,human_reviewed=False,world_root_contact='UNKNOWN',Native24='NOT_EVALUABLE',
                status='CROSS_SOURCE_DURATION_EVALUATED',can_route=False)
            records.append(record)
            print('R62_CROSS',key,'G1',methods['G1']['constructible'],flush=True)
    metadata=[{k:v for k,v in p.items() if k not in ('left_core','right_core')} for p in roster]
    result=dict(binding=binding(),records=records,roster=metadata,population=selection,
        reference=reference,controls=controls,summary=duration_summary(records),
        algorithms={p:sha(p) for p in ('src/dunhuang/research/bridge_r38/geometry.py',
                    'src/dunhuang/research/bridge_audit_r46/admission.py','src/dunhuang/research/manifold_bridge_r47/geometry.py')},
        source_independence='HISTORICALLY_EXPOSED_FROZEN_HEURISTIC_SELECTED_PAIRS_NOT_UNSEEN_GENERALIZATION',
        training_steps=0,optimizer_calls=0,planner_runs=0,P2='NOT_RUN',can_route=False)
    if controls:
        parent = Path(main) if main is not None else ROOT/'main/cross_source'
        old = read(parent/'result.json')
        assert old['roster']==metadata and old['population']==selection
        lookup = {r['key']:r for r in old['records']}
        with np.load(parent/'arrays.npz') as prior:
            for r in records:
                expected=lookup[r['key']]
                assert r['methods']['G1']==expected['methods']['G1']
                key=r['methods']['G1']['arrays_key']
                if key is not None:
                    assert np.array_equal(arrays[key],prior[key])
        result['main_G1_exact']=True
        result['main_result_sha256']=sha(parent/'result.json')
    save_arrays(out/'arrays.npz',arrays)
    write(out/'result.json',result)
    return result
