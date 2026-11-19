"""Effect-first fixed-budget attribution; no production planner is modified."""
import copy
from pathlib import Path
from collections import Counter
import numpy as np
from .common import ROOT, CFG, HERE, binding, fresh, read, write, sha, save_arrays, check_seal
from .population import build, native
from .engine import evaluate_pair, attach_retention
from .policies import feature_row, fit, predict, passes, shortest_screened


def seals():
    rows = []
    for r in read('reports/r62_release/scientific_files.json')['files']:
        p = Path('outputs/duration_r62_01')/r['path']
        if sha(p) != r['sha256']:
            raise ValueError('Frozen R62 scientific file changed: '+str(p))
        rows.append(dict(path=str(p), sha256=r['sha256']))
    # Include all directly imported frozen algorithms and source manifests.
    for directory in ('duration_r62','complete_r36','warp_coupling_r53','bridge_r38',
                      'bridge_audit_r46','manifold_bridge_r47','bridge_dynamics_r48','edit_support_r61'):
        for p in sorted((HERE.parent/directory).glob('*')):
            if p.suffix in ('.py','.json'):
                rows.append(dict(path=str(p),sha256=sha(p)))
    for p in ('outputs/phase_integrity_r60_02/population/result.json',
              'outputs/event_coverage_r47_01/index/result.json',
              'outputs/strong_baseline_r52_01/plans/result.json',
              'outputs/strong_baseline_r52_01/plans/arrays.npz',
              'outputs/validation_r51_01/controls/plans/result.json'):
        rows.append(dict(path=p,sha256=sha(p)))
    return rows


def summarize(cases, records, selections):
    by_key = {r['key']:r for r in records}
    result = {}
    for role in ('DEV','EVAL'):
        group = [c for c in cases if c['role']==role]
        policy_rows = {}
        for name in sorted({s['policy'] for s in selections}):
            rows = [s for s in selections if s['role']==role and s['policy']==name]
            metrics = {}
            for field in ('D1_Q95','D2_Q95','D3_Q95'):
                vals = [by_key[s['record_key']]['methods'][s['method']]['full_dynamics'][field]
                        for s in rows if s['constructible']]
                metrics[field+'_conditional_mean'] = float(np.mean(vals)) if vals else None
            policy_rows[name] = dict(tasks=len(group),selected=len(rows),
                constructible=sum(s['constructible'] for s in rows),
                passes_original_screen=sum(s['accepted'] for s in rows),
                retention_guard_passes=sum(s['retention_guard_passed'] for s in rows),
                duration_counts=dict(Counter(str(s['gap_frames']) for s in rows)),
                logical_screen_trials=sum(len(s['trials']) for s in rows),
                failed_cases=[s['case_key'] for s in rows if not s['accepted']],
                **metrics)
        incidence = Counter(c[side]['source_uid'] for c in group for side in ('left','right') if c[side])
        own_keys={c['key'] for c in group}
        own_records=[r for r in records if r['case_key'] in own_keys]
        result[role] = dict(tasks=len(group),source_incidence=dict(incidence),
            distinct_allocation_recompilations=len(own_records),
            G1_solver_calls=sum(r['status']=='GEOMETRY_EVALUATED' for r in own_records),
            policies=policy_rows,independent_edge_sampling=False,
            scope='FINITE_EXPOSED_SOURCE_DYADS_NOT_INDEPENDENT_EDGE_GENERALIZATION')
    return result


def selected(case, policy, g, cache, trials, prediction=None):
    r = cache.get((case['key'],g))
    method = 'G0' if policy=='Fixed48_G0' else 'G1'
    m = r['methods'].get(method,{}) if r else {}
    old = cache.get((case['key'],48))
    guards = []
    if r and old and m.get('constructible'):
        for side in ('left','right'):
            ratios = r[side].get('retention_vs_fixed48') or {}
            guards.extend([ratios.get('FK_path_length_mean_m_ratio'),ratios.get('FK_motion_range_rms_m_ratio')])
        d3 = old['methods'].get(method,{}).get('full_dynamics',{}).get('D3_Q95')
        guards.append((d3*CFG['retention_guard']['whole_D3_Q95_max_ratio'] >= m['full_dynamics']['D3_Q95'])
                      if d3 is not None else False)
    limits=CFG['retention_guard']
    lower=(limits['core_FK_path_min_ratio'],limits['core_FK_range_min_ratio'])*2
    guard_pass = len(guards)==5 and all(v is not None and v>=t for v,t in zip(guards[:4],lower)) and bool(guards[-1])
    return dict(case_key=case['key'],role=case['role'],policy=policy,gap_frames=g,
        record_key=r['key'] if r else None,method=method,trials=list(trials),prediction=prediction,
        constructible=bool(m.get('constructible')),accepted=bool(m.get('passes_original_screen')),
        retention_guard_passed=bool(guard_pass),fallback_used=False,
        failure='NO_REQUESTED_CANDIDATE' if r is None else (None if m.get('passes_original_screen') else r['status']),
        qualification='NUMERICAL_LOCAL_ONLY_NOT_ARTISTIC_OR_NATIVE24')


def run(out, controls=False, gate=True, render=True, replay_model=None):
    if controls and gate:
        receipt = read(ROOT/'receipts/main_browser.json')
        assert receipt['passed'] and receipt['main_result_sha256']==sha(ROOT/'main/result.json')
    out = fresh(out)
    before = seals()
    cases, lib, J, parents, reference, population = build()
    write(out/'population_seal.json',dict(cases=cases,manifest=population))
    arrays, cache, features = dict(J=J,parents=parents), {}, {}
    def compute(c,g,extra=False):
        k = (c['key'],g)
        if k not in cache or extra and 'G0' not in cache[k]['methods']:
            r,a = evaluate_pair(c,g,lib,J,parents,reference,controls=extra)
            if k in cache:
                # Extra baseline method must leave G1 and both Core arrays exact.
                for name in ('left','right','G1'):
                    key = r['key']+'_'+name
                    if key in a:
                        assert np.array_equal(a[key], arrays[key]), key
            cache[k]=r
            arrays.update(a)
            print('R63',c['role'],c['case'],g,r['status'],flush=True)
        return cache[k]
    dev_rows = []
    for c in [c for c in cases if c['role']=='DEV']:
        base = compute(c,48,controls)
        # Preserve the published strong baseline, including exact E1-W samples.
        assert base['left']['full_core_R_sha256']==c['r62_left_core_sha256']
        assert base['right']['full_core_R_sha256']==c['r62_right_core_sha256']
        rows = [compute(c,g) for g in CFG['durations']]
        row = feature_row(c,base,arrays)
        features[c['key']]=row
        dev_rows.append(dict(row,shortest_gap=shortest_screened(rows)))
    model = fit(dev_rows) if replay_model is None else copy.deepcopy(replay_model)
    if replay_model is not None:
        expected = fit(dev_rows)
        # Arithmetic reconstruction validates the sealed lookup; this is not an
        # optimizer or model-selection call. Production replay uses frozen model.
        assert expected==model
    write(out/'duration_policy.json',model)
    model_sha = sha(out/'duration_policy.json')
    predictions = []
    for c in [c for c in cases if c['role']=='EVAL']:
        base = compute(c,48,controls)
        row = feature_row(c,base,arrays)
        features[c['key']]=row
        pred = predict(model,row['features'])
        predictions.append(dict(case_key=c['key'],**pred))
    # Predictions are sealed before any alternative EVAL length is solved.
    write(out/'evaluation_predictions.json',predictions)
    prediction_sha = sha(out/'evaluation_predictions.json')
    prediction_index = {p['case_key']:p for p in predictions}
    selections = []
    for c in cases:
        base=compute(c,48,controls)
        trials=[]
        for g in CFG['durations']:
            r=compute(c,g);trials.append(g)
            if passes(r):
                break
        sg=shortest_screened([cache[(c['key'],g)] for g in trials])
        # Every duration gets its own recompiled Core retention relative to48.
        for (key,g),r in cache.items():
            if key==c['key']:
                attach_retention(r,base)
        selections.extend([selected(c,'Fixed48',48,cache,[48]),
                           selected(c,'ShortestScreened',sg,cache,trials)])
        if controls:
            pred = prediction_index.get(c['key']) or dict(case_key=c['key'],**predict(model,features[c['key']]['features']))
            g=pred['gap_frames']
            if g is not None:
                attach_retention(compute(c,g),base)
            attach_retention(compute(c,24),base)
            selections.extend([selected(c,'Fixed24',24,cache,[24]),
                               selected(c,'BoundaryConditioned',g,cache,[g] if g else [],pred),
                               selected(c,'Fixed48_G0',48,cache,[48])])
    records=[cache[k] for k in sorted(cache)]
    for r in records:
        attach_retention(r,cache[(r['case_key'],48)])
    if controls:
        saved=read(ROOT/'main/result.json')
        assert cases==saved['cases'] and population==saved['population']
        assert model==read(ROOT/'main/duration_policy.json')
        assert predictions==read(ROOT/'main/evaluation_predictions.json')
        olda=np.load(ROOT/'main/arrays.npz',allow_pickle=False)
        for name in olda.files:
            assert np.array_equal(olda[name],arrays[name]),name
        olda.close()
    result=dict(binding=binding(),protocol=CFG,controls=controls,cases=cases,population=population,
        records=records,selections=selections,features=[features[k] for k in sorted(features)],
        policy_sha256=model_sha,predictions_sha256=prediction_sha,
        policy_fitted_before_alternative_eval_lengths=True,
        policy_label_degenerate=len({r['shortest_gap'] for r in model['fitted_rows']})<=1,
        summary=summarize(cases,records,selections),parent_seal=before,
        parent_files_unchanged=check_seal(before),
        neural_training_steps=0,optimizer_calls=0,boundary_lookup_fit=1,
        planner_runs=0,full_song_runs=0,human_reviewed=False,
        Native24='NOT_EVALUABLE',can_route=False,
        P2='DEFERRED_LOCAL_FIXED_BUDGET_PILOT_ONLY_NO_NEW_FULL_SONG_SEARCH')
    save_arrays(out/'arrays.npz',arrays)
    write(out/'result.json',native(result))
    if render:
        from .viewer import viewer
        viewer(ROOT,controls=controls)
    return result


def normalized(result):
    value=copy.deepcopy(result)
    value['binding']['commit']='PACKAGE_CODE_HASHES_CHECKED'
    return value


def verify():
    out=fresh(ROOT/'independent')
    logs=[]
    model=read(ROOT/'main/duration_policy.json')
    for mode in ('main','controls'):
        saved=ROOT/mode
        old=read(saved/'result.json')
        assert binding()['code']==old['binding']['code']
        rebuilt=out/('rebuild_'+mode)
        result=run(rebuilt,controls=mode=='controls',gate=False,render=False,replay_model=model)
        assert normalized(result)==normalized(old)
        for name in ('arrays.npz','duration_policy.json','evaluation_predictions.json','population_seal.json'):
            assert sha(saved/name)==sha(rebuilt/name),name
        logs.append(dict(mode=mode,scientific_values_exact=True,arrays_bytes_exact=True,
            source_result_sha256=sha(saved/'result.json'),source_commit=old['binding']['commit']))
    inherited=read('outputs/phase_integrity_r60_02/population/result.json')['parent_seal']
    result=dict(binding=binding(),records=logs,frozen_parent_files_unchanged=check_seal(inherited),
        optimizer_calls=0,neural_training_steps=0,production_lookup_refits=0,
        arithmetic_dev_fit_reconstructions=2,fresh_source_recompile_and_geometry=True,can_route=False)
    write(out/'result.json',result)
    print('R63_INDEPENDENT_EXACT',flush=True)
    return result
