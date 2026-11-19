"""R61 centers, new gap durations; paired method effects only within a duration."""
import inspect
import numpy as np
from .common import ROOT,CFG,binding,fresh,read,write,sha,save_arrays
from .methods import solve,measure,absent,angles
from dunhuang.research.edit_support_r61.intervention import population
from dunhuang.research.edit_support_r61.mask_audit import audit_mask
from dunhuang.research.manifold_bridge_r47.geometry import segmented


def source_balanced(rows, value):
    groups={}
    for row in rows:
        item=row.get(value)
        if item is not None:groups.setdefault(row['source_uid'],[]).append(item)
    v=np.array([np.mean(groups[k]) for k in sorted(groups)],dtype=float)
    if not len(v):return dict(sources=0,tasks=0,mean=None,ci95=None)
    rng=np.random.default_rng(CFG.get('seed',62062))
    draws=v[rng.integers(len(v),size=(CFG.get('bootstrap_draws',3000),len(v)))].mean(1)
    return dict(sources=len(v),tasks=sum(len(x) for x in groups.values()),mean=float(v.mean()),
        ci95=[float(x) for x in np.quantile(draws,[.025,.975])],
        scope='RECORDING_BALANCED_SAME_DURATION_PAIRED_METHOD_EFFECT; NOT_LENGTH_CAUSAL_EFFECT')


def interval(center,gap,component):
    if isinstance(gap,(bool,np.bool_)) or not isinstance(gap,(int,np.integer)) or gap<=0 or gap%2:
        raise ValueError('Even positive integer gap required by the frozen centered task')
    start,end=int(center)-int(gap)//2-4,int(center)+int(gap)//2+4
    return dict(source_interval=[start,end],edit_source_interval=[start+4,end-4],
                eligible=bool(component[0]<=start<end<=component[1]))


def _phase_band(phase,n):
    return [[max(0,phase['frame']-12),min(n,phase['frame']+13)]] if phase['type']=='speed_peak' else [phase['evidence_span']]


def summarize(records,controls):
    result={}
    durations=CFG.get('durations',[24,36,48,60,72])
    names=('G1','SLERP','SingleChartHermite') if controls else ('G1',)
    for gap in durations:
        datasets={}
        for dataset in ('ALL','ChangE','FineDance','AIST++'):
            registered=[r for r in records if r['gap']==gap and (dataset=='ALL' or r['dataset']==dataset)]
            rows=[r for r in registered if r['status']=='CONTINUOUS_GAP_EVALUATED']
            paired={}
            if controls:
                for baseline in ('SLERP','SingleChartHermite'):
                    for metric in ('error_rad','fk_error_m'):
                        key='G1_minus_'+baseline+'_'+metric
                        paired[key]=source_balanced(rows,key)
            datasets[dataset]=dict(registered=len(registered),eligible=len(rows),not_eligible=len(registered)-len(rows),
                sources=len({r['source_uid'] for r in rows}),
                methods={name:dict(constructed=sum(r['methods'][name]['constructible'] for r in rows),
                    passed_original_screen=sum(r['methods'][name]['passes_original_screen'] for r in rows),
                    rejection_count=sum(not r['methods'][name]['constructible'] for r in rows)) for name in names},
                paired=paired,interpretation='WITHIN_GAP_METHOD_COMPARISON; HIDDEN_TARGET_CHANGES_ACROSS_GAPS')
        result[str(gap)]=datasets
    return result


def run(out,controls=False,gate=True):
    main_path=ROOT/'main/continuous/result.json'
    if controls and gate:
        receipt=read(ROOT/'receipts/main_browser.json')
        if not receipt['passed'] or receipt['main_result_sha256']!=sha(ROOT/'main/result.json'):
            raise ValueError('Main actual-source viewer must be accepted before method controls')
    out=fresh(out);assets,parent,fresh_cases=population()
    frozen_path=ROOT.parent/'edit_support_r61_01/main/result.json';frozen=read(frozen_path)
    R61_fixed_path=ROOT.parent/'edit_support_r61_01/fixed_gap/result.json'
    R61_fixed=read(R61_fixed_path)
    R61_index={r['case']:r for r in R61_fixed['records']}
    index={c['case']:c for c in fresh_cases}
    arrays=dict(J=assets.J,parents=assets.parents);records=[]
    durations=CFG.get('durations',[24,36,48,60,72])
    if tuple(durations)!=(24,36,48,60,72):raise ValueError('R62 fixed duration population changed')
    main_rows={r['key']:r for r in read(main_path)['records']} if controls else None
    main_arrays=np.load(ROOT/'main/continuous/arrays.npz') if controls else None
    R61_arrays=np.load(ROOT.parent/'edit_support_r61_01/fixed_gap/arrays.npz')
    difficulty={}
    for c in frozen['cases']:
        q=c['selection']
        if q['status']!='MATCHED':continue
        iv=interval(int(q['evidence']['frame']),48,q['component'])
        if iv['eligible']:
            a,b=iv['source_interval'];R=assets.R[c['source_uid']]
            difficulty[c['case']]=float(angles(R[a+3],R[b-4]).mean())
    median=float(np.median(list(difficulty.values()))) if difficulty else None
    labels={k:('lower_or_equal_median' if value<=median else 'above_median') for k,value in difficulty.items()}
    parity_rows=[]
    try:
        for case in frozen['cases']:
            original=index[case['case']]
            assert case['source_uid']==original['source_uid'] and case['selection']==original['selection']
            uid=case['source_uid'];source=assets.sources[uid]
            for gap in durations:
                key='case%d_d%d'%(case['case'],gap)
                base=dict(case=case['case'],key=key,source_uid=uid,dataset=case['dataset'],role=case['role'],gap=int(gap),
                    task='CONTINUOUS_SOURCE_RECONSTRUCTION',source_R_sha256=case['source_R_sha256'],
                    human_reviewed=False,verified_phase='UNKNOWN',world_root_contact='UNKNOWN',
                    motion_difficulty=labels.get(case['case'],'UNAVAILABLE_BASELINE48_SUPPORT'),
                    baseline48_endpoint_mean_angle_rad=difficulty.get(case['case']))
                selected=case['selection']
                if selected['status']!='MATCHED':
                    records.append(dict(base,status='NOT_MATCHABLE_R61_SOURCE',selection=selected));continue
                phase=selected['evidence'];center=int(phase['frame']);iv=interval(center,gap,selected['component'])
                base.update(source_interval=iv['source_interval'],edit_source_interval=iv['edit_source_interval'],
                    source_component=selected['component'],target_phase=phase,center=center)
                if not iv['eligible']:
                    records.append(dict(base,status='NOT_CONSTRUCTIBLE_REAL_CONTEXT',reason='FROZEN_CENTER_OUTSIDE_ACCEPTED_SOURCE_SUPPORT_NO_REPLACEMENT'));continue
                start,end=iv['source_interval'];truth=assets.R[uid][start:end].copy()
                assert len(truth)==gap+8
                observed=np.full_like(truth,np.nan);observed[:4]=truth[:4];observed[-4:]=truth[-4:]
                curves,proof=solve(observed,gap,controls=controls);arrays[key+'_truth']=truth;methods={}
                for name,Q in curves.items():
                    if Q is None:methods[name]=absent('FROZEN_CHART_OR_JET_REJECTION_NO_FALLBACK')
                    else:
                        arrays[key+'_'+name]=Q
                        methods[name]=dict(measure(Q,truth,assets.J,assets.parents,assets.reference,gap),arrays_key=key+'_'+name)
                if controls:
                    previous=main_rows[key]
                    assert methods['G1']==previous['methods']['G1']
                    if methods['G1']['constructible']:
                        assert np.array_equal(arrays[key+'_G1'],main_arrays[key+'_G1'])
                positions=np.arange(start,end,dtype=float);edit=np.zeros(gap+8,bool);edit[4:gap+4]=True
                target=audit_mask(positions,edit,[phase['evidence_span']],_phase_band(phase,len(assets.R[uid])),[center])
                all_evidence=audit_mask(positions,edit,[p['evidence_span'] for p in source['phase_evidence']],source['protected_spans'],
                    sorted({int(p['frame']) for p in source['phase_evidence']}))
                row=dict(base,status='CONTINUOUS_GAP_EVALUATED',frames=gap+8,origin=start,
                    target_interval=[0,gap+8],edit_target_interval=[4,gap+4],truth_arrays_key=key+'_truth',
                    observed_source_intervals=[[start,start+4],[end-4,end]],
                    source_to_target_clock='IDENTITY_ORIGINAL_30FPS_MINUS_SOURCE_START',
                    methods=methods,solver=proof,target_phase_audit=target,all_evidence_audit=all_evidence,
                    truth_scope='EXACT_SAME_HIDDEN_TARGET_WITHIN_THIS_GAP; DIFFERENT_REAL_TARGET_ACROSS_GAPS',
                    mask_scope='RECONSTRUCTION_NOT_SOURCE_PHASE_PRESERVATION')
                if gap==48:
                    previous=R61_index[case['case']]
                    assert previous['status']=='FIXED_GAP_EVALUATED'
                    assert np.array_equal(truth,R61_arrays[previous['truth_arrays_key']])
                    assert methods['G1']['constructible']==previous['methods']['G1']['constructible']
                    if methods['G1']['constructible']:
                        assert np.array_equal(arrays[key+'_G1'],R61_arrays[previous['methods']['G1']['arrays_key']])
                        for field,value in previous['methods']['G1'].items():
                            if field not in ('arrays_key','target_center_scope'):
                                assert methods['G1'][field]==value,field
                    assert row['target_phase_audit']==previous['target_phase_audit']
                    assert row['all_evidence_audit']==previous['all_evidence_audit']
                    parity_rows.append(dict(case=case['case'],truth_exact=True,G1_curve_exact=True,
                        scalar_metrics_exact=True,target_and_all_masks_exact=True))
                if controls:
                    for name in ('SLERP','SingleChartHermite'):
                        paired=methods['G1']['constructible'] and methods[name]['constructible']
                        for metric in ('error_rad','fk_error_m'):
                            row['G1_minus_'+name+'_'+metric]=methods['G1'][metric]-methods[name][metric] if paired else None
                records.append(row)
                print('R62_CONTINUOUS',key,'G1',methods['G1']['constructible'],flush=True)
    finally:
        if main_arrays is not None:main_arrays.close()
        R61_arrays.close()
    result=dict(binding=binding(),records=records,summary=summarize(records,controls),controls=bool(controls),
        R61_main_result_sha256=sha(frozen_path),R60_population_sha256=sha(ROOT.parent/'phase_integrity_r60_02/population/result.json'),
        R61_fixed_result_sha256=sha(R61_fixed_path),
        R61_fixed_arrays_sha256=sha(ROOT.parent/'edit_support_r61_01/fixed_gap/arrays.npz'),baseline48_parity=parity_rows,
        difficulty_stratification=dict(scope='DESCRIPTIVE_ONLY; BASELINE48_VISIBLE_ENDPOINT_DISTANCE_MEDIAN; NO_ERROR_READS',
            median_endpoint_mean_angle_rad=median,case_values={str(k):v for k,v in difficulty.items()},
            case_labels={str(k):v for k,v in labels.items()},all_durations_share_same_source_stratum=True),
        difficulty_summary={label:summarize([r for r in records if r['motion_difficulty']==label],controls)
                            for label in ('lower_or_equal_median','above_median')},
        frozen_geometry_sha256=sha(inspect.getsourcefile(segmented)),reference=assets.reference,
        selection='FROZEN_R61_CASES_AND_CENTERS; NO_SHIFT_OR_REPLACEMENT_FOR_LONGER_GAPS',
        duration_scope='LENGTH_RESPONSE_DESCRIPTIVE; METHODS_PAIRED_ONLY_WITHIN_IDENTICAL_GAP',
        solver_phase_labels_used=False,training_steps=0,optimizer_calls=0,planner_runs=0,can_route=False)
    save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
    return result
