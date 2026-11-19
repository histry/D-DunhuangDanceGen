"""Matched phase-support, seam construction, and actual E1-W support attribution."""
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from .common import *
from .population import Assets
from dunhuang.research.event_feasibility_r48.study import compile_clock
from dunhuang.research.tempo_attribution_r57.common import stats, amplification
from dunhuang.research.bridge_audit_r46.admission import certificate
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.manifold_bridge_r47.geometry import segmented
from dunhuang.research.bridge_dynamics_r48.study import metrics

def error(Q,truth):
    return float(Rotation.from_matrix((truth.swapaxes(-1,-2)@Q).reshape(-1,3,3)).magnitude().mean())

def gap_task(arg):
    key,truth,J,parents,reference=arg;h=CFG['hidden_gap'];gap=(4,4+h)
    # Only the four observed frames on each side enter the solver. Truth stays outside it.
    observed=np.full_like(truth,np.nan);observed[:4]=truth[:4];observed[-4:]=truth[-4:]
    cert=certificate(observed,gap);methods={};curves={};Q,guard=segmented(observed,gap,(4,8,16),np.pi-.05)
    candidates={'G1':Q}
    if cert['chart_eligible']:candidates['G0']=observed_bridge(observed,gap)
    for name,Q in candidates.items():
        if Q is None:methods[name]=dict(constructible=False,passes_original_screen=False,error_rad=None);continue
        assert np.array_equal(Q[:4],truth[:4]) and np.array_equal(Q[-4:],truth[-4:])
        m=metrics(Q,J,parents,gap);m.pop('jerk_curve');passes=m['gap_D2']<=reference['D2'] and m['gap_D3']<=reference['D3'] and m['shoulder_wrist_min_m']>=reference['shoulder_wrist_lower_m']
        methods[name]=dict(constructible=True,passes_original_screen=bool(passes),error_rad=error(Q[4:4+h],truth[4:4+h]),
                           fk_error_m=float(np.linalg.norm(relative_fk(Q,J,parents)[4:4+h]-relative_fk(truth,J,parents)[4:4+h],axis=-1).mean()),metrics=m)
        curves[key+'_'+name]=Q
    if 'G0' not in methods:methods['G0']=dict(constructible=False,passes_original_screen=False,error_rad=None)
    curves[key+'_truth']=truth
    cut=len(truth)//2;virtual=np.concatenate([truth[:cut],truth[cut:]])
    assert np.array_equal(virtual,truth)
    return dict(key=key,methods=methods,G0=cert,G1=guard,virtual_split_array_exact=True,
                virtual_split_error_rad=0.,continuous_native=stats(relative_fk(truth,J,parents)),
                hidden_truth='ACTUAL_CONTINUOUS_RECORDING_48_FRAMES; NOT_SEMANTIC_PHASE_LABEL',solver_hidden_read=False),curves

def source_ci(rows,key):
    grouped={}
    for r in rows:
        if r.get(key) is not None:grouped.setdefault(r['source_uid'],[]).append(r[key])
    values=np.array([np.mean(grouped[k]) for k in sorted(grouped)],float)
    if not len(values):return dict(sources=0,mean=None,ci95=None)
    rng=np.random.default_rng(CFG['seed']);draw=values[rng.integers(len(values),size=(CFG['bootstrap_draws'],len(values)))].mean(1)
    return dict(sources=len(values),pairs=sum(map(len,grouped.values())),mean=float(values.mean()),
                ci95=[float(x) for x in np.quantile(draw,[.025,.975])],sampling_unit='SOURCE_RECORDING_NOT_WINDOW; NO_UNSEEN_SEMANTIC_GENERALIZATION')

def fixed_window(unit, length, source):
    lo,hi=unit['support_component'];mid=sum(unit['source_interval'])/2
    if hi-lo<length:return dict(length=length,status='NOT_CONSTRUCTIBLE_ON_ACCEPTED_COMPONENT')
    center=int(round(mid-length/2));choices=[]
    for a in range(max(lo,center-CFG['pair_max_shift']),min(hi-length,center+CFG['pair_max_shift'])+1):
        count=len(crossings([a,a+length],source['protected_spans']));choices.append((count,abs(a-center),a))
    if not choices:return dict(length=length,status='NOT_CONSTRUCTIBLE_IN_REGISTERED_NEIGHBORHOOD')
    count,_,a=min(choices);return dict(length=length,source_interval=[a,a+length],status='KINEMATIC_PROTECTION_SATISFIED' if count==0 else 'PROTECTION_NOT_SATISFIABLE',protected_crossings=count)

def controls(out, workers=2, gate=True):
    if gate:
        receipt=read(ROOT/'receipts/main_browser.json');assert receipt['passed'] and receipt['main_result_sha256']==sha(ROOT/'main/result.json')
    out=fresh(out);population=read(ROOT/'population/result.json');assets=Assets().build()
    assert assets.units==population['units'];assert list(assets.sources.values())==population['sources']
    unit_rows=[];tasks=[];taskmeta={};fixed=[];legal=[]
    beatpath=Path('outputs/complete_r36_01/planning/song0_music.json');beats=read(beatpath)['beats'];clockcfg=read('src/dunhuang/research/event_feasibility_r48/config.json')
    arrays=dict(J=assets.J,parents=assets.parents)
    for i,u in enumerate(assets.units):
        uid=u['source_uid'];R=assets.R[uid];source=assets.sources[uid];pair=u['pair']
        for length in CFG['fixed_lengths']:
            fixed.append(dict(unit_uid=u['unit_uid'],source_uid=uid,dataset=u['dataset'],**fixed_window(u,length,source)))
        row=dict(unit_uid=u['unit_uid'],source_uid=uid,dataset=u['dataset'],pair=pair,primary_error_A4_minus_A5=None)
        if pair['status']!='MATCHED':unit_rows.append(row);continue
        for arm in ('A4','A5'):
            a,b=pair[arm];native=stats(relative_fk(R[a:b],assets.J,assets.parents))
            phase=phase_slice(source['phase_evidence'],[a,b]);node=dict(source_interval=[a,b],phases=phase)
            # Identical duration domains for equal-length arms; phase anchors are attractions, not hard beat constraints.
            for target in sorted(set(CFG['target_lengths']+[b-a])):
                clock=compile_clock(node,[0,target],beats,clockcfg)
                record=dict(unit_uid=u['unit_uid'],source_uid=uid,dataset=u['dataset'],arm=arm,source_length=b-a,target_length=target,
                            theoretical_ratio=(target-1)/(b-a-1),legal=clock is not None,protected_crossings=len(crossings([a,b],source['protected_spans'])),
                            phase_evidence_truncated=sum(p['evidence_span'][0]<a or p['evidence_span'][1]>b for p in phase))
                if clock is not None:
                    table=np.asarray(clock['time_warp']);q=np.interp(np.arange(target),table[:,1],table[:,0]);sample=rotation_sample(R,q)
                    m=stats(relative_fk(sample,assets.J,assets.parents),table[1:-1,1]);record.update(warp=clock['time_warp'],metrics=m,native=native,amplification=amplification(m,native),
                        phase_order_preserved=True,source_positions_sha256=array_hash(q),boundary_upper_speed_radps=[float(x) for x in np.linalg.norm(Rotation.from_matrix((sample[[0,-2]].swapaxes(-1,-2)@sample[[1,-1]]).reshape(-1,3,3)).as_rotvec().reshape(2,22,3)*30,axis=-1).mean(1)])
                legal.append(record)
            for side in pair['common_hidden_seams']:
                cut=a if side=='left' else b;start=cut-CFG['hidden_gap']//2-4;end=start+CFG['hidden_gap']+8
                proof_interval([start,end],u['support_component']);key='u%d_%s_%s'%(i,arm,side);taskmeta[key]=dict(unit=i,arm=arm,side=side,source_interval=[start,end],cut_frame=cut)
                tasks.append((key,R[start:end].copy(),assets.J,assets.parents,assets.reference))
        unit_rows.append(row)
    results=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k,(record,curves) in enumerate(pool.map(gap_task,tasks)):
            record.update(taskmeta[record['key']]);results.append(record);arrays.update(curves)
            if (k+1)%25==0:print('R60_HIDDEN_BRIDGES',k+1,'/',len(tasks),flush=True)
    lookup={r['key']:r for r in results}
    for i,row in enumerate(unit_rows):
        pair=row['pair']
        if pair['status']!='MATCHED':continue
        values={arm:[] for arm in ('A4','A5')};passes={arm:[] for arm in values}
        for side in pair['common_hidden_seams']:
            entries={arm:lookup['u%d_%s_%s'%(i,arm,side)]['methods']['G1'] for arm in values}
            for arm in values:passes[arm].append(entries[arm]['passes_original_screen'])
            if all(entries[arm]['constructible'] for arm in values):
                for arm in values:values[arm].append(entries[arm]['error_rad'])
        row.update(common_constructible_seams=len(values['A4']),G1_pass_fraction={arm:float(np.mean(v)) if v else None for arm,v in passes.items()},
                   error_rad={arm:float(np.mean(v)) if v else None for arm,v in values.items()})
        if values['A4']:row['primary_error_A4_minus_A5']=row['error_rad']['A4']-row['error_rad']['A5']
    summary={}
    for dataset in ('ChangE','AIST++','FineDance'):
        rows=[r for r in unit_rows if r['dataset']==dataset];paired=[r for r in rows if r['pair']['status']=='MATCHED']
        counts={arm:sum(r['legal'] for r in legal if r['dataset']==dataset and r['arm']==arm) for arm in ('A4','A5')}
        summary[dataset]=dict(units=len(rows),matched=len(paired),not_matchable=len(rows)-len(paired),
            with_common_hidden_support=sum(bool(r['pair']['common_hidden_seams']) for r in paired),
            primary=source_ci(rows,'primary_error_A4_minus_A5'),legal_counts=counts,
            legal_duration_identical_for_equal_length=counts['A4']==counts['A5'])
    result=dict(binding=binding(),population_sha256=sha(ROOT/'population/result.json'),units=unit_rows,fixed_lengths=fixed,legal_profiles=legal,
                hidden_bridges=results,summary=summary,beat_template=dict(path=str(beatpath),sha256=sha(beatpath),scope='FROZEN_UNPAIRED_BEAT_TEMPLATE_NOT_CONTENT_ALIGNMENT'),
                G1_phase_evidence_gate='NOT_ESTABLISHED_NO_INDEPENDENT_HUMAN_OR_CROSS_RECORD_STAGE_TRUTH',
                G2_primary='WITHIN_RECORD_BOUNDARY_SHIFT_ASSOCIATION; NO_SEMANTIC_PHASE_CAUSALITY_OR_NATURALNESS_CLAIM',
                G3_joint_planning='NOT_STARTED; G1_NOT_ESTABLISHED',
                legal_domain_note='EQUAL_SOURCE_LENGTH_HAS_SAME_ENDPOINT_RATE_DOMAIN; EXISTING_OPTIONAL_BEAT_ATTRACTION_CANNOT_PROVE_PHASE_DEPENDENT_FEASIBILITY',
                human_reviewed=False,semantic_completeness='UNKNOWN',Native24='NOT_EVALUABLE',optimizer_calls=0,training_updates=0,can_route=False)
    save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result);print('R60_CONTROLS_COMPLETE',summary,flush=True)
    return result

def verify(out, source, workers=2):
    out=fresh(out);source=Path(source);expected=read(source/'result.json');assert binding()['code']==expected['binding']['code']
    actual=controls(out/'rebuild',workers,False);actual['binding']['commit']=expected['binding']['commit']
    assert actual==expected;assert sha(source/'arrays.npz')==sha(out/'rebuild/arrays.npz')
    pop=read(ROOT/'population/result.json');n=check_seal(pop['parent_seal'])
    write(out/'result.json',dict(result_exact=True,arrays_bytes_exact=True,reencoded_sources=len(pop['sources']),
                                parent_files_unchanged=n,optimizer_calls=0,source_result_sha256=sha(source/'result.json'),
                                replay_commit=binding()['commit'],source_commit=expected['binding']['commit']))
