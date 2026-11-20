"""R68 fixed local time interventions; source reconstruction, not neural generation."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import scipy
from dunhuang.research.complete_r36.core import rotation_sample,relative_fk
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.rhythm_r66 import metrics as legacy_metrics
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.paired_event_r41.events import profiles,phases
from dunhuang.research.timing_r64.music import _reward
from .common import CFG,ROOT,read,write,sha,fresh,binding,seal,verify_manifest,digest
from . import warp,metrics


def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v


def select_event(case,kind):
    """Metadata selection is fixed before legality or motion metrics are read."""
    w=np.asarray(case['time_warp']);a,b=case['source_interval'];A=case['target_interval'][0]
    matching=[p for p in case['phase_inputs'] if p['type']==kind]
    whole=[p for p in matching if a<=p['evidence_span'][0]<=p['frame']<p['evidence_span'][1]<=b]
    eligible=[p for p in whole if 64<=np.interp(p['frame'],w[:,0],w[:,1])-A<=175]
    chosen=min(eligible,key=lambda p:(abs(np.interp(p['frame'],w[:,0],w[:,1])-A-119.5),p['frame'])) if eligible else None
    return deepcopy(chosen),dict(kind=kind,matching=len(matching),whole_evidence=len(whole),guardable=len(eligible),
        rule=CFG['event_selection'],semantic_truth='UNKNOWN',replacement_after_failed_intervention=False)


def plan_case(case):
    events={};selections={};maps={'E1W':warp.baseline(case['time_warp'],case['target_interval']),
        'Uniform':warp.uniform(case['source_interval'],case['target_interval'])}
    for label,kind in CFG['event_types'].items():
        event,proof=select_event(case,kind);events[label]=event;selections[label]=proof
        for name,delta in [('early',-6),('late',6)]:
            maps[label+'_'+name]=warp.intervene(case['time_warp'],case['target_interval'],event,
                delta=delta,half_width=CFG['bump_half_width_target_frames'],guard_frames=4)
    case=deepcopy(case);case.update(selected_events=events,event_selection=selections)
    return case,maps


def frame_masks(case,label,position_map):
    support=np.zeros(240,bool);event=np.zeros(240,bool);guard=np.zeros(240,bool);guard[:4]=True;guard[-4:]=True
    phase=case['selected_events'].get(label)
    if phase is not None:
        w=np.asarray(case['time_warp']);A=case['target_interval'][0]
        centre=float(np.interp(phase['frame'],w[:,0],w[:,1])-A)
        support=(np.arange(240)>centre-60)&(np.arange(240)<centre+60)
        # Fixed evidence+6 envelope is shared by both signs, never outcome-selected.
        lo,hi=phase['evidence_span']
        left=np.interp(lo,w[:,0],w[:,1])-A-6
        right=np.interp(hi-1,w[:,0],w[:,1])-A+6
        event=(np.arange(240)>=np.floor(left))&(np.arange(240)<=np.ceil(right))
    if position_map.get('method')=='UNIFORM_EXPLICIT':
        support[:]=True;guard[:]=False
    return dict(event_mask=event,support_mask=support,edit_mask=support.copy(),
        compensation_mask=support&~event&~guard,guard_mask=guard)


def compact_map(m):
    return plain({k:v for k,v in m.items() if k not in ('q',)})


def source_metrics(source,case,J,parents):
    a,b=case['source_interval'];native=source[a:b]
    value=legacy_metrics._motion(native,relative_fk(native,J,parents),False)
    value.update(scope='ORIGINAL_CONTIGUOUS_SOURCE_AT_NATIVE30FPS; NOT_SAME_DURATION_AS240_CONTROL',
        source_interval=[a,b],full_record_derivatives_then_crop_equivalent_for_wholly_internal_stencils=True,
        source_native_duration_seconds=(b-a)/30,source_R_sha256=array_hash(native))
    return value


def pair_metrics(R,J,parents):
    if R.shape!=(528,22,3,3):raise ValueError('Complete528-frame pair required')
    d=legacy_metrics._motion(R,relative_fk(R,J,parents),False)
    d.update(scope='ACTUAL_FULL528_BODY22_PAIR_ALL_STENCILS_NOT_NATIVE24',can_route=False)
    return d


def execute(case,maps,method,ctx,base,base_connection,base_geometry):
    m=maps[method];row=dict(case_key=case['key'],method=method,status=m['status'],reason=m.get('reason'),
        source_uid=case['source_uid'],source_interval=case['source_interval'],target_interval=case['target_interval'],
        audio_sha256=case['audio_sha256'],warp=compact_map(m),semantic_phase_truth='UNKNOWN',
        human_reviewed=False,Native24='NOT_EVALUABLE',artistic_quality='NOT_EVALUABLE',can_route=False)
    label='B' if method.startswith('B_') else 'A';masks=frame_masks(case,label,m)
    row['viewer']=dict(**{k:plain(v) for k,v in masks.items()},
        event_target_before_local=m.get('event_target_frame_before'),event_target_after_local=m.get('event_target_frame_after'))
    if m['status']!='LEGAL':return dict(method=method,R=None,q=None,row=row),{}
    q=np.asarray(m['source_positions']);q0=np.asarray(maps['E1W']['source_positions'])
    source=ctx['lib'].rotations[case['source_uid']]
    R=base.copy() if method=='E1W' else rotation_sample(source,q)
    guard=0 if method=='Uniform' else 4
    if method=='E1W':
        # Baseline has no actual edits but uses the same A-region population.
        masks['edit_mask']=masks['support_mask'].copy()
    measured=metrics.measure(R,base,ctx['J'],ctx['parents'],q,q0,masks['event_mask'],masks['edit_mask'],
        support_mask=masks['support_mask'],guard_frames=guard,include_curves=True)
    if method=='E1W':connection,geometry=deepcopy(base_connection),{k:v.copy() for k,v in base_geometry.items()}
    else:connection,geometry=connectivity.check(case,R,ctx)
    baseline_bridge=base_geometry.get('bridge_R');actual_bridge=geometry.get('bridge_R')
    equal_bridge=(array_hash(actual_bridge)==array_hash(baseline_bridge)
                  if baseline_bridge is not None and actual_bridge is not None else None)
    same_readset=connection['boundary_readset_sha256']==base_connection['boundary_readset_sha256']
    if guard and (not same_readset or equal_bridge is not True):
        raise AssertionError('Strict internal intervention changed actual G1 readset or bridge bytes')
    row.update(metrics=measured,connection=connection,
        bridge_invariance=dict(boundary_readset_identical=same_readset,actual_bridge_bytes_identical=equal_bridge,
            fixed_gap_frames=48,baseline_bridge_R_sha256=base_connection.get('bridge_R_sha256'),
            scope='ACTUAL_RECOMPUTATION_NOT_INHERITED_QUALIFICATION; UNIFORM_MAY_CHANGE_EDGE_JETS'),
        pair_dynamics=pair_metrics(geometry['full_R'],ctx['J'],ctx['parents']) if geometry.get('full_R') is not None else None,
        baseline_pair_dynamics=pair_metrics(base_geometry['full_R'],ctx['J'],ctx['parents']) if base_geometry.get('full_R') is not None else None,
        R_sha256=array_hash(R),q_sha256=array_hash(q),
        source_native=source_metrics(source,case,ctx['J'],ctx['parents']))
    row['viewer']=dict(**{k:plain(v) for k,v in masks.items()},
        event_target_before_local=m.get('event_target_frame_before'),event_target_after_local=m.get('event_target_frame_after'),
        actual_FK_D3=measured['whole']['actual']['FK']['D3']['curve'],
        baseline_FK_D3=measured['whole']['baseline']['FK']['D3']['curve'])
    arrays=dict(R=R,q=q,**{'G1_'+k:v for k,v in geometry.items()})
    return dict(method=method,R=R,q=q,row=row),arrays


def music_proxy(R,q,case,clock,origin=None):
    """Original R64 reward, no fit or selection; both peak sets use same ROI."""
    origin=case['target_interval'][0] if origin is None else origin
    A,B=origin,origin+240
    if not 0<=A<B<=clock['frames']:raise ValueError('Music control leaves real audio clock')
    detections=phases(profiles(R),read('src/dunhuang/research/manifold_bridge_r47/config.json'))
    accents=np.asarray(clock['accent_frames']);strength=np.interp(accents,np.arange(clock['frames']),clock['accent_strength'])
    def score(local):
        absolute=np.asarray(local)+origin
        beat=_reward(absolute,clock['beat_frames'],3);accent=_reward(absolute,accents,3,strength)
        return dict(query_count=len(local),beat=beat,accent=accent,
            primary=.5*beat['score']+.5*accent['score'] if beat['score'] is not None and accent['score'] is not None else None)
    result={}
    for label,kind in CFG['event_types'].items():
        rawkind=kind.removeprefix('upper_')
        detected=[p['frame'] for p in detections if p['type']==rawkind and p['body_region']=='upper' and p['evidence_span'][0]>=3 and p['evidence_span'][1]<=237]
        old=[float(np.interp(p['frame'],q,np.arange(240))) for p in case['phase_inputs'] if p['type']==kind and case['source_interval'][0]<=p['evidence_span'][0] and p['evidence_span'][1]<=case['source_interval'][1]]
        mapped=[v for v in old if 3<=v<237]
        result[label]=dict(mapped_source=score(mapped),redetected_output=score(detected),
            peak_mismatch_is_not_a_bug_or_semantic_label=True)
    return dict(by_type=result,audio_sha256=clock['audio_sha256'],audio_interval=[A,B],
        scope='FROZEN_R64_MANY_TO_NEAREST_PROXY_ONLY; NO_MUSIC_QUALITY_TRUTH_NO_SELECTION')


def prepare(out):
    parent=Path('outputs/sdmr_warp_rank_01/main')
    if sha(parent/'manifest.json')!=CFG['parent_r67_main_manifest_sha256']:raise ValueError('R67 frozen parent manifest differs')
    verify_manifest(parent)
    ctx=connectivity.prepare();planned=[plan_case(c) for c in ctx['change_cases']]
    if len(planned)!=12:raise ValueError('No population replacement permitted')
    clocks_path=Path('outputs/timing_r64_01/main/music_clocks.json')
    if sha(clocks_path)!='0a7bd68287f1d98e81639fb3545d78ddca9ae22f01abb89e333eef6587d30247':raise ValueError('Music clock changed')
    write(out/'population.json',dict(cases=[c for c,m in planned],registered_cases=12,unique_source_intervals=8,
        source_semantics='KINEMATIC_PROXIES_ONLY',music_exposure='THREE_HISTORICALLY_EXPOSED_TRACKS'))
    write(out/'plan.json',[dict(case_key=c['key'],maps={n:compact_map(m) for n,m in maps.items()}) for c,maps in planned])
    write(out/'registration.json',dict(binding=binding(),protocol=CFG,parent_manifest_sha256=sha(parent/'manifest.json'),
        original_numerical_gate=ctx['reference'],all_candidates_before_measurements=True,
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=sys.platform)))
    return ctx,planned,read(clocks_path)


def summarize(records):
    by_method={}
    for method in CFG['control_methods']:
        own=[r for r in records if r['method']==method]
        if not own:continue
        legal=[r for r in own if r['status']=='LEGAL']
        intervals={}
        for r in legal:
            key=r['source_uid']+':'+str(r['source_interval'])
            intervals.setdefault(key,[]).append(r)
        groups=[]
        for key,rs in sorted(intervals.items()):
            groups.append(dict(source_interval_identity=key,cases=len(rs),
                FK_D3_Q95_delta_mean=float(np.mean([v['metrics']['whole']['actual']['FK']['D3']['p95']-v['metrics']['whole']['baseline']['FK']['D3']['p95'] for v in rs])),
                actual_source_shift_mean=float(np.mean([v['metrics']['dose']['maximum_abs_source_position_change_frames'] for v in rs]))))
        by_method[method]=dict(registered=len(own),status_counts=dict(Counter(r['status'] for r in own)),legal=len(legal),
            unique_legal_source_intervals=len(groups),source_interval_means=groups,
            source_interval_balanced_FK_D3_Q95_delta=float(np.mean([g['FK_D3_Q95_delta_mean'] for g in groups])) if groups else None,
            changes_in_D3_are_not_quality_wins=True,no_artistic_label=True,
            G1_passed_original_screen=sum(r['connection']['passes_original_screen'] is True for r in legal),
            actual_bridge_identical_to_E1W=sum(r['bridge_invariance']['actual_bridge_bytes_identical'] is True for r in legal))
    return dict(methods=by_method,registered_rows=len(records),human_reviewed=False,Native24='NOT_EVALUABLE',
        artistic_quality='NOT_EVALUABLE',stage_semantics='UNKNOWN',can_route=False,neural_training_steps=0)


def _run(mode,out=None):
    target=fresh(out or ROOT/mode)
    if mode=='controls':
        preview=ROOT/'preview';verify_manifest(preview)
        acceptance=ROOT/'receipts/preview_browser_acceptance.json'
        a=read(acceptance)
        if a.get('status')!='PASS' or a.get('source_manifest_sha256')!=sha(preview/'manifest.json'):
            raise ValueError('Actual preview UI acceptance must precede controls')
    ctx,planned,clocks=prepare(target)
    methods=CFG['preview_methods'] if mode=='preview' else CFG['control_methods']
    frozen_rows=read(ROOT/'preview/records.json') if mode=='controls' else None
    reference_rows={(r['case_key'],r['method']):r for r in frozen_rows} if frozen_rows else {}
    arrays={};records=[];display=[]
    for case,maps in planned:
        q0=np.asarray(maps['E1W']['source_positions']);source=ctx['lib'].rotations[case['source_uid']]
        base=rotation_sample(source,q0)
        if array_hash(base)!=case['baseline_core_R_sha256']:raise ValueError('Actual original E1-W Core differs')
        base_connection,base_geometry=connectivity.check(case,base,ctx);panels=[]
        for method in methods:
            panel,values=execute(case,maps,method,ctx,base,base_connection,base_geometry)
            row=panel['row']
            if row['status']=='LEGAL':row['music_proxy']=music_proxy(panel['R'],panel['q'],case,clocks[str(case['songidx'])])
            if mode=='controls' and method in CFG['preview_methods']:
                old=reference_rows[(case['key'],method)]
                if row['status']!=old['status'] or row.get('R_sha256')!=old.get('R_sha256') or row.get('q_sha256')!=old.get('q_sha256'):
                    raise ValueError('Previously delivered candidate changed during controls')
                row['preview_candidate_identity_exact']=True
            if mode=='controls':
                songs=sorted({c['songidx'] for c,m in planned});next_song=songs[(songs.index(case['songidx'])+1)%len(songs)]
                donor=next(c for c,m in planned if c['songidx']==next_song)
                case['alternate_music']=dict(songidx=donor['songidx'],audio_path=clocks[str(next_song)]['audio_path'],
                    audio_sha256=donor['audio_sha256'],target_interval=donor['target_interval'])
                if row['status']=='LEGAL':
                    row['alternate_music_proxy']=music_proxy(panel['R'],panel['q'],case,clocks[str(next_song)],donor['target_interval'][0])
                    row['music_cross_invariance']=dict(motion_R_sha256=row['R_sha256'],q_sha256=row['q_sha256'],
                        shared_same_motion_array=True,shared_same_dynamics_record=True,
                        shared_same_dynamics_sha256=digest(row['metrics']),
                        shared_same_pair_dynamics_sha256=digest(row['pair_dynamics']),
                        shared_same_bridge_R_sha256=row['connection'].get('bridge_R_sha256'),
                        correct_audio_sha256=case['audio_sha256'],alternate_audio_sha256=donor['audio_sha256'],
                        correct_audio_interval=case['target_interval'],alternate_audio_interval=donor['target_interval'],
                        music_quality='NOT_EVALUABLE',context_matching=CFG['music_cross'])
            panels.append(panel);records.append(plain(row))
            for name,value in values.items():arrays[case['key']+'__'+method+'__'+name]=value
        a,b=case['source_interval'];native=dict(R=source[a:b],fps=30,source_start_frame=a)
        display.append((case,panels,native));print('R68_'+mode.upper(),len(display),12,flush=True)
    save_arrays(target/'arrays.npz',arrays);write(target/'records.json',records);write(target/'summary.json',summarize(records))
    registration=read(target/'registration.json');registration.update(connectivity_counts=ctx['counters'])
    if mode=='controls':registration['preview_acceptance_sha256']=sha(ROOT/'receipts/preview_browser_acceptance.json')
    write(target/'registration.json',registration)
    from .viewer import build
    build(target,display,ctx['J'],ctx['parents'],clocks,methods=methods)
    seal(target);print('R68_FINISHED',mode,len(records),flush=True)
    return target


def run_preview(out=None):return _run('preview',out)
def run_controls(out=None):return _run('controls',out)
def verify(mode,out):
    source=ROOT/mode;old=verify_manifest(source);now=binding()
    if now['code']!=old['binding']['code'] or now['dependencies']!=old['binding']['dependencies']:raise ValueError('Scientific code changed')
    _run(mode,out)
    differences=[name for name,h in old['files'].items() if sha(Path(out)/name)!=h]
    write(Path(out)/'independent.json',dict(mode=mode,scientific_outputs_exact=not differences,differences=differences,
        rebuilt_files=len(old['files']),source_manifest_sha256=sha(source/'manifest.json'),binding=now,
        neural_training_steps=0,scope='FRESH_RECOMPUTATION_ALL_SAVED_CANDIDATES_AND_REGION_STENCILS'))
    if differences:raise ValueError('Independent replay differs')
