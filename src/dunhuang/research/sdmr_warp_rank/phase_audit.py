"""Existing kinematic event types against real audio clocks; no reward fitting."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from dunhuang.research.paired_event_r41.events import profiles,phases
from dunhuang.research.timing_r64.music import _reward
from dunhuang.research.timing_validity_r66.engine import load_motion
from dunhuang.research.warp_coupling_r53.common import save_arrays
from .common import CFG,ROOT,read,write,sha,fresh,binding

def parent_file(root,name):
    manifest=read(Path(root)/'artifacts_manifest.json')['files']
    if sha(Path(root)/name)!=manifest[name]:raise ValueError('Frozen parent changed: '+name)
    return Path(root)/name

def rewards(times,clock):
    accent=np.asarray(clock['accent_frames']);strength=np.interp(accent,np.arange(clock['frames']),clock['accent_strength'])
    a=_reward(times,clock['beat_frames'],3)['score'];b=_reward(times,accent,3,strength)['score']
    return .5*a+.5*b if a is not None and b is not None else None

def consistency(record):
    meta=record.get('metrics')
    if not meta:return dict(status=record['status'])
    src=meta['source_phase_diagnostics'];out=meta['actual_output_peaks'].get('peaks',[])
    result=[]
    for region in ('upper','lower','trunk'):
        a=np.asarray([p['target_frame'] for p in src if p['type'].endswith('speed_peak') and p['body_region']==region and p['source_evidence_whole_inside_core'] and 3<=p['target_frame']<237])
        b=np.asarray([p['frame'] for p in out if p['body_region']==region])
        distances=abs(a[:,None]-b[None,:]) if len(a) and len(b) else None
        if distances is not None:
            i,j=linear_sum_assignment(distances);matched=distances[i,j];within=int((matched<=3).sum())
        else:matched=np.asarray([]);within=0
        result.append(dict(region=region,mapped_source_peaks=len(a),redetected_output_peaks=len(b),
            one_to_one_within3_count=within,source_fraction_within3=within/len(a) if len(a) else None,
            mean_assigned_distance_frames=float(matched.mean()) if len(matched) else None))
    return dict(status='DETECTOR_CONSISTENCY_ONLY_NOT_EVENT_TRUTH',regions=result,
        mapped_alignment=meta['mapped_source_peaks']['alignment'],output_alignment=meta['actual_output_peaks'].get('alignment'),
        reason='OUTPUT_SPEED_EQUALS_SOURCE_SPEED_TIMES_INVERSE_CLOCK_RATE; MAXIMA_NEED_NOT_BE_INVARIANT')

def _balanced(rows,values):
    groups=defaultdict(lambda:defaultdict(list))
    for row,value in zip(rows,values):
        if value is not None:groups[row['group']][row['raw_motion_sha256']].append(float(value))
    means={g:float(np.mean([np.mean(v) for v in sources.values()])) for g,sources in sorted(groups.items())}
    return dict(registered_queries=len(rows),evaluable_queries=sum(v is not None for v in values),groups=len(means),
        group_means=means,mean=float(np.mean(list(means.values()))) if means else None)

def run(out=None):
    target=fresh(out or ROOT/'phase_audit');parent=Path(CFG['phase_audit']['parent'])
    pop=read(parent_file(parent,'population.json'));old_scores=read(parent_file(parent,'scores.json'))
    sources=pop['sources'];old={r['uid']:r for r in old_scores};clocks={}
    required={s['audio_sha256'] for s in sources.values()}
    required.update(r['selected_donor']['audio_sha256'] for r in old_scores if r['selected_donor'])
    for key in sorted(required):clocks[key]=read(parent_file(parent,'clocks/'+key+'.json'))
    cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json');full={};proof={};types=set()
    for uid,source in sorted(sources.items()):
        R,p=load_motion(source);events=phases(profiles(R),cfg);full[uid]=events;proof[uid]=p
        types.update((e['type'],e['body_region']) for e in events if e['type'] in CFG['phase_audit']['types'])
    lags=CFG['phase_audit']['lags'];deltas=CFG['phase_audit']['offsets'];rows=[];arrays={}
    for case in pop['cases']:
        a,b=case['source_interval'];offset=case['source_audio_offset_frames'];clock=clocks[case['audio_sha256']]
        donor=old[case['query_uid']]['selected_donor'];other=clocks[donor['audio_sha256']] if donor else None
        for kind,region in sorted(types):
            events=[e for e in full[case['source_uid']] if (e['type'],e['body_region'])==(kind,region) and a+30<=e['frame']<b-30]
            p=np.asarray([e['frame']+offset for e in events],float);matrix=np.full((len(lags),len(deltas)),np.nan);wrong=np.full(len(lags),np.nan)
            for i,lag in enumerate(lags):
                for j,delta in enumerate(deltas):
                    value=rewards(p+lag+delta,clock);matrix[i,j]=np.nan if value is None else value
                if other is not None and len(p) and p.min()-15>=0 and p.max()+15<other['frames']:
                    value=rewards(p+lag,other);wrong[i]=np.nan if value is None else value
            key='r%05d'%len(rows);arrays[key]=matrix;arrays[key+'_wrong']=wrong
            rows.append(dict(key=key,uid=case['query_uid'],dataset=case['dataset'],role=case['role'],group=case['group'],
                source_uid=case['source_uid'],raw_motion_sha256=case['raw_motion_sha256'],type=kind,region=region,
                event_count=len(events),frame_scope=[a+30,b-30],source_phase_spans=[e['evidence_span'] for e in events],
                unknown_semantics=True,clock_scope=case['clock_scope']))
    save_arrays(target/'scores.npz',arrays);summary=[]
    for dataset,role in sorted({(r['dataset'],r['role']) for r in rows}):
        for kind,region in sorted(types):
            own=[r for r in rows if (r['dataset'],r['role'],r['type'],r['region'])==(dataset,role,kind,region)]
            lag_rows=[]
            for i,lag in enumerate(lags):
                wins=[];wrong_wins=[]
                for r in own:
                    v=arrays[r['key']][i];correct=v[2]
                    if np.isfinite(v).all():
                        diff=correct-np.delete(v,2);wins.append(float(np.mean(np.where(abs(diff)<=1e-12,.5,diff>0))))
                    else:wins.append(None)
                    w=arrays[r['key']+'_wrong'][i]
                    wrong_wins.append(float(.5 if abs(correct-w)<=1e-12 else correct>w) if np.isfinite([correct,w]).all() else None)
                lag_rows.append(dict(lag=lag,correct_vs_shift=_balanced(own,wins),correct_vs_wrong_song=_balanced(own,wrong_wins)))
            summary.append(dict(dataset=dataset,role=role,type=kind,region=region,lags=lag_rows,
                event_total=sum(r['event_count'] for r in own),no_event_queries=sum(not r['event_count'] for r in own)))
    r66root=Path('outputs/rhythm_r66_01/main');r66=read(parent_file(r66root,'result.json'))
    diagnostic=[dict(case_key=r['case_key'],method=r['method'],source_uid=r['source_uid'],result=consistency(r)) for r in r66['records']]
    write(target/'summary.json',dict(records=summary,scope='DESCRIPTIVE_LATENCY_SWEEP_ALL_LAGS_NOT_BEST_LAG_SELECTION',
        no_new_reward=True,latency_fitted=False,pose_extrema='NOT_EVALUABLE_NO_REGISTERED_DETECTOR',
        motion_semantic_truth='UNKNOWN',human_reviewed=False,can_route=False))
    write(target/'r66_detector_consistency.json',diagnostic);write(target/'rows.json',rows);write(target/'source_proofs.json',proof)
    from .population import paired
    from .phase_viewer import build as build_viewer
    chosen=[];seen=set()
    for item in paired(CFG['r59_manifest'])['cases']:
        if item['group'] in seen:continue
        seen.add(item['group']);a,b=item['source_interval'];A,B=item['audio_interval'];source=item['source']
        R,_=load_motion(source);events=[]
        for e in full[item['source_uid']]:
            if a<=e['frame']<b:
                v=dict(e);v['frame']=e['frame']-a;v['evidence_span']=[x-a for x in e['evidence_span']];events.append(v)
        clock=clocks[item['audio_sha256']]
        chosen.append(dict(key=item['key'],source_uid=item['source_uid'],group=item['group'],R=R[a:b],audio_path=source['audio']['path'],source_interval=[a,b],audio_interval=[A,B],events=events,beat_frames=[x-A for x in clock['beat_frames'] if A<=x<B],accent_frames=[x-A for x in clock['accent_frames'] if A<=x<B]))
        if len(chosen)==3:break
    sk=read(CFG['r59_manifest'])['skeleton'];build_viewer(target,chosen,np.asarray(sk['J']),sk['parents'])
    write(target/'registration.json',dict(binding=binding(),protocol=CFG['phase_audit'],parent_sha256=sha(parent/'artifacts_manifest.json'),
        source_queries=338,all_delays_and_shifts_share_events=True,event_point='EXISTING_DETECTOR_FRAME; HOLD_AND_TURN_SPAN_CENTER_NOT_SEMANTIC_COMPLETION'))
    write(target/'manifest.json',dict(files={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()},binding=binding()))
    print('R67_PHASE_AUDIT',len(rows),len(types),flush=True)
