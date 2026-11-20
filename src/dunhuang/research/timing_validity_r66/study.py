"""A fixed source-clock validity experiment, separate from choreography quality."""
from pathlib import Path
import numpy as np
from dunhuang.io import checked
from dunhuang.research.paired_event_r41.events import profiles
from .common import CFG, ROOT, binding, digest, fresh, read, sha, write
from . import engine, population, metrics


def energy_score(speed, clock, interval, offset):
    a,b=interval
    times=np.arange(a,b,dtype=float)+offset
    curve=np.asarray(clock['energy_curve'],float)
    if len(times)==0 or times[0]<0 or times[-1]>len(curve)-1:
        return None
    x=np.asarray(speed[a:b],float);y=np.interp(times,np.arange(len(curve)),curve)
    x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den>1e-12 else None


def donor_distance(clock,other,interval):
    a,b=interval
    if a-15<0 or b+15>len(other['energy_curve']) or b+15>len(clock['energy_curve']):return None
    def tempo(c):
        beats=np.asarray(c['beat_frames']);beats=beats[(beats>=a)&(beats<b)]
        return float(1800/np.median(np.diff(beats))) if len(beats)>=2 else None
    t,u=tempo(clock),tempo(other)
    if t is None or u is None or min(t,u)<=0:return None
    times=np.arange(a,b)
    x=np.sqrt(np.mean(np.interp(times,np.arange(len(clock['energy_curve'])),clock['energy_curve'])**2))
    y=np.sqrt(np.mean(np.interp(times,np.arange(len(other['energy_curve'])),other['energy_curve'])**2))
    if min(x,y)<=1e-12:return None
    return float(abs(np.log(y/x))+abs(np.log(u/t)))


def register(out):
    out=fresh(out)
    if sha(CFG['data'])!=CFG['data_sha256']:raise ValueError('R59 registration changed')
    pop=population.select(read(CFG['data']),CFG)
    if pop['registered_query_count']!=CFG['registered_query_count']:raise ValueError('Query population changed')
    write(out/'population.json',pop)
    if pop['excluded'] or pop['selected_query_count']!=pop['registered_query_count']:
        raise ValueError('Stop: registered query support changed; no silent population exclusion')
    write(out/'registration.json',dict(binding=binding(),protocol=CFG,population_sha256=sha(out/'population.json'),
        primary_null='SOURCE_CLOCK_PAIRED_WIN_0_5_AND_FIVE_WAY_TOP1_0_2; GROUP_CI_REQUIRED',
        rank_does_not_establish_unique_artistic_correctness=True,
        no_outcome_dependent_change=True))
    return pop


def compute(out,emit_viewer=False):
    pop=register(out);out=Path(out)
    audio={};motions={};rows=[];source_proofs={}
    audio_refs={s['audio_sha256']:s['audio'] for s in pop['sources'].values()}
    for case in pop['cases']:
        for donor in case['wrong_song_candidates']:audio_refs[donor['audio_sha256']]=donor['audio']
    for i,(key,ref) in enumerate(sorted(audio_refs.items())):
        path=checked(ref);clock=engine.audio_clock(path)
        if clock['audio_sha256']!=key:raise ValueError('Audio clock identity changed')
        audio[key]=clock
        write(out/'clocks'/(key+'.json'),clock)
        print('R66_AUDIO',i+1,len(audio_refs),flush=True)
    for i,(uid,source) in enumerate(sorted(pop['sources'].items())):
        R,proof=engine.load_motion(source)
        phases,phase_proof=engine.motion_events(R)
        # The R59 prefix is an additional sealed-data equality check; derivatives
        # and peaks have already been computed on the complete original record.
        z=np.load(checked(dict(path=source['arrays'],sha256=source['arrays_sha256'])))
        if not np.array_equal(R[:len(z['R'])],z['R']):raise ValueError('Full raw rotations differ from R59')
        motions[uid]=(R,phases,profiles(R)['global'])
        source_proofs[uid]=dict(raw=proof,phases=phase_proof,shared_R59_rotation_prefix_exact=True,
            shared_arrays_sha256=source['arrays_sha256'])
        print('R66_MOTION',i+1,len(pop['sources']),flush=True)
    for case in pop['cases']:
        R,phases,speed=motions[case['source_uid']];clock=audio[case['audio_sha256']]
        interval=case['common_motion_interval'];offset=case['source_audio_offset_frames']
        result=engine.score_offsets(phases,clock,offset,interval,CFG['offsets'])
        scores=result['scores']
        scores['Energy']=[energy_score(speed,clock,interval,offset+d) for d in CFG['offsets']]
        # Density-only has no local correspondence: same full-record density
        # under every shift. Its ties expose rank/tie handling, not new learning.
        density=(len(clock['beat_frames'])+len(clock['accent_frames']))/max(clock['frames'],1)
        scores['Density']=[float(density)]*len(CFG['offsets'])
        options=[]
        for donor in case['wrong_song_candidates']:
            other=audio[donor['audio_sha256']]
            distance=donor_distance(clock,other,case['common_audio_interval'])
            options.append(dict(audio_sha256=donor['audio_sha256'],distance=distance,
                status='ACOUSTIC_MATCH_AVAILABLE' if distance is not None else 'NO_ACTUAL_COMMON_SUPPORT_OR_TEMPO_OR_RMS'))
        available=[d for d in options if d['distance'] is not None]
        chosen=min(available,key=lambda d:(d['distance'],d['audio_sha256'])) if available else None
        wrong={k:[] for k in scores};wrong_details=None
        if chosen is not None:
            other=audio[chosen['audio_sha256']]
            # Pass the original frozen query phases, never a newly detected set.
            wrong_details=engine.score_offsets(result['events'],other,offset,interval,CFG['offsets'])
            if wrong_details['events']!=result['events']:raise ValueError('Wrong-song changed motion events')
            wrong={k:[v[2]] for k,v in wrong_details['scores'].items()}
            wrong['Energy']=[energy_score(speed,other,interval,offset)]
            wrong['Density']=[float((len(other['beat_frames'])+len(other['accent_frames']))/max(other['frames'],1))]
        row=dict(uid=case['query_uid'],group=case['group'],source_uid=case['source_uid'],dataset=case['dataset'],role=case['role'],
            scores=scores,wrong_scores=wrong,motion_event_count=len(result['events']),
            no_motion_events=not result['events'],source_interval=case['source_interval'],
            common_motion_interval=interval,common_audio_interval=case['common_audio_interval'],
            raw_motion_sha256=case['raw_motion_sha256'],audio_sha256=case['audio_sha256'],clock_scope=case['clock_scope'],
            source_audio_offset_frames=offset,offsets=CFG['offsets'],event_clock='RAW_AUDIO_30FPS',
            event_observations=result['events'],evidence=result.get('evidence',{}),
            donor_candidates=options,selected_donor=chosen,wrong_details=wrong_details,
            semantic_event_truth='UNKNOWN',artistic_wrongness_truth='UNKNOWN')
        rows.append(row)
    write(out/'scores.json',rows)
    write(out/'source_proofs.json',source_proofs)
    write(out/'summary.json',metrics.summarize(rows,seed=CFG['seed'],draws=CFG['bootstrap_draws']))
    syn=engine.synthetic_control() if hasattr(engine,'synthetic_control') else synthetic_control()
    write(out/'synthetic_control.json',syn)
    write(out/'qualification.json',dict(actual_paired_clock_scores=True,all_registered_queries=len(rows),
        human_reviewed=False,semantic_event_truth='UNKNOWN',artistic_quality='NOT_EVALUABLE',
        Native24='NOT_EVALUABLE',new_independent_generalization=False,can_route=False,
        planner_calls=0,geometry_solver_calls=0,optimizer_calls=0,training_steps=0,test_tensor_reads=0))
    if emit_viewer:
        from .viewer import build
        build(out,pop,rows,motions,audio)
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out/'artifacts_manifest.json',dict(files=files,binding=binding()))
    print('R66_FINISHED',len(rows),flush=True)


def synthetic_control():
    return engine.synthetic_control()


def verify(source,out):
    source=Path(source);manifest=read(source/'artifacts_manifest.json')
    now=binding()
    if now['code']!=manifest['binding']['code'] or now['dependencies']!=manifest['binding']['dependencies']:
        raise ValueError('Scientific code changed before replay')
    for name,value in manifest['files'].items():
        if sha(source/name)!=value:raise ValueError('Frozen output changed: '+name)
    compute(out,True)
    names=['population.json','scores.json','source_proofs.json','summary.json','synthetic_control.json','qualification.json','viewer_manifest.json']
    differences=[name for name in names if sha(source/name)!=sha(Path(out)/name)]
    clocks=sorted((source/'clocks').glob('*.json'))
    differences += ['clocks/'+p.name for p in clocks if sha(p)!=sha(Path(out)/'clocks'/p.name)]
    viewer=sorted((source/'viewer').glob('*'))
    differences += ['viewer/'+p.name for p in viewer if sha(p)!=sha(Path(out)/'viewer'/p.name)]
    result=dict(scientific_outputs_exact=not differences,differences=differences,
        viewer_files_rebuilt_exact=not any(n.startswith('viewer') for n in differences),viewer_files_rebuilt=len(viewer),raw_audio_clocks_reextracted=len(clocks),full_record_motion_recomputed=len(read(source/'source_proofs.json')),
        original_frozen_files_rehashed=len(manifest['files']),source_manifest_sha256=sha(source/'artifacts_manifest.json'),
        fresh_process_required=True,binding=now,optimizer_calls=0,training_steps=0,new_planner_runs=0)
    write(Path(out)/'result.json',result)
    if differences:raise ValueError('Independent replay differs: '+str(differences))
