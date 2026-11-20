"""Fixed-source, fixed240-frame Core effects before the clock-shift control."""
from pathlib import Path
import copy
import numpy as np
from dunhuang.research.complete_r36.core import rotation_sample,multi_warp
from dunhuang.research.budget_r63.engine import _phases
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
from dunhuang.research.timing_r64.music import validate_clock,_reward
from .common import ROOT,CFG,binding,read,write,sha,fresh
from . import population_b,warp,metrics

PARENT=Path('outputs/timing_r64_01')

def clocks():
    path=PARENT/'main/music_clocks.json'
    entry=next(r for r in read('reports/r64_release/scientific_files.json')['files'] if r['path']=='main/music_clocks.json')
    if sha(path)!=entry['sha256']:raise ValueError('Frozen R64 audio clocks changed')
    result=read(path)
    for clock in result.values():validate_clock(clock)
    return result

def score_map(case,phases,q,clock):
    a,b=case['source_interval'];A,B=case['target_interval']
    peaks=[p['frame'] for p in phases if p['type'].endswith('speed_peak') and a<=p['frame']<=b-1]
    times=np.interp(peaks,q,np.arange(A,B))
    accents=np.asarray(clock['accent_frames']);strength=np.interp(accents,np.arange(clock['frames']),clock['accent_strength'])
    beat=_reward(times,clock['beat_frames'],3);accent=_reward(times,accents,3,strength)
    primary=.5*beat['score']+.5*accent['score'] if beat['score'] is not None and accent['score'] is not None else None
    return dict(beat=beat,accent=accent,primary=primary,scope='MAPPED_SOURCE_PEAK_PROXY_NOT_ARTISTIC_QUALITY',
        interpolation='INVERSE_DENSE_SAMPLED_SOURCE_POSITIONS_TO_OUTPUT_FRAME_CENTERS')

def compute(mode='main',out=None):
    target=fresh(out or ROOT/mode)
    pop=population_b.build(PARENT);assets=population_b.load_assets(PARENT)
    lib,J,parents=assets['lib'],np.asarray(assets['J']),assets['parents'];music=clocks()
    write(target/'population.json',pop)
    write(target/'registration.json',dict(binding=binding(),protocol=CFG,population_sha256=sha(target/'population.json'),
        warp_search_config=warp.CONFIG,case_selection_independent_of_scores=True))
    rows=[];arrays={};display=[]
    methods=CFG['main_methods'] if mode=='main' else CFG['control_methods']
    prior_rows=[];prior_arrays=None
    if mode=='controls':
        prior_manifest=read(ROOT/'main/artifacts_manifest.json')
        for name,value in prior_manifest['files'].items():
            if sha(ROOT/'main'/name)!=value:raise ValueError('Main candidate changed before controls')
        prior_rows=read(ROOT/'main/result.json')['records'];prior_arrays=np.load(ROOT/'main/arrays.npz')
    for case in pop['cases']:
        source=lib.rotations[case['source_uid']];clock=music[str(case['songidx'])]
        if clock['audio_sha256']!=case['audio_sha256']:raise ValueError('Frozen music identity differs')
        a,b=case['source_interval'];A,B=case['target_interval'];phases=case['phase_inputs']
        if case.get('source_R_sha256') and array_hash(source)!=case['source_R_sha256']:
            raise ValueError('Full source rotation identity differs from frozen parent')
        old=multi_warp([a,b],[A,B],_phases(phases),case['beats'])
        if old is None or old['time_warp']!=case['time_warp']:raise ValueError('Frozen E1-W map changed')
        oldmap=np.asarray(old['time_warp']);oldq=np.interp(np.arange(A,B),oldmap[:,1],oldmap[:,0])
        oldR=rotation_sample(source,oldq)
        if case.get('baseline_core_R_sha256') and array_hash(oldR)!=case['baseline_core_R_sha256']:
            raise ValueError('Frozen E1-W body rotation hash changed')
        own=[]
        if prior_arrays is not None:
            for prior in prior_rows:
                if prior['case_key']==case['key']:
                    key=case['key']+'_'+prior['method'];own.append(dict(method=prior['method'],R=prior_arrays[key] if prior['status']=='LEGAL' else None,q=prior_arrays[key+'_q'] if prior['status']=='LEGAL' else None,row=prior))
        for method in methods:
            if method=='Uniform':result=warp.uniform([a,b],[A,B],phases,clock,case['support_component'])
            elif method=='E1W':result=dict(status='LEGAL',source_positions=oldq.tolist(),source_to_target_anchors=old['time_warp'],
                interpolation='FROZEN_R36_PIECEWISE_LINEAR',cost=old['cost'],frozen_parent_rotation_exact=True)
            else:result=warp.optimize([a,b],[A,B],phases,clock,case['support_component'],control=method=='ShuffledRhythm')
            row=dict(case_key=case['key'],source_uid=case['source_uid'],source_interval=[a,b],target_interval=[A,B],
                songidx=case['songidx'],method=method,status=result['status'],warp=result,
                quality='NOT_EVALUABLE',semantic_event_truth='UNKNOWN',adjacent_bridge_revalidated=False,
                transition_frames_reserved=48,neural_training_steps=0)
            if result['status']=='LEGAL':
                q=np.asarray(result['source_positions']);R=rotation_sample(source,q)
                if not (len(q)==240 and q[0]==a and q[-1]==b-1 and np.all(np.diff(q)>0)):
                    raise ValueError('Fixed source and target Core contract violated')
                key=case['key']+'_'+method;arrays[key]=R;arrays[key+'_q']=q
                row['rotation_sha256']=array_hash(R);row['actual_music_reward']=score_map(case,phases,q,clock)
                row['metrics']=metrics.measure(R,source,J,parents,q,[a,b],source_events=case['phases'],phase_config=read('src/dunhuang/research/manifold_bridge_r47/config.json'),
                    audio_events=dict(beat=clock['beat_frames'],accent=clock['accent_frames'],accent_strength=clock['accent_strength']),audio_origin_frame=A,
                    optimized_event_ids=[p for p in case['phases'] if p['frame'] in result.get('source_phase_anchors',())],include_curves=True)
                own.append(dict(method=method,R=R,q=q,row=row))
            if result['status']!='LEGAL':own.append(dict(method=method,R=None,q=None,row=row))
            rows.append(row)
        display.append((case,own))
        print('R66B_CORE',len(display),len(pop['cases']),[(r['method'],r['row']['status']) for r in own],flush=True)
    save_arrays(target/'arrays.npz',arrays)
    write(target/'result.json',dict(population=dict(cases=len(pop['cases']),unique_source_intervals=pop['source_interval_count'],
        sources=pop['source_count'],music_tracks=pop['music_count']),methods=methods,records=rows,
        not_independent_case_count=True,artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE',can_route=False,
        inference='PROXY_REACHABILITY_AND_DYNAMIC_DIAGNOSTICS_ONLY; NO_QUALITY_WINNER',binding=binding()))
    from .viewer import build
    build(target,display,J,parents,music)
    manifest={str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    write(target/'artifacts_manifest.json',dict(files=manifest,binding=binding()))
    print('R66B_FINISHED',mode,len(rows),flush=True)

def verify(source,out,mode='main'):
    source=Path(source);old=read(source/'artifacts_manifest.json');now=binding()
    if old['binding']['code']!=now['code'] or old['binding']['dependencies']!=now['dependencies']:
        raise ValueError('R66-B science changed before independent replay')
    for name,value in old['files'].items():
        if sha(source/name)!=value:raise ValueError('Frozen output changed: '+name)
    compute(mode,out)
    differences=[]
    for name in old['files']:
        if name in ('registration.json','result.json'): # compare bindings with commit disclosed separately
            a,b=read(source/name),read(Path(out)/name)
            for item in (a,b):
                if 'binding' in item:item['binding']['commit']='PINNED_SCIENTIFIC_CODE_HASH'
            if a!=b:differences.append(name)
        elif sha(source/name)!=sha(Path(out)/name):differences.append(name)
    result=dict(scientific_outputs_exact=not differences,differences=differences,all_frozen_files_rebuilt=len(old['files']),
        source_manifest_sha256=sha(source/'artifacts_manifest.json'),binding=now,neural_training_steps=0,new_planner_searches=0,
        geometry_bridge_solves=0,source_rotation_interpolation_recomputed=True,warp_search_reexecuted=True)
    write(Path(out)/'independent.json',result)
    if differences:raise ValueError('Independent R66-B replay differed')
