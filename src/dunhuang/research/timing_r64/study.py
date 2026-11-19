"""Shared local geometry graph, effect-first viewing, same-graph objectives."""
import copy
from pathlib import Path
from collections import Counter
import numpy as np
from .common import ROOT,CFG,HERE,fresh,binding,read,write,sha,save_arrays,array_hash,check_seal
from .engine import evaluate,from_frozen
from .music import build as music_build,score as music_score
from .policies import balanced_cost,guard,selection,original_pass
from dunhuang.research.budget_r63.population import build as population_build,native
from dunhuang.research.budget_r63.engine import attach_retention
from dunhuang.research.budget_r63.study import seals as older_seals,summarize

PARENT=Path('outputs/budget_r63_01')

def seals():
    rows=older_seals()
    for r in read('reports/r63_release/scientific_files.json')['files']:
        p=PARENT/r['path'];assert sha(p)==r['sha256'],str(p)
        rows.append(dict(path=str(p),sha256=r['sha256']))
    for folder in ('budget_r63','music_structure_r56'):
        for p in sorted((HERE.parent/folder).glob('*')):
            if p.suffix in ('.py','.json','.html'):rows.append(dict(path=str(p),sha256=sha(p)))
    return rows

def registered_parent():
    records={};sources={}
    for mode in ('main','controls'):
        p=PARENT/mode
        source=dict(result_path=str(p/'result.json'),result_sha256=sha(p/'result.json'),
            arrays_path=str(p/'arrays.npz'),arrays_sha256=sha(p/'arrays.npz'))
        for row in read(p/'result.json')['records']:
            k=(row['case_key'],row['gap_frames'])
            if k not in records:
                records[k]=row;sources[k]=dict(source,record_key=row['key'])
    return records,sources

def relation(record,clock):
    """Describe geometry timing to acoustic section candidates; no correctness."""
    if not record.get('target_gap_interval'):return dict(status='NOT_EVALUABLE')
    a,b=record['target_gap_interval'];center=(a+b)/2
    near=min(clock['section_proxy_frames'],key=lambda x:abs(x-center)) if clock['section_proxy_frames'] else None
    if near is None:return dict(status='NO_SECTION_PROXY',independent_phrase_truth='UNKNOWN')
    kind='ENDS_BEFORE' if b<near-3 else ('ENDS_AT_WITHIN3_FRAMES' if abs(b-near)<=3 else ('CROSSES' if a<=near<b else 'STARTS_AFTER'))
    return dict(status='ACOUSTIC_PROXY_ONLY',relation=kind,section_proxy_frame=near,
        gap_absolute=[a,b],completion_minus_proxy_frames=b-near,independent_phrase_truth='UNKNOWN')

def add_summary(result):
    out=summarize(result['cases'],result['records'],result['selections'])
    index={r['key']:r for r in result['records']}
    for role,group in out.items():
        own={c['key'] for c in result['cases'] if c['role']==role}
        graph=[r for r in result['records'] if r['case_key'] in own]
        group['shared_allocation_candidates']=group.pop('distinct_allocation_recompilations')
        group['graph_G1_constructed_including_frozen']=group.pop('G1_solver_calls')
        group['production_new_G1_solver_calls']=sum(r['status']=='GEOMETRY_EVALUATED' and r['registered_frozen_r63_input'] is None for r in graph)
        group['production_frozen_cache_records']=sum(r['registered_frozen_r63_input'] is not None for r in graph)
        for policy,p in group['policies'].items():
            selected=[s for s in result['selections'] if s['role']==role and s['policy']==policy]
            p['offset_counts']=dict(Counter(str(s['timing_offset_frames']) for s in selected))
            p['music_primary_conditional_mean']=mean([index[s['record_key']]['music_reward']['primary_music_score'] for s in selected if s['record_key']])
            p['same_shared_graph']=True
            p['original_and_retention_passes']=sum(s['accepted'] and s['retention_guard_passed'] for s in selected)
    return out

def mean(values):
    x=[v for v in values if v is not None]
    return float(np.mean(x)) if x else None

def run_main(root=ROOT,render=True,replay=False):
    root=Path(root);out=fresh(root/'main');before=seals()
    cases,lib,J,parents,reference,population=population_build()
    old=read(PARENT/'main/result.json')
    assert cases==old['cases'] and population==old['population']
    clocks=music_build();write(out/'music_clocks.json',clocks)
    write(out/'population_seal.json',dict(cases=cases,population=population))
    parent,parent_sources=registered_parent()
    all_records=[];array_files={};reuse_registered=0
    for c in cases:
        folder=fresh(root/'graph'/c['key']);rows=[];arrays=dict(J=J,parents=parents)
        options=[(48,0)]+[(g,o) for g in CFG['durations'] for o in CFG['offsets'] if (g,o)!=(48,0)]
        for g,offset in options:
            prior=parent.get((c['key'],g)) if offset==0 else None
            registration=parent_sources.get((c['key'],g)) if offset==0 else None
            if prior is not None and not replay:
                with np.load(registration['arrays_path'],allow_pickle=False) as z:
                    r,a=from_frozen(c,g,prior,z)
            else:r,a=evaluate(c,g,offset,lib,J,parents,reference)
            if registration is not None:reuse_registered+=1
            r['registered_frozen_r63_input']=registration
            if g==48 and offset==0:
                baseline=r
                assert r['methods']['G1']['full_R_sha256']==parent[(c['key'],48)]['methods']['G1']['full_R_sha256']
            attach_retention(r,baseline)
            clock=clocks[int(c['songidx'])]
            r['audio_sha256']=c['audio_sha256']
            r['balanced_cost']=balanced_cost(r,reference)
            r['retention_guard_passed']=guard(r,baseline)
            r['music_reward']=music_score(r,clock)
            r['section_relation_proxy']=relation(r,clock)
            rows.append(r)
            arrays.update({k:v for k,v in a.items() if k.endswith('_G1')})
            print('R64',c['role'],c['case'],g,offset,r['status'],flush=True)
        rows.sort(key=lambda r:(r['gap_frames'],r['timing_offset_frames']))
        assert len(rows)==25 and len({r['execution_cache_key'] for r in rows})==25
        save_arrays(folder/'arrays.npz',arrays);write(folder/'result.json',dict(records=rows))
        array_files[c['key']]='graph/'+c['key']+'/arrays.npz'
        all_records.extend(rows)
    selections=[selection(c,[r for r in all_records if r['case_key']==c['key']],name)
        for c in cases for name in CFG['main_policies']]
    result=dict(binding=binding(),protocol=CFG,cases=cases,population=population,records=all_records,
        selections=selections,array_files=array_files,music_clocks=clocks,
        registered_parent_cache_candidates=reuse_registered,new_candidate_identities=len(all_records)-reuse_registered,
        summary={},parent_seal=before,parent_files_unchanged=check_seal(before),
        hierarchy_policies=CFG['unavailable_hierarchy_policies'],
        P0='AUDIO_CLOCK_VALIDATED; BEAT_ACCURACY_DOWNBEAT_PHRASE_NOT_EVALUABLE_WITHOUT_INDEPENDENT_LABELS',
        P2='NOT_ADMITTED_LOCAL_PROXY_DIAGNOSTIC_ONLY',
        neural_training_steps=0,optimizer_calls=0,full_song_runs=0,human_reviewed=False,Native24='NOT_EVALUABLE',can_route=False)
    result['summary']=add_summary(result);result=native(result);write(out/'result.json',result)
    if render:
        from .viewer import viewer
        viewer(root)
    return result

def run_controls(root=ROOT,gate=True,render=True):
    root=Path(root)
    if gate:
        receipt=read(root/'receipts/main_browser.json')
        assert receipt['passed'] and receipt['main_result_sha256']==sha(root/'main/result.json')
    out=fresh(root/'controls');result=read(root/'main/result.json')
    result=copy.deepcopy(result);result['binding']=binding();result['controls']=True
    result['main_result_sha256']=sha(root/'main/result.json')
    for r in result['records']:
        case=next(c for c in result['cases'] if c['key']==r['case_key'])
        clock=result['music_clocks'][str(case['songidx'])]
        r['music_reward_shuffled']=music_score(r,clock,shuffle=True)
    result['selections']=[selection(c,[r for r in result['records'] if r['case_key']==c['key']],name)
        for c in result['cases'] for name in CFG['control_policies']]
    result['summary']=add_summary(result)
    selected={(s['case_key'],s['policy']):s for s in result['selections']}
    delta=[]
    for c in result['cases']:
        rows=[r for r in result['records'] if r['case_key']==c['key']]
        real=[r['music_reward']['primary_music_score'] for r in rows]
        shifted=[r['music_reward_shuffled']['primary_music_score'] for r in rows]
        diffs=[abs(a-b) for a,b in zip(real,shifted) if a is not None and b is not None]
        delta.append(dict(case_key=c['key'],available_rewards=len(diffs),
            mean_absolute_reward_change=mean(diffs),
            selected_record_changed=selected[(c['key'],'LocalBeat')]['record_key']!=selected[(c['key'],'ShuffledBeat')]['record_key'],
            shift_preserves_potential_beat_periodicity=True))
    result['reward_control_effect']=delta
    result['new_geometry_solves']=0;result['shared_graph_exact']=True
    write(out/'result.json',result)
    if render:
        from .viewer import viewer
        viewer(root,controls=True)
    return result

def normalized(value):
    value=copy.deepcopy(value);value['binding']['commit']='CODE_HASHES_CHECKED'
    for r in value['records']:
        for name in ('prior_bridge_or_core_reused','execution_cache_lookup','independent_e1w_recompile'):
            r.pop(name,None)
    value.pop('main_result_sha256',None)
    return value

def verify():
    out=fresh(ROOT/'independent');rebuilt=out/'fresh'
    main=read(ROOT/'main/result.json');assert binding()['code']==main['binding']['code']
    actual=run_main(rebuilt,render=False,replay=True)
    assert normalized(main)==normalized(actual)
    for case,path in main['array_files'].items():assert sha(ROOT/path)==sha(rebuilt/path),case
    assert sha(ROOT/'main/music_clocks.json')==sha(rebuilt/'main/music_clocks.json')
    controls=read(ROOT/'controls/result.json');new=run_controls(rebuilt,gate=False,render=False)
    assert normalized(controls)==normalized(new)
    inherited=read('outputs/phase_integrity_r60_02/population/result.json')['parent_seal']
    result=dict(binding=binding(),main_result_sha256=sha(ROOT/'main/result.json'),
        controls_result_sha256=sha(ROOT/'controls/result.json'),scientific_values_exact=True,arrays_bytes_exact=True,
        full_candidates_freshly_rebuilt=len(actual['records']),production_frozen_cache_reads=main['registered_parent_cache_candidates'],
        allowed_replay_differences=['binding.commit','cache_execution_flags','controls.main_result_sha256'],
        independent_music_clocks_exact=True,frozen_parent_files_unchanged=check_seal(inherited),
        optimizer_calls=0,neural_training_steps=0,full_song_runs=0,Native24='NOT_EVALUABLE',can_route=False)
    write(out/'result.json',result);print('R64_INDEPENDENT_EXACT',flush=True)
    return result
