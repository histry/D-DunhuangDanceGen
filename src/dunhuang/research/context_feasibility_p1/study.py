from concurrent.futures import ProcessPoolExecutor
import numpy as np
from dunhuang.research.warp_coupling_r53.population import Population
from dunhuang.research.complete_r36.core import relative_fk
from .common import *
from .geometry import solve

def register(p,c,rows,slots):
    failures={(s['song'],k['slot'],event) for s in p['songs'] for k in s['slots'] for event in {r['event_uid'] for r in k['candidates']}
              if not any(all(x['passes_original_screen'] for x in rr['connections'][i]) for rr in c['slots'] if rr['song']==s['song'] and rr['slot']==k['slot'] for i,r in enumerate(k['candidates']) if r['event_uid']==event)}
    actual={(r['song'],r['slot'],r['event_uid']) for r in CFG['cases'] if r['role']=='failure'};assert failures==actual and len(actual)==14
    requests={}
    for case in CFG['cases']:
        s,k=case['song'],case['slot'];centers=[r['uid'] for r in slots[s,k]['candidates'] if r['event_uid']==case['event_uid']];assert centers==case['center_uids']
        for u in centers:
            for side,t in [('left',k-1),('right',k+1)]:
                for v in neighbors(slots,s,t):
                    L,R=(v,u) if side=='left' else (u,v);requests[L,R]=dict(song=s,boundary=min(k,t),left=L,right=R)
    return dict(protocol=CFG,science=binding(),candidate_count=len(rows),source_population='SAME622_P0_EXECUTED_CANDIDATES; NEW_CLOCKS_ZERO',
                requests=[requests[key] for key in sorted(requests)],new_training_updates=0,can_route=False)

def seed(p,c,rows,slots,J,parents,reference,R):
    graph={};cache={}
    for r in c['slots']:
        s,k=r['song'],r['slot']
        for u,values in zip(r['candidate_uids'],r['connections']):
            for receipt in values:
                v=slots[s,k+(-1 if receipt['side']=='left' else 1)]['selected_baseline_uid'];L,V=(v,u) if receipt['side']=='left' else (u,v)
                assert array_hash(R[L])==receipt['left_rotation_sha256'] and array_hash(R[V])==receipt['right_rotation_sha256']
                key=boundary_key(R[L],R[V],J,parents,reference);numeric={n:value for n,value in receipt.items() if n not in ('side','cache_key','left_rotation_sha256','right_rotation_sha256')}
                if key in cache:
                    assert cache[key]['bridge_sha256']==numeric['bridge_sha256'] and cache[key]['passes_original_screen']==numeric['passes_original_screen']
                else:cache[key]=numeric
                graph[L,V]=dict(song=s,boundary=min(k,k+(-1 if receipt['side']=='left' else 1)),left=L,right=V,boundary_key=key,numeric=numeric,
                               measured=True,eligible=bool(numeric['passes_original_screen'] and admissible(rows[L],rows[V])),policy_admissible=admissible(rows[L],rows[V]),origin='P0_ACTUAL_ARRAY_BOUND_LOCAL_RESULT')
    return graph,cache

def witnesses(case,rows,slots,graph):
    s,k=case['song'],case['slot'];base={side:(slots[s,t]['selected_baseline_uid'] if (s,t) in slots else None) for side,t in [('left',k-1),('right',k+1)]};choices=[]
    centers=sorted(case['center_uids'],key=lambda u:(rows[u]['origin']!='E1W',u))
    for rank,u in enumerate(centers):
        selected={};extra=[]
        for side,t in [('left',k-1),('right',k+1)]:
            n=neighbors(slots,s,t)
            good=[v for v in n if graph.get((v,u) if side=='left' else (u,v),{}).get('eligible')]
            selected[side]=good[0] if good else None
            if (s,t) in slots and not good:break
        else:
            changes=sum(v is not None and v!=base[side] for side,v in selected.items());choices.append((changes,rank,selected.get('left') or '',u,selected.get('right') or ''))
    if not choices:return dict(recovered=False,reason='NO_COMMON_CENTER_WITH_ALL_SIDES_PASSING_IN_REGISTERED_DOMAIN',baseline_neighbors=base,center_uid=centers[0],neighbors=base)
    change,_,L,u,V=min(choices)
    return dict(recovered=True,changed_neighbors=change,center_uid=u,neighbors=dict(left=L or None,right=V or None),baseline_neighbors=base,
                reason='MEASURED_LOCAL_WITNESS_NOT_COMPLETE_SONG',both_panels_same_center_clock=True)

def main(out):
    out=fresh(out);p,c,z,rows,slots,R=load();pop=Population();J,parents=z['J'],z['parents'];reference=pop.reference
    registration=register(p,c,rows,slots);write(out/'frozen_population.json',registration)
    graph,cache=seed(p,c,rows,slots,J,parents,reference,R);before=[dict(e) for e in graph.values()];requests=registration['requests'];jobs={};hits=0;policy_excluded=0
    for r in requests:
        L,V=r['left'],r['right']
        if not admissible(rows[L],rows[V]):policy_excluded+=1;continue
        key=boundary_key(R[L],R[V],J,parents,reference)
        if key in cache:hits+=1
        elif key not in jobs:jobs[key]=(key,R[L],R[V],J,parents,reference)
        else:hits+=1
    assert len(jobs)<=CFG['max_geometry_calls'];bridges={};new_keys=set(jobs)
    with ProcessPoolExecutor(max_workers=CFG['workers']) as executor:
        for n,(key,numeric,Q) in enumerate(executor.map(solve,[jobs[k] for k in sorted(jobs)],chunksize=4)):
            cache[key]=numeric
            if Q is not None:bridges[key]=Q
            if n%100==0:print('P1_GEOMETRY',n+1,len(jobs),flush=True)
    trials=sum(len(cache[k]['proof']['guards']['G1']['trials']) for k in new_keys);assert trials<=CFG['max_chart_trials']
    for r in requests:
        L,V=r['left'],r['right'];key=boundary_key(R[L],R[V],J,parents,reference);policy=admissible(rows[L],rows[V])
        if not policy:
            graph[L,V]=dict(**r,measured=(L,V) in graph,numeric=graph.get((L,V),{}).get('numeric'),eligible=False,policy_admissible=False,origin='FROZEN_R58_SAME_SOURCE_EXCLUSION',boundary_key=key);continue
        numeric=cache[key];graph[L,V]=dict(**r,boundary_key=key,numeric=numeric,measured=True,eligible=bool(numeric['passes_original_screen']),policy_admissible=True,
                                        origin='NEW_ACTUAL_G1' if key in new_keys else 'EXACT_BOUNDARY_CACHE_REUSE')
    materializations=0;arrays=dict(J=J,parents=parents);display=[];case_results=[]
    def bridge(L,V):
        nonlocal materializations
        key=boundary_key(R[L],R[V],J,parents,reference)
        if key not in bridges:
            if materializations>=CFG['max_materializations']:raise ValueError('Materialization budget exhausted')
            _,actual,Q=solve((key,R[L],R[V],J,parents,reference));materializations+=1
            prior=cache[key]
            assert actual['bridge_sha256']==prior['bridge_sha256'] and actual['passes_original_screen']==prior['passes_original_screen']
            if Q is not None:bridges[key]=Q
        return bridges.get(key)
    def clip(center,near):
        ids=[u for u in (near['left'],center,near['right']) if u is not None];chunks=[]
        for k,u in enumerate(ids):
            chunks.append(R[u])
            if k+1<len(ids):
                Q=bridge(u,ids[k+1]);chunks.append(np.full((48,22,3,3),np.nan) if Q is None else Q)
        return np.concatenate(chunks),ids
    for index,case in enumerate(CFG['cases']):
        w=witnesses(case,rows,slots,graph);u=w['center_uid'];old,old_ids=clip(u,w['baseline_neighbors']);new,ids=clip(u,w['neighbors']) if w['recovered'] else (None,[])
        key='case'+str(index);arrays[key+'_old']=old
        if new is not None:arrays[key+'_new']=new
        center_offset=(len(R[old_ids[0]])+48) if w['baseline_neighbors']['left'] else 0
        assert np.array_equal(old[center_offset:center_offset+len(R[u])],R[u])
        if new is not None:assert np.array_equal(new[center_offset:center_offset+len(R[u])],R[u]) and len(new)==len(old)
        row=dict(index=index,case=case,witness=w,origin=rows[old_ids[0]]['target_interval'][0],frames=len(old),center_offset=center_offset,center_frames=len(R[u]),
                 center_rotation_sha256=array_hash(R[u]),old_rotation_sha256=array_hash(old),new_rotation_sha256=None if new is None else array_hash(new),source_preserved_exact=True,
                 old_edges=[graph[a,b] for a,b in zip(old_ids,old_ids[1:])],new_edges=[graph[a,b] for a,b in zip(ids,ids[1:])],can_route=False)
        case_results.append(row)
    # Full explicit UNKNOWN coverage for all pairs; no outcomes outside requests inferred.
    coverage=[]
    for s in p['songs']:
        for k in range(12):
            counts=dict(total=0,policy_excluded=0,measured_pass=0,measured_fail=0,unknown=0)
            for a in slots[s['song'],k]['candidates']:
                for b in slots[s['song'],k+1]['candidates']:
                    counts['total']+=1
                    if not admissible(a,b):counts['policy_excluded']+=1
                    elif (a['uid'],b['uid']) not in graph:counts['unknown']+=1
                    elif graph[a['uid'],b['uid']]['eligible']:counts['measured_pass']+=1
                    else:counts['measured_fail']+=1
            coverage.append(dict(song=s['song'],boundary=k,**counts))
    result=dict(cases=case_results,graph=list(graph.values()),P0_graph=before,coverage=coverage,science=binding(),parent_preview_manifest_sha256=sha(P0/'preview/manifest.json'),parent_controls_manifest_sha256=sha(P0/'controls/manifest.json'),
                geometry_calls=len(jobs),chart_trials=trials,boundary_cache_reuses=hits,policy_excluded_requests=policy_excluded,materialization_calls=materializations,geometry_requests=len(requests),
                source_observed='P0_SAME622_ROTATIONS_VERIFIED',candidate_pool_changed=False,new_training_updates=0,can_route=False,native24='NOT_EVALUABLE')
    save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
    from .view import build
    build(out,arrays,result,p);seal(out);print('P1_MAIN_PASS',len(jobs),materializations,flush=True)

def controls(out):
    receipt=read(ROOT/'receipts/main_browser.json');assert receipt['passed'] and receipt['main_result_sha256']==sha(ROOT/'main/result.json')
    out=fresh(out);m=check_seal(ROOT/'main');a=read(ROOT/'main/result.json');assert a['science']==binding();p,c,z,rows,slots,R=load();pop=Population()
    from .graph import inspect
    reference=pop.reference;layers={s['song']:[slot['candidates'] for slot in s['slots']] for s in p['songs']}
    for s in layers:
        energy=np.asarray(pop.tracks[s]['structure']['energy_percentile'])
        N=next(song['frames'] for song in p['songs'] if song['song']==s)
        if len(energy)<N:assert len(energy)==N-1;energy=np.r_[energy,energy[-1]]
        for layer in layers[s]:
            for r in layer:
                from dunhuang.research.complete_r36.core import relative_fk
                P=relative_fk(R[r['uid']],z['J'],z['parents']);activity=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean()
                A,B=r['target_interval'];values=energy[A:min(B,len(energy))];r['heuristic']=-float(abs(np.clip(activity/2,0,1)-values.mean()))
    paths=[]
    for song in p['songs']:
        s=song['song'];all_cases=[None]+[case for case in CFG['cases'] if case['song']==s]
        for case in all_cases:
            nodes=[[dict(r) for r in layer] for layer in layers[s]]
            if case:nodes[case['slot']]=[r for r in nodes[case['slot']] if r['uid'] in case['center_uids']]
            compared={}
            for name,graph in [('P0',a['P0_graph']),('P1',a['graph'])]:
                edges=[e for e in graph if e['song']==s and e['eligible']];q=inspect(nodes,edges,reference)
                for method in ('strong_sequential_path','R58_adapter_path'):
                    ids=q[method]
                    if ids:
                        assert len(ids)==13 and all(any(e['left']==u and e['right']==v and e['measured'] and e['eligible'] for e in edges) for u,v in zip(ids,ids[1:]))
                compared[name]=q
            paths.append(dict(song=s,case=case,comparisons=compared,coverage_exact=True,path_is_not_new_assembled_trajectory=True,Native24='NOT_EVALUABLE'))
    selections=read('reports/paper1_sdmr_integration_p0_release/selections.json');music=[]
    for r in selections:
        values=r['ties'];music.append(dict(song=r['song'],slot=r['slot'],event_uid=r['event_uid'],ties=values,
               actual_wrong_different=None if not values['Single64'] or not values['WrongMusic'] else set(values['Single64'])!=set(values['WrongMusic']),
               actual_zero_different=None if not values['Single64'] or not values['MusicZero'] else set(values['Single64'])!=set(values['MusicZero']),
               E1W=values['E1W'],GTW=values['GTWLocal'],quality_gain='NOT_EVALUABLE',input_rotations_unchanged=True))
    write(out/'complete_path_witness.json',paths);write(out/'sdmr_music_controls.json',music);write(out/'qualification.json',dict(no_new_scoring_or_training=True,scope='MEASURED_GRAPH_PREFLIGHT_AND_PAIRED_CONDITION_RESPONSE_NOT_ARTISTIC_ACCURACY',can_route=False,Native24='NOT_EVALUABLE'));seal(out)

def replay(stage):
    from dunhuang.research.source_tasks.common import forbid_optimizers
    forbid_optimizers();old=ROOT/stage;new=ROOT/('independent_'+stage)
    (main if stage=='main' else controls)(new)
    assert read(old/'manifest.json')==read(new/'manifest.json')
    write(ROOT/'receipts'/('independent_'+stage+'.json'),dict(status='PASS',fresh_process=True,all_scientific_files_and_viewer_exact=True,optimizer_calls=0,source_manifest_sha256=sha(old/'manifest.json'),can_route=False))

if __name__=='__main__':
    import sys
    stage=sys.argv[1]
    if stage.startswith('replay-'):replay(stage[7:])
    elif stage=='main':main(ROOT/'main')
    elif stage=='controls':controls(ROOT/'controls')
    else:raise ValueError('Unknown frozen stage')
