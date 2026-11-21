"""Actual whole-song execution of frozen P1 witnesses, then fixed controls."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .common import *
from .assembly import assemble,changes
from .metrics import evaluate,music_diagnostic
from dunhuang.research.warp_coupling_r53.population import Population
from dunhuang.research.context_feasibility_p1.geometry import solve
from dunhuang.research.duration_r62.cross_source import make_observed,measure

def roster(saved,stage):
    failures=[r for r in saved if r['case'] and r['case']['role']=='failure']
    assert len(failures)==14
    result=[]
    if stage=='main':
        for r in failures:result.append(dict(song=r['song'],case=r['case'],method='P1_SAVED_STRONG_CENTER',ids=r['comparisons']['P1']['strong_sequential_path']))
        good=[r for r in result if r['ids']];assert len(good)==10 and len({tuple(r['ids']) for r in good})==10
    else:
        for r in saved:
            if r['case'] is None:
                for arm in ('P0','P1'):result.append(dict(song=r['song'],case=None,method=arm+'_SAVED_STRONG_UNCONSTRAINED',ids=r['comparisons'][arm]['strong_sequential_path']))
        for r in failures:result.append(dict(song=r['song'],case=r['case'],method='P1_SAVED_R58_ADAPTER_CENTER',ids=r['comparisons']['P1']['R58_adapter_path']))
    return result

def simple_node(row):
    names=('uid','source_uid','event_uid','source_interval','target_interval','time_warp','q_sha256','rotation_sha256','source_R_sha256','origin')
    return {k:row[k] for k in names}

def run(out,stage):
    if stage=='controls':
        receipt=read(ROOT/'receipts/main_browser.json')
        assert receipt['passed'] and receipt['main_result_sha256']==sha(ROOT/'main/result.json')
    out=fresh(out);p,z,rows,slots,old,saved,saved_paths=inputs();pop=Population();J,parents=z['J'],z['parents']
    assert np.array_equal(J,pop.net.joints) and np.array_equal(parents,pop.net.parents)
    tasks=roster(saved_paths,stage);write(out/'registration.json',dict(tasks=tasks,protocol=CFG,parent_manifest_hashes=CFG['parent_manifests'],science=binding(),paths_reselected=False))
    arrays=dict(J=J,parents=parents);bodies={};clocks={};required={}
    graph={(e['left'],e['right']):e for e in saved['graph']}
    for task in tasks:
        if task['ids'] is None:continue
        s=task['song'];assert len(task['ids'])==13
        for k,u in enumerate(task['ids']):
            assert rows[u]['target_interval']==slots[s,k]['target_interval']
            if u not in bodies:bodies[u],clocks[u]=reconstruct(rows[u],z,pop.lib.rotations[rows[u]['source_uid']])
        for u,v in zip(task['ids'],task['ids'][1:]):
            e=graph.get((u,v));assert e and e['measured'] and e['eligible'] and e['policy_admissible']
            key=boundary_key(bodies[u],bodies[v],J,parents,pop.reference);assert key==e['boundary_key']
            if key in required:assert required[key]['numeric']['bridge_sha256']==e['numeric']['bridge_sha256']
            required[key]=e
    assert len(required)<=CFG['max_distinct_bridge_materializations_per_stage']
    bridges={};materialized=[]
    jobs=[(key,bodies[e['left']],bodies[e['right']],J,parents,pop.reference) for key,e in sorted(required.items())]
    with ProcessPoolExecutor(max_workers=CFG['workers']) as executor:
        for i,(key,numeric,Q) in enumerate(executor.map(solve,jobs,chunksize=2)):
            prior=required[key]['numeric'];assert Q is not None and numeric['passes_original_screen']
            assert numeric['bridge_sha256']==prior['bridge_sha256'] and numeric['metrics']==prior['metrics']
            bridges[key]=Q;arrays['bridge_'+key]=Q
            materialized.append(dict(boundary_key=key,bridge_sha256=array_hash(Q),original_numeric_metrics_exact=True,original_screen_pass=True))
            if i%25==0:print('P2_MATERIALIZE',i+1,len(jobs),flush=True)
    records=[];aliases={};source_units={};catalogs={s['song']:read(f"outputs/music_events_01/main/{s['song']}/catalog.json") for s in p['songs']}
    def store(key,R,meta,kind,owner,song):
        P,curves,metrics=evaluate(R,J,parents,kind,owner)
        arrays[key+'_R']=R;arrays[key+'_P']=P.astype('<f4');arrays[key+'_kind']=kind;arrays[key+'_owner']=owner
        for name,x in curves.items():arrays[key+'_'+name]=x
        seams=[]
        for region in meta['regions']:
            if region['kind']!='bridge':continue
            a,b=region['interval'];obs,gap=make_observed(R[:a],R[b:],48,context=4)
            actual=np.concatenate((R[a-4:a],R[a:b],R[b:b+4]));m=measure(actual,obs,J,parents,gap,pop.reference)
            seams.append(dict(interval=[a,b],metrics=m,global_D3_crossing_stencils=[list(range(i,i+4)) for cut in (a,b) for i in range(cut-3,cut)],
                              original_frozen_gate_pass=bool(m['passes_original_screen'])))
        return dict(key=key,song=song,frames=len(R),**meta,metrics=metrics,seams=seams,music=music_diagnostic(P,catalogs[song]),source_and_physics_scope='BODY22_ROTATIONS; WORLD_ROOT_CONTACT_NATIVE24_UNKNOWN')
    for task in tasks:
        ids=task['ids'];entry=dict(task)
        if ids is None:
            entry.update(status='NO_COMPLETE_PATH_IN_FROZEN_MEASURED_GRAPH',motion_key=None)
            entry['local_recovered']=next(r['witness']['recovered'] for r in saved['cases'] if r['case']==task['case']) if task['case'] else None
        else:
            s=task['song'];key='motion_'+identity(dict(song=s,ids=ids));entry.update(status='ACTUAL_WHOLE_SONG_ASSEMBLED',motion_key=key)
            if key not in aliases:
                R,meta,kind,owner=assemble(ids,rows,bodies,graph,bridges,p['songs'][s]['frames'])
                q=np.full(len(R),np.nan)
                for u in ids:
                    a,b=rows[u]['target_interval'];q[a:b]=clocks[u]
                    event=rows[u]['event_uid']
                    if event not in source_units:
                        lo,hi=rows[u]['source_interval'];source_units[event]=simple_node(rows[u]);arrays['source_'+identity(event)+'_P']=relative_fk(pop.lib.rotations[rows[u]['source_uid']][lo:hi],J,parents).astype('<f4')
                arrays[key+'_source_frame']=q;meta.update(nodes=[simple_node(rows[u]) for u in ids],source_core_exact=True,all_bridge_hashes_exact=True,saved_path_method=task['method'])
                record=store(key,R,meta,kind,owner,s);assert all(v['original_frozen_gate_pass'] for v in record['seams']);aliases[key]=record
            entry['rotation_sha256']=aliases[key]['rotation_sha256']
        records.append(entry)
    baselines=[]
    for song in p['songs']:
        s=song['song'];R=z[f'song{s}_baseline_R'];assert array_hash(R)==song['baseline_rotation_sha256']
        assert np.array_equal(relative_fk(R,J,parents).astype('<f4'),z[f'song{s}_baseline_P'])
        kind=np.full(len(R),-1,np.int16);owner=np.full(len(R),-1,np.int16);regions=[];counter={0:0,1:0}
        for reg in song['regions']:
            k=0 if reg['kind']=='core' else 1;a,b=reg['interval'];assert np.all(kind[a:b]==-1);kind[a:b]=k;owner[a:b]=counter[k];regions.append(dict(kind='core' if k==0 else 'bridge',index=counter[k],interval=[a,b]));counter[k]+=1
        assert np.all(kind>=0)
        nodes=[simple_node(rows[slot['selected_baseline_uid']]) for slot in song['slots']]
        for node in nodes:
            u=node['uid'];body,q=reconstruct(rows[u],z,pop.lib.rotations[node['source_uid']]);a,b=node['target_interval'];assert np.array_equal(body,R[a:b])
            event=node['event_uid']
            if event not in source_units:
                lo,hi=node['source_interval'];source_units[event]=node;arrays['source_'+identity(event)+'_P']=relative_fk(pop.lib.rotations[node['source_uid']][lo:hi],J,parents).astype('<f4')
        meta=dict(regions=regions,nodes=nodes,rotation_sha256=array_hash(R),coverage_exact_once=True,historical_B0_exact=True,source_core_exact=True,bridge_methods=[e['selected'] for e in song['baseline_connections']])
        baselines.append(store('b0_song'+str(s),R,meta,kind,owner,s))
    result=dict(task=CFG['task'],stage=stage,protocol=CFG,science=binding(),source_binding=pop.binding,tasks=records,motions=list(aliases.values()),baselines=baselines,source_units=source_units,
                counts=dict(registered_tasks=len(records),assembled_display_positions=sum(r['motion_key'] is not None for r in records),missing=sum(r['motion_key'] is None for r in records),
                    distinct_node_paths=len({r['path_identity'] for r in aliases.values()}),distinct_song_path_executions=len(aliases),distinct_rotation_arrays=len({r['rotation_sha256'] for r in aliases.values()}),bridge_materializations=len(materialized)),
                materialized=materialized,new_unknown_edges_solved=0,planner_calls=0,training_updates=0,can_route=False,Native24='NOT_EVALUABLE',independent_artistic_quality='NOT_EVALUABLE')
    write(out/'result.json',result);save_arrays(out/'arrays.npz',arrays)
    if stage=='main':
        from .view import build
        build(out,result,arrays,p)
    else:
        original=read(ROOT/'main/result.json');compare=[]
        control_tasks={ (r['song'],r['method'],None if r['case'] is None else identity(r['case'])):r for r in records }
        for r in original['tasks']:
            if r['motion_key'] is None:continue
            s=r['song'];candidates=[('EXACT_HISTORICAL_B0',[slot['selected_baseline_uid'] for slot in p['songs'][s]['slots']])]
            for arm in ('P0','P1'):candidates.append((arm+'_UNCONSTRAINED_STRONG',control_tasks[s,arm+'_SAVED_STRONG_UNCONSTRAINED',None]['ids']))
            candidates.append(('P1_R58_ADAPTER_SAME_CENTER',control_tasks[s,'P1_SAVED_R58_ADAPTER_CENTER',identity(r['case'])]['ids']))
            compare.append(dict(case=r['case'],main_key=r['motion_key'],main_equals_saved_constrained_strong=True,
                comparisons=[dict(method=name,available=ids is not None,differences=None if ids is None else changes(r['ids'],ids,rows)) for name,ids in candidates],
                difference_counts_are_descriptive_not_isolated_causal_effects=True))
        write(out/'comparisons.json',compare)
    seal(out);print('P2_COMPLETE',stage,result['counts'],flush=True)

def replay(stage):
    from dunhuang.research.source_tasks.common import forbid_optimizers
    forbid_optimizers();old=ROOT/stage;check_seal(old);new=ROOT/('independent_'+stage);run(new,stage)
    assert read(old/'manifest.json')==read(new/'manifest.json'),'Fresh process scientific files differ'
    write(ROOT/'receipts'/('independent_'+stage+'.json'),dict(status='PASS',fresh_process=True,all_scientific_arrays_and_viewer_exact=True,optimizer_calls=0,source_manifest_sha256=sha(old/'manifest.json'),can_route=False))

if __name__=='__main__':
    import sys
    stage=sys.argv[1]
    if stage in ('main','controls'):run(ROOT/stage,stage)
    elif stage in ('replay-main','replay-controls'):replay(stage[7:])
    else:raise ValueError('Unregistered stage')
