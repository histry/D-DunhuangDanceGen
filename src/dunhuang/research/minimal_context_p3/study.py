from .common import *
from .optimal import constrained,solve,change_counts,verify
from .execution import execute,motion_key

def run(out,stage):
    if stage=='controls':
        receipt=read(ROOT/'receipts/main_browser.json');assert receipt['passed'] and receipt['main_result_sha256']==sha(ROOT/'main/result.json')
    p,z,rows,slots,graph,parent=inputs();out=fresh(out);objective='source_warp' if stage=='main' else 'source_only'
    tasks=parent['tasks'];assert len(tasks)==14
    write(out/'registration.json',dict(protocol=CFG,objective=objective,parent_P2_tasks=tasks,science=binding(),all_failed_cases_retained=True,graph_not_expanded=True))
    plans=[];certificates=[];attribution=[];used_edges={}
    parent_index=None
    if stage=='controls':parent_index={identity(r['case']):r for r in read(ROOT/'main/minimal_edit_paths.json')}
    for old in tasks:
        case=old['case'];s,k=case['song'],case['slot'];baseline=[slot['selected_baseline_uid'] for slot in p['songs'][s]['slots']]
        layers=constrained([[r['uid'] for r in slot['candidates']] for slot in p['songs'][s]['slots']],k,case['center_uids'])
        assert {rows[u]['event_uid'] for u in layers[k]}=={case['event_uid']}
        edges=eligible_edges(graph,s,layers);cert=solve(layers,edges,rows,baseline,objective);assert verify(cert,layers,edges,rows,baseline)
        ids=cert['ids'];assert (ids is not None)==(old['ids'] is not None),'Same frozen graph must preserve existence'
        selected=change_counts(ids,baseline,rows,k);strong=change_counts(old['ids'],baseline,rows,k)
        if ids is not None:
            assert tuple(cert['cost'])<=tuple([strong['source_units'],strong['same_unit_clocks']][:len(cert['cost'])])
            if parent_index:assert selected['source_units']==parent_index[identity(case)]['changes_vs_B0']['source_units']
            for u,v in zip(ids,ids[1:]):
                if (u,v) in used_edges:assert used_edges[u,v]['numeric']['bridge_sha256']==edges[u,v]['numeric']['bridge_sha256']
                used_edges[u,v]=edges[u,v]
        plan=dict(song=s,case=case,ids=ids,objective=objective,cost=cert['cost'],status=cert['status'],motion_key=None if ids is None else motion_key(s,ids),
            strong_key=old['motion_key'],strong_ids=old['ids'],baseline_key='b0_song'+str(s),changes_vs_B0=selected,strong_changes_vs_B0=strong,
            source_reduction_vs_saved_strong=None if ids is None else strong['source_units']-selected['source_units'],
            local_recovered=old.get('local_recovered'),center_constraint_unchanged=True,can_route=False)
        plans.append(plan);certificates.append(dict(case=case,certificate=cert));attribution.append(dict(case=case,min_edit=selected,saved_strong=strong,
            source_reduction=plan['source_reduction_vs_saved_strong'],method_scope='FINITE_MEASURED_GRAPH_MINIMUM; NOT_INDEPENDENT_MUSIC_OR_ARTISTIC_PREFERENCE'))
    assert sum(r['ids'] is not None for r in plans)==10
    write(out/'minimal_edit_paths.json',plans);write(out/'exact_measured_graph_optima.json',certificates);write(out/'source_warp_change_attribution.json',attribution)
    arrays,motions,metrics,sources,materialized,source_binding=execute(plans,p,z,rows,used_edges)
    result=dict(task=CFG['task'],stage=stage,objective=objective,protocol=CFG,science=binding(),tasks=plans,motions=list(motions.values()),source_units=sources,source_binding=source_binding,
        materialized=materialized,counts=dict(registered=14,actual=10,missing=4,distinct_node_paths=len(motions),distinct_rotation_arrays=len({r['rotation_sha256'] for r in motions.values()}),
            distinct_FK_arrays=len({r['dynamic']['FK_positions_sha256'] for r in metrics.values()}),bridge_materializations=len(materialized),
            fewer_source_changes_than_strong=sum((r['source_reduction_vs_saved_strong'] or 0)>0 for r in plans)),
        new_unknown_edges_solved=0,new_clocks=0,new_candidates=0,training_updates=0,can_route=False,Native24='NOT_EVALUABLE',independent_artistic_quality='NOT_EVALUABLE')
    write(out/'full_motion_metrics.json',metrics);write(out/'result.json',result);save_arrays(out/'arrays.npz',arrays)
    if stage=='main':
        from .view import build
        build(out,result,metrics,arrays)
    else:
        primary=read(ROOT/'main/result.json');previous={identity(r['case']):r for r in primary['tasks']};comparisons=[]
        for r in plans:
            a=previous[identity(r['case'])];comparisons.append(dict(case=r['case'],primary=a['changes_vs_B0'],source_only=r['changes_vs_B0'],
                both_graph_existence_equal=(a['ids'] is None)==(r['ids'] is None),
                primary_same_source_clock_count_no_worse=None if r['ids'] is None else a['changes_vs_B0']['same_unit_clocks']<=r['changes_vs_B0']['same_unit_clocks']))
        assert all(r['primary_same_source_clock_count_no_worse'] is not False for r in comparisons)
        write(out/'source_only_comparisons.json',comparisons)
    seal(out);print('P3_COMPLETE',stage,result['counts'],flush=True)

def replay(stage):
    from dunhuang.research.source_tasks.common import forbid_optimizers
    forbid_optimizers();old=ROOT/stage;check_seal(old);new=ROOT/('independent_'+stage);run(new,stage)
    assert read(old/'manifest.json')==read(new/'manifest.json'),'Independent DP certificate or execution differs'
    write(ROOT/'receipts'/('independent_'+stage+'.json'),dict(status='PASS',fresh_process=True,all_certificates_rotations_FK_bridges_metrics_and_viewer_exact=True,
        optimizer_calls=0,new_unknown_edges_solved=0,source_manifest_sha256=sha(old/'manifest.json'),can_route=False))

if __name__=='__main__':
    import sys
    phase=sys.argv[1]
    if phase in ('main','controls'):run(ROOT/phase,phase)
    elif phase in ('replay-main','replay-controls'):replay(phase[7:])
    else:raise ValueError('Unregistered phase')
