"""Scalar evidence only. Measured edges, UNKNOWN frontier, frozen score controls."""
from collections import Counter
from dunhuang.research.context_feasibility_p1.common import *
from dunhuang.research.context_feasibility_p1.study import witnesses

def run():
    check_seal(ROOT/'main');check_seal(ROOT/'controls');a=read(ROOT/'main/result.json');p=read(P0/'preview/result.json');c=read(ROOT/'controls/complete_path_witness.json')
    for stage in ('main','controls'):assert read(ROOT/'receipts'/('independent_'+stage+'.json'))['status']=='PASS'
    out=fresh('reports/paper1_context_feasibility_p1_release');rows={r['uid']:r for s in p['songs'] for k in s['slots'] for r in k['candidates']};slots={(s['song'],k['slot']):k for s in p['songs'] for k in s['slots']}
    old={(r['left'],r['right']):r for r in a['P0_graph']};new={(r['left'],r['right']):r for r in a['graph']};rescues=[];counts=Counter()
    for r in a['cases']:
        prior=witnesses(r['case'],rows,slots,old);current=r['witness'];role=r['case']['role'];counts[role+'_registered']+=1;counts[role+'_P0_recombined_witness']+=prior['recovered'];counts[role+'_P1_witness']+=current['recovered']
        needed=[e for e in r['new_edges'] if (e['left'],e['right']) not in old or not old[e['left'],e['right']]['eligible']]
        rescued=bool(current['recovered'] and not prior['recovered']);counts[role+'_newly_witnessed_beyond_P0_graph']+=rescued
        rescues.append(dict(r,P0_recombined_witness=prior,newly_witnessed_beyond_P0_measured_graph=rescued,display_edges_not_eligible_in_P0=len(needed),quality_gain='NOT_EVALUABLE'))
    unknown=[]
    for song in p['songs']:
        s=song['song']
        for k in range(12):
            for x in slots[s,k]['candidates']:
                for y in slots[s,k+1]['candidates']:
                    if admissible(x,y) and (x['uid'],y['uid']) not in new:unknown.append(dict(song=s,boundary=k,left=x['uid'],right=y['uid'],status='UNKNOWN_NOT_MEASURED'))
    for r in c:
        r['registered_timeline_coverage_exact']=r.pop('coverage_exact');r['actual_full_trajectory_assembled']=False
    path_counts={}
    for arm in ('P0','P1'):
        path_counts[arm]={}
        for kind in ('unconstrained','failure','control'):
            selected=[r for r in c if ('unconstrained' if r['case'] is None else r['case']['role'])==kind]
            path_counts[arm][kind]=dict(tasks=len(selected),exact_suffix_paths=sum(r['comparisons'][arm]['exact_suffix_has_path'] for r in selected),
                strong_sequential=sum(r['comparisons'][arm]['strong_sequential_path'] is not None for r in selected),R58_adapter=sum(r['comparisons'][arm]['R58_adapter_path'] is not None for r in selected))
    write(out/'frozen_population.json',read(ROOT/'main/frozen_population.json'))
    write(out/'context_connection_graph.json',dict(edges=a['graph'],P0_edges=a['P0_graph'],coverage=a['coverage'],unknown_edges=unknown,
          counters={k:a[k] for k in ('geometry_calls','chart_trials','boundary_cache_reuses','policy_excluded_requests','materialization_calls','geometry_requests')},same_boundary_reuse_requires_exact_readset=True))
    write(out/'rescued_edges.json',dict(cases=rescues,counts=dict(counts),scope='LOCAL_ALL_SIDES_SAME_CENTER_CLOCK; P0_GRAPH_RECOMBINATION_SEPARATE_FROM_NEW_EDGE_EVIDENCE'))
    write(out/'sdmr_music_controls.json',dict(groups=read(ROOT/'controls/sdmr_music_controls.json'),source_recall=[dict(song=r['song'],slot=r['slot'],candidate_uids=r['candidate_uids'],
          source_E1W_candidates=r['source_recall_shadow']['candidate_uids'],source_recall_shadow=r['source_recall_shadow'],global_scores={name:values['GlobalLocalTokenMean'] for name,values in r['scores'].items()}) for r in read(P0/'controls/result.json')['slots']],
          no_new_encoder_calls=True,no_independent_quality_labels=True,scope='PAIRED_FROZEN_CONDITION_RESPONSE_NOT_ACCURACY'))
    write(out/'complete_path_witness.json',dict(tasks=c,counts=path_counts,unknown_edge_count=len(unknown),same_measured_graph_per_comparison=True,
          same_search_upper_bound=CFG['path_budget_extensions'],R58_adapter_not_full_reproduction=True,path_not_new_full_motion=True))
    write(out/'independent.json',{stage:read(ROOT/'receipts'/('independent_'+stage+'.json')) for stage in ('main','controls')})
    write(out/'provenance.json',dict(science=a['science'],protocol=CFG,parent_preview_manifest_sha256=a['parent_preview_manifest_sha256'],parent_controls_manifest_sha256=a['parent_controls_manifest_sha256'],
          main_manifest_sha256=sha(ROOT/'main/manifest.json'),controls_manifest_sha256=sha(ROOT/'controls/manifest.json'),private_arrays_sha256=sha(ROOT/'main/arrays.npz'),
          geometry='FROZEN_R62_G1_NUMERICAL_GATE_UNCHANGED; FOUR_EDGE_MEASUREMENT_EQUIVALENCE_TESTED',optimizer_calls=0))
    write(out/'qualification.json',dict(status='FROZEN_CONTEXT_ATTRIBUTION_COMPLETE',counts=dict(counts),path_counts=path_counts,unknown_edges=len(unknown),
          can_route=False,training_updates=0,new_clocks=0,new_candidates=0,new_whole_song_trajectory=False,Native24='NOT_EVALUABLE',artistic_quality='NOT_EVALUABLE',new_music_generalization=False))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir())}));print('PUBLIC_BATCH',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()));print(dict(counts));print(path_counts)

if __name__=='__main__':run()
