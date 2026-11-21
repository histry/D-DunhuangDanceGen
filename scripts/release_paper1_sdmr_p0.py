"""Whitelisted scalar publication. No arrays, media, features or weights."""
from collections import Counter
from dunhuang.paper1.contracts import *

def ties(values,indices,maximize=True):
    good=[i for i in indices if values[i] is not None]
    if not good:return []
    best=(max if maximize else min)(values[i] for i in good)
    return [i for i in good if abs(values[i]-best)<=1e-12]

def build():
    p=read(ROOT/'preview/result.json');c=read(ROOT/'controls/result.json');check_seal(ROOT/'preview');check_seal(ROOT/'controls')
    for stage in ['preview','controls']:assert read(ROOT/'receipts'/('independent_'+stage+'.json'))['status']=='PASS'
    unique_edges={x['cache_key']:x for r in c['slots'] for group in r['connections'] for x in group}
    chart_trials=sum(len(x['proof']['guards']['G1']['trials']) for x in unique_edges.values())
    assert len(unique_edges)==c['new_geometry_calls'] and chart_trials<=3*2496
    target=fresh('reports/paper1_sdmr_integration_p0_release');table=[];songs=[];selection=[];groups=Counter();connection_counts=Counter();mapped=[];output=[];motion_change=[]
    for song in p['songs']:
        songs.append(dict(song=song['song'],frames=song['frames'],regions=song['regions'],audio_sha256=song['audio']['audio_sha256'],music_catalog_sha256=song['music_catalog_sha256'],
                          baseline_rotation_sha256=song['baseline_rotation_sha256'],baseline_exact=song['baseline_exact'],
                          historical_bridge_methods=dict(Counter(e['selected'] for e in song['baseline_connections']))))
        for slot in song['slots']:
            sr=next(x for x in c['slots'] if x['song']==song['song'] and x['slot']==slot['slot']);rows=slot['candidates'];assert [x['uid'] for x in rows]==sr['candidate_uids']
            for i,row in enumerate(rows):
                alignment={k:v for k,v in row['alignment'].items() if k!='redetected_events'}
                table.append(dict(song=song['song'],slot=slot['slot'],**{k:v for k,v in row.items() if k!='alignment'},alignment=alignment))
                for a in alignment['correspondences']:
                    if a['source_anchor_error'] is not None:mapped.append(a['source_anchor_error'])
                    if a['output_event_error'] is not None:output.append(a['output_event_error'])
                for x in sr['connections'][i]:connection_counts['requests']+=1;connection_counts['constructed']+=int(x['constructible']);connection_counts['passed']+=int(x['passes_original_screen'])
            for event in sorted({x['event_uid'] for x in rows}):
                ids=[i for i,x in enumerate(rows) if x['event_uid']==event];base=next(i for i in ids if rows[i]['origin']=='E1W');eligible=[i for i in ids if all(x['passes_original_screen'] for x in sr['connections'][i])]
                options=dict(E1W=[base],GTWLocal=ties(sr['GTWLocal'],ids,False),Single64=ties(sr['scores']['actual']['Single64'],ids),Multi64=ties(sr['scores']['actual']['Multi64'],ids),WrongMusic=ties(sr['scores']['wrong']['Single64'],ids),MusicZero=ties(sr['scores']['zero']['Single64'],ids))
                selected={m:[rows[i]['uid'] for i in v] for m,v in options.items()}
                selection.append(dict(song=song['song'],slot=slot['slot'],event_uid=event,candidate_uids=[rows[i]['uid'] for i in ids],baseline_uid=rows[base]['uid'],ties=selected,
                                      local_G1_eligible_uids=[rows[i]['uid'] for i in eligible],whole_alternative_path_checked=False,can_route=False))
                groups['source_slot_groups']+=1;groups['dual_clock_groups']+=int(len(ids)==2)
                for m,v in options.items():groups[m+'_scorable']+=bool(v);groups[m+'_baseline_among_ties']+=base in v;groups[m+'_all_ties_nonbaseline']+=bool(v) and base not in v
    def stats(v):return dict(count=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,maximum=float(np.max(v)) if v else None)
    write(target/'population.json',dict(task=p['task'],config=p['config'],songs=songs,source_bank=p['source_bank'],source_slot_scope='HISTORICAL_DEVELOPMENT_SONGS_NOT_NEW_MUSIC_GENERALIZATION',candidate_count=len(table),source_slot_groups=groups['source_slot_groups'],distinct_source_units=len({x['event_uid'] for x in table}),source_bank_units=len(p['source_bank']),source_bank_units_without_duration_eligible_core=[u for u in p['source_bank'] if u not in {x['event_uid'] for x in table}],duration_rejections=[dict(song=s['song'],slot=k['slot'],**r) for s in p['songs'] for k in s['slots'] for r in k['rejections']]))
    write(target/'candidates.json',table)
    write(target/'selections.json',selection)
    write(target/'readouts.json',c)
    write(target/'summary.json',dict(selection_counts=dict(groups),local_connection_counts=dict(connection_counts),actual_geometry_calls=c['new_geometry_calls'],geometry_requests=c['geometry_requests'],actual_chart_trials=chart_trials,chart_trial_upper_bound=3*2496,
          mapped_source_anchor_error_frames=stats(mapped),redetected_output_event_error_frames=stats(output),error_scope='ALL_UNVALIDATED_PROPOSALS_NOT_EXECUTED_TARGET_CONSTRAINTS_OR_ARTISTIC_TRUTH',
          source_and_output_event_error_have_different_missingness=True,score_change_is_not_quality_gain=True,stage_A='GLOBAL_MEAN_LOCAL_TOKENS_NOT_TRAINED_GLOBAL_HEAD',
          official_path_changed=False,new_planner_runs=0,new_training_updates=0,artistic_quality='NOT_EVALUABLE',native24='NOT_EVALUABLE',can_route=False))
    receipts={stage:read(ROOT/'receipts'/('independent_'+stage+'.json')) for stage in ['preview','controls']}
    write(target/'independent.json',receipts)
    write(target/'provenance.json',dict(preview_science=p['science'],controls_science=c['science'],preview_producer_commit='9e8a1661b713e08d907b05f0fe59567c58461fef',baseline_commit=BASE,parent_binding=p['parent_binding'],parent_result_sha256=p['parent_result_sha256'],parent_arrays_sha256=p['parent_arrays_sha256'],preview_manifest_sha256=sha(ROOT/'preview/manifest.json'),controls_manifest_sha256=sha(ROOT/'controls/manifest.json'),encoder_binding=c['encoder_binding'],network_digest=c['network_digest']))
    write(target/'qualification.json',dict(status='READ_ONLY_INTEGRATION_COMPLETE',learning_routing=False,source_recall_pruning=False,new_music_test=False,artistic_quality='NOT_EVALUABLE',semantic_correspondence_accuracy='NOT_EVALUABLE',native24='NOT_EVALUABLE',new_full_graph_search=False,new_whole_song_method_trajectory=False,local_G1_not_whole_path_qualification=True,can_route=False))
    write(target/'manifest.json',dict(files={x.name:sha(x) for x in sorted(target.iterdir())}))
    print('PUBLIC_BATCH',len(list(target.iterdir())),sum(x.stat().st_size for x in target.iterdir()));print(read(target/'summary.json'))

if __name__=='__main__':build()
