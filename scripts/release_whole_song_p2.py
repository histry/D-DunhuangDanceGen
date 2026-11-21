"""Bounded scalar export. No coordinates, raw tensors, audio, weights or private paths."""
import argparse
from pathlib import Path
from dunhuang.research.whole_song_witness_p2.common import ROOT,CFG,read,write,sha,fresh,identity,check_seal

def produce(out):
    for stage in ('main','controls'):
        check_seal(ROOT/stage)
        receipt=read(ROOT/'receipts'/('independent_'+stage+'.json'));assert receipt['status']=='PASS' and receipt['source_manifest_sha256']==sha(ROOT/stage/'manifest.json')
    browser=read(ROOT/'receipts/main_browser.json');assert browser['passed'] and browser['main_result_sha256']==sha(ROOT/'main/result.json')
    a=read(ROOT/'main/result.json');b=read(ROOT/'controls/result.json');assert a['science']==b['science']
    out=fresh(out)
    def filtered(r):
        names=('task','stage','protocol','science','tasks','motions','baselines','source_units','counts','new_unknown_edges_solved','planner_calls','training_updates','can_route','Native24','independent_artistic_quality')
        return {name:r[name] for name in names}
    write(out/'registration.json',{stage:read(ROOT/stage/'registration.json') for stage in ('main','controls')})
    write(out/'execution.json',filtered(a));write(out/'controls.json',filtered(b))
    motions={r['key']:r for doc in (a,b) for r in doc['motions']+doc['baselines']}
    index={(r['song'],r['method'],None if r['case'] is None else identity(r['case'])):r for r in b['tasks']}
    differences=read(ROOT/'controls/comparisons.json');compared=[]
    for task,row in zip([r for r in a['tasks'] if r['motion_key']],differences):
        assert task['case']==row['case'];s=task['song'];main=motions[task['motion_key']]
        choices=[('EXACT_HISTORICAL_B0','b0_song'+str(s))]
        for arm in ('P0','P1'):choices.append((arm+'_UNCONSTRAINED_STRONG',index[s,arm+'_SAVED_STRONG_UNCONSTRAINED',None]['motion_key']))
        choices.append(('P1_R58_ADAPTER_SAME_CENTER',index[s,'P1_SAVED_R58_ADAPTER_CENTER',identity(task['case'])]['motion_key']))
        values=[]
        for name,key in choices:
            delta=None
            if key:
                other=motions[key];delta={}
                for n in ('D1','D2','D3'):
                    x=main['metrics']['FK_RMS_derivatives'][n]['full']['q95'];y=other['metrics']['FK_RMS_derivatives'][n]['full']['q95']
                    delta[n]=dict(main=x,comparator=y,main_minus_comparator=x-y)
            values.append(dict(method=name,motion_key=key,available=key is not None,full_FK_RMS_Q95=delta,quality_preference='NOT_EVALUABLE'))
        compared.append(dict(**row,actual_numeric_comparisons=values))
    write(out/'comparisons.json',dict(cases=compared,source_Warp_neighbor_changes_are_descriptive_not_causal=True,main_is_already_saved_constrained_strong=True,independent_music_count=0,historical_music_count=3))
    keys={v['boundary_key'] for doc in (a,b) for v in doc['materialized']}
    calls=sum(doc['counts']['bridge_materializations'] for doc in (a,b))
    write(out/'connections.json',dict(materialized={stage:doc['materialized'] for stage,doc in [('main',a),('controls',b)]},distinct_boundary_keys_across_stages=len(keys),
        original_and_control_constructor_calls=calls,including_independent_replays_constructor_calls=2*calls,new_unknown_edges_evaluated=0,
        only_previously_measured_eligible_edges=True,original_G1_gate_unchanged=True))
    source=a['source_binding']
    write(out/'provenance.json',dict(science=a['science'],parent_manifests=CFG['parent_manifests'],stage_manifests={stage:sha(ROOT/stage/'manifest.json') for stage in ('main','controls')},
        raw_private_arrays_sha256={stage:sha(ROOT/stage/'arrays.npz') for stage in ('main','controls')},
        native_source_sha256={u:r['sha256'] for u,r in source['native_sources'].items()},J_sha256=source['J_sha256'],parents_sha256=source['parents_sha256'],
        reference=source['reference'],public_algorithm_file_hashes=source['algorithms'],music=[dict(audio_sha256=r['audio_sha256'],structure_sha256=r['structure_sha256']) for r in source['music']],
        private_paths_input_tensors_rotations_coordinates_audio_weights_excluded=True))
    write(out/'independent.json',dict(main=read(ROOT/'receipts/independent_main.json'),controls=read(ROOT/'receipts/independent_controls.json'),browser=browser))
    write(out/'qualification.json',dict(status='ACTUAL_WHOLE_SONG_WITNESSES_AND_FROZEN_CONTROLS_EXACTLY_REPLAYED',main=a['counts'],controls=b['counts'],
        whole_song_So3_and_FK_arrays_exist=True,all_assembled_frame_coverage_exact_once=True,source_cores_and_saved_bridge_hashes_exact=True,all_new_assembled_G1_edges_pass_original_screen=True,
        baseline_preserved_with_historical_G0_G1=True,new_paths_selected=0,new_unknown_edges_solved=0,training_updates=0,can_route=False,
        Native24='NOT_EVALUABLE',world_Root='UNKNOWN',contacts='UNKNOWN',independent_artistic_quality='NOT_EVALUABLE',new_music_generalization=False,new_planner_superiority=False))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir())}))
    print('P2_SCALAR_BATCH',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--compare');parser.add_argument('--receipt');args=parser.parse_args()
    if args.compare:
        from dunhuang.research.source_tasks.common import forbid_optimizers
        forbid_optimizers()
    produce(Path(args.out))
    if args.compare:
        assert read(Path(args.out)/'manifest.json')==read(Path(args.compare)/'manifest.json')
        assert sha(Path(args.out)/'manifest.json')==sha(Path(args.compare)/'manifest.json')
        write(args.receipt,dict(status='PASS',fresh_process=True,all_nine_scalar_files_exact=True,optimizer_calls=0,source_manifest_sha256=sha(Path(args.compare)/'manifest.json')))
