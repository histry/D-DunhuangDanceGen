"""Scalar-only P3 evidence; private motion arrays and source paths never exported."""
import argparse
from pathlib import Path
from dunhuang.research.minimal_context_p3.common import ROOT,CFG,read,write,sha,fresh,check_seal

def produce(out):
    stages=('main','controls');docs={}
    for stage in stages:
        check_seal(ROOT/stage);receipt=read(ROOT/'receipts'/('independent_'+stage+'.json'))
        assert receipt['status']=='PASS' and receipt['source_manifest_sha256']==sha(ROOT/stage/'manifest.json')
        docs[stage]=read(ROOT/stage/'result.json')
    assert docs['main']['science']==docs['controls']['science']
    browser=read(ROOT/'receipts/main_browser.json');assert browser['passed'] and browser['main_result_sha256']==sha(ROOT/'main/result.json')
    out=fresh(out)
    write(out/'minimal_edit_paths.json',{stage:dict(registration=read(ROOT/stage/'registration.json'),paths=read(ROOT/stage/'minimal_edit_paths.json')) for stage in stages})
    write(out/'exact_measured_graph_optima.json',{stage:read(ROOT/stage/'exact_measured_graph_optima.json') for stage in stages})
    comparison=read(ROOT/'controls/source_only_comparisons.json')
    write(out/'source_warp_change_attribution.json',dict(arms={stage:read(ROOT/stage/'source_warp_change_attribution.json') for stage in stages},secondary_objective_comparison=comparison,
          objective_preservation_not_independent_artistic_quality=True,historical_music_count=3,new_independent_music_count=0))
    write(out/'full_motion_metrics.json',{stage:read(ROOT/stage/'full_motion_metrics.json') for stage in stages})
    names=('counts','motions','source_units','materialized','objective','stage','new_unknown_edges_solved','new_clocks','new_candidates','training_updates','can_route','Native24','independent_artistic_quality')
    write(out/'execution.json',{stage:{name:docs[stage][name] for name in names} for stage in stages})
    source=docs['main']['source_binding']
    write(out/'provenance.json',dict(science=docs['main']['science'],protocol=CFG,stage_manifests={stage:sha(ROOT/stage/'manifest.json') for stage in stages},
          private_arrays_sha256={stage:sha(ROOT/stage/'arrays.npz') for stage in stages},native_source_sha256={uid:r['sha256'] for uid,r in source['native_sources'].items()},
          J_sha256=source['J_sha256'],parents_sha256=source['parents_sha256'],reference=source['reference'],public_algorithm_file_hashes=source['algorithms'],
          music=[dict(audio_sha256=r['audio_sha256'],structure_sha256=r['structure_sha256']) for r in source['music']],
          private_paths_motion_coordinates_rotations_tensors_audio_weights_excluded=True))
    runtime={stage:read(Path('outputs/minimal_context_p3_launch')/('source_version_'+stage+'.json')) for stage in ('controls','replay-controls')}
    write(out/'independent.json',dict(replays={stage:read(ROOT/'receipts'/('independent_'+stage+'.json')) for stage in stages},browser=browser,runtime_source_identity=runtime))
    motions=[m for doc in docs.values() for m in doc['motions']];keys={m['key'] for doc in docs.values() for m in doc['materialized']};calls=sum(d['counts']['bridge_materializations'] for d in docs.values())
    present=[r for r in comparison if r['primary'] is not None]
    assert all(r['primary']['source_units']==r['source_only']['source_units'] and r['primary']['same_unit_clocks']<=r['source_only']['same_unit_clocks'] for r in present)
    write(out/'qualification.json',dict(status='PRIMARY_AND_SOURCE_ONLY_EXACT_GRAPH_OPTIMA_ACTUALLY_EXECUTED_AND_REPLAYED',counts={stage:docs[stage]['counts'] for stage in stages},
          distinct_rotations_across_arms=len({m['rotation_sha256'] for m in motions}),distinct_boundary_keys=len(keys),original_constructor_calls=calls,including_independent_replays_constructor_calls=2*calls,
          primary_extra_clock_changes=[r['primary']['same_unit_clocks'] for r in present],source_only_extra_clock_changes=[r['source_only']['same_unit_clocks'] for r in present],
          exact_minimum_only_in_frozen_measured_graph=True,new_unknown_edges_solved=0,new_clocks=0,new_candidates=0,training_updates=0,can_route=False,
          Native24='NOT_EVALUABLE',independent_artistic_quality='NOT_EVALUABLE',independent_music_quality='NOT_EVALUABLE',new_music_generalization=False,continuous_domain_global_optimality=False))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir())}))
    print('P3_SCALAR_BATCH',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()),flush=True)

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
