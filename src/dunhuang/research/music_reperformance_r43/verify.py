"""Independent fresh-process artifacts, exact replay, lineage and phase-order audit."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.source_context_r42.contracts import exact_arrays


def run(generation,ranking,out):
    generation=Path(generation);ranking=Path(ranking);out=Path(out)
    if out.exists():raise ValueError('Fresh independent audit')
    forbid_optimizers();packs=[]
    for root,a,b in [(generation,'main','main_replay'),(generation,'controls','controls_replay'),(generation,'main_guarded','guarded_replay'),(ranking,'features','features_replay'),(ranking,'evaluation','evaluation_replay'),(ranking,'change_shadow','change_shadow_replay')]:
        exact_arrays(root/a/'arrays.npz',root/b/'arrays.npz');packs.append(dict(root=str(root),first=a,replay=b,arrays=len(np.load(root/a/'arrays.npz').files)))
    assert read(ranking/'benchmark/manifest.json')==read(ranking/'benchmark_replay/manifest.json')
    first=read(generation/'qualification.json');second=read(generation/'qualification_replay.json')
    for field in ('records','train_music_gate','train_change_gate','empirical_training_gate_passed','training_admitted','official_fk_executed'):assert first[field]==second[field],field
    parents=[]
    for name in ('bridge_r38_01','stabilize_r39_01','contract_r40_01','paired_event_r41_01','source_context_r42_01'):
        p=Path('outputs')/name;meta=read(p/'artifacts_manifest.json')
        for row in meta['files']:assert sha(p/row['path'])==row['sha256'],(name,row['path'])
        parents.append(dict(root=name,files=len(meta['files']),manifest_sha256=sha(p/'artifacts_manifest.json')))
    for root in (generation,ranking):
        for viewer in root.glob('viewer*'):
            if not (viewer/'protocol.json').exists():continue
            for row in read(viewer/'protocol.json')['files']:assert sha(viewer/row['name'])==row['sha256'],str(viewer/row['name'])
    assert (generation/'receipts/main_browser_acceptance.json').stat().st_mtime<(generation/'controls/result.json').stat().st_mtime
    assert (ranking/'receipts/p0_browser_acceptance.json').stat().st_mtime<(ranking/'features/result.json').stat().st_mtime
    benchmark=read(ranking/'benchmark/manifest.json');rows=benchmark['rows'];isolation={}
    for field in ('raw_motion_sha256','audio_sha256','source_group'):
        mapping={}
        for r in rows:assert mapping.setdefault(r[field],r['role'])==r['role']
        isolation[field]=len(mapping)
    for task in benchmark['tasks']:
        query=rows[task['query']]
        for i in task['candidates']:assert rows[i]['role']==query['role'] and rows[i]['frames']==query['frames'] and rows[i]['raw_motion_sha256']!=query['raw_motion_sha256']
    origin={r['case']:r for r in read('outputs/reference_r26_01/data/result.json')['records']};aist=[]
    for r in read(generation/'main/result.json')['records']:
        if r['domain']=='AIST++':aist.append(dict(case=r['case'],adaptation_role=r['role'],original_metadata=origin[r['case']],scope='FROZEN_PARENT_TRAIN_CACHE_NOT_NEW_DATA_V3_TEST'))
    feature=read(ranking/'features/result.json');assert feature['optimizer_steps']==0 and feature['model_parameters']==522632
    native=read(ranking/'change_shadow/result.json');assert not native['semantic_ground_truth'] and not native['can_route']
    write(out,dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),payloads=packs,benchmark_identity_exact=True,qualification_values_exact=True,parents=parents,role_isolation=isolation,actual_aist_lineage=aist,all_same_population_controls=True,main_delivery_preceded_controls=True,ranking_p0_delivery_preceded_scoring=True,optimizer_steps=0,whole_song_generations=0,operational_qualified=False,can_route=False));print('R43_INDEPENDENT_VERIFIED',len(packs),sum(p['files'] for p in parents),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--generation',required=True);p.add_argument('--ranking',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.generation,a.ranking,a.out)
