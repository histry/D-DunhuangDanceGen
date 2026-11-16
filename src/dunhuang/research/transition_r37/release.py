"""Small public evidence package; model/data tensors remain in the server archive."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha

def run(root,out):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh public evidence directory')
    out.mkdir(parents=True);verified=read(root/'verification.json');planning=read(root/'planning/result.json');smoothed=read(root/'smoothing/result.json');model=read(root/'content_model/result.json');content=read(root/'content_data/manifest.json')
    songs=[]
    for s in planning['songs']:
        plan=read(s['new_plan']);keys=('unit_uid','candidate_uid','source_uid','source_interval','music_interval','family_uid','phase_anchors','time_warp','history_evidence','scores')
        songs.append(dict(s,selected_events=[{k:e.get(k) for k in keys} for e in plan['events']]))
    smooth=[]
    for row in smoothed['records']:
        methods={}
        for arm,r in row['methods'].items():
            keys=[k for k,v in r['metrics'][0].items() if isinstance(v,(int,float))]
            methods[arm]=dict(mean={k:float(np.mean([v[k] for v in r['metrics']])) for k in keys},all_seeds=r['metrics'])
        smooth.append(dict(key=row['key'],source_intervals_local=row['intervals'],methods=methods))
    receipts=[]
    for folder in (root/'receipts',Path('outputs/transition_r37_preview_01/receipts'),Path('outputs/transition_r37_01/receipts')):
        if not folder.exists():continue
        for p in sorted(folder.rglob('*')):
            if p.is_file():receipts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    summary=dict(schema='r37_evidence_release_v1',release_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        planning=songs,generation=verified['generation'],smoothing=smooth,
        content=dict(counts=content['counts'],train_groups=content['train_groups'],val_groups=content['val_groups'],
                     main=model,diagnostics=read(root/'content_diagnostics.json')),
        checks=verified['checks'],full_regression_log_sha256=sha(root/'receipts/finalize/regression.log'),
        scientific_archive_sha256=sha(root/'artifacts_manifest.json'),final_viewer_protocol_sha256=sha(root/'viewer_final/protocol.json'),
        main_optimizer_updates=80,bridge_optimizer_updates=0,shuffled_training_ablation_updates=0,whole_song_generations=0,
        sdmr_can_route=False,operational_qualified=False,human_reviewed=False,full_native='NOT_EVALUABLE_22_OF24',
        source_retention_partly_enforced_not_learned=True,root_contact_truth='UNKNOWN',
        r36_comparison_scope='R37_SMOOTHING_BOUNDARY_METRIC_COVERS_ALL_CUTS_IN256; NOT_IDENTICAL_TO_R36_TWO_SELECTED_BOUNDARIES',
        new_plan_consumption_scope='ONE_TWO_EVENT_BOUNDARY_WITHIN256; NOT_ENTIRE_LONG_UNITS_OR_WHOLE_SONG')
    write(out/'r37_summary.json',summary);write(out/'r37_verification.json',verified);write(out/'r37_receipts.json',dict(files=receipts))
    print('R37_RELEASE',len(receipts),out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
