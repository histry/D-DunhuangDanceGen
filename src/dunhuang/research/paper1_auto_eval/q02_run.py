"""Bounded Q02 lifecycle: register+view, accept, score, fresh-process replay."""
import sys
import numpy as np
from .q02_common import ROOT,Q01,HERE,CFG,binding,parents,read,write,sha,fresh,seal,verify_files
from . import q02_music as audio,q02_motion as motion,q02_view as view
from dunhuang.research.warp_coupling_r53.common import save_arrays

def preview():
    root=fresh(ROOT);paired=audio.register();write(root/'registration.json',dict(protocol=CFG,code=binding(),parents=parents(),paired=paired))
    out=fresh(root/'preview');rows,arrays=motion.build_arrays();save_arrays(out/'arrays.npz',arrays);write(out/'controls.json',rows);view.build(rows,arrays,out/'viewer');seal(out)
    print('Q02_PREVIEW_READY',len(rows),flush=True)

def score(replay=False):
    verify_files(ROOT/'preview');assert read(ROOT/'registration.json')['code']==binding()
    if not replay:
        receipt=read(ROOT/'receipts/browser_acceptance.json');assert receipt['status']=='PASS' and receipt['preview_manifest_sha256']==sha(ROOT/'preview/manifest.json')
    out=fresh(ROOT/('replay' if replay else 'main'));rows=read(ROOT/'preview/controls.json')
    if replay:
        recreated,arrays=motion.build_arrays();assert rows==recreated;save_arrays(out/'controls_replay.npz',arrays);assert sha(out/'controls_replay.npz')==sha(ROOT/'preview/arrays.npz')
    else:
        with np.load(ROOT/'preview/arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    results,deltas=motion.score(rows,arrays);write(out/'motion_scores.json',results);write(out/'sensitivity.json',deltas);summary=audio.evaluate(out)
    write(out/'qualification.json',dict(schema=CFG['schema'],music_clock_qualification='PASS_LIMITED_NEW_TO_METRIC_POPULATION' if len(summary)==2 and all(r['qualifies'] for r in summary) else 'NOT_ESTABLISHED',generalization='NOT_CERTIFIED_HISTORICAL_PROJECT_EXPOSURE',motion_controls=len(rows),observed_controls=sum(r['status']=='OBSERVED' for r in rows),source_groups=len({r['source_uid'] for r in rows}),motion_scope='SENSITIVITY_AND_OBSERVABILITY_NOT_ARTISTIC_ORDERING',MotionCritic=CFG['MotionCritic'],human_records=0,artistic_quality='NOT_EVALUABLE',training_steps=0,planner_calls=0,new_G1_calls=0,joint_source_time_decision='NOT_STARTED_QUALIFICATION_PENDING',can_route=False));seal(out)
    if replay:
        original=verify_files(ROOT/'main');diff=[n for n,h in original['files'].items() if sha(out/n)!=h];assert not diff,diff
        write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,scalar_differences=diff,control_arrays_exact=True,audio_replay=read(ROOT/'receipts/audio_replay.json'),parents_unchanged=parents(),training_steps=0,can_route=False))
    print('Q02_SCORES_READY',replay,flush=True)

def release():
    assert read(ROOT/'receipts/independent.json')['status']=='PASS';out=fresh('reports/paper1_auto_eval_q02_release')
    for name in ['motion_scores.json','sensitivity.json','paired_scores.json','paired_summary.json','qualification.json','manifest.json']:write(out/('main_manifest.json' if name=='manifest.json' else name),read(ROOT/'main'/name))
    # No private array paths, audio bytes, body coordinates or blinded identities copied.
    for name in ['independent','browser_acceptance','audio_replay']:write(out/(name+'.json'),read(ROOT/'receipts'/(name+'.json')))
    write(out/'protocol.json',CFG);write(out/'registration.json',read(ROOT/'registration.json'));write(out/'audio_events.json',read(ROOT/'audio/events.json'));write(out/'preview_manifest.json',read(ROOT/'preview/manifest.json'));write(out/'manifest.json',dict(files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}))

if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='preview':preview()
    elif stage=='score':score()
    elif stage=='replay':score(True)
    elif stage=='release':release()
    else:raise ValueError(stage)
