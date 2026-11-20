"""Publish bounded diagnostics and receipts; never copy media or model assets."""
from pathlib import Path
from collections import Counter
import shutil
from .infer import ROOT
from dunhuang.research.local_timing_r68.common import read,write,sha


def run():
    verified=read(ROOT/'receipts/independent_metrics.json');assert verified['status']=='PASS'
    out=Path('reports/music_events_release');out.mkdir(exist_ok=False);songs=[]
    for p in sorted((ROOT/'main').glob('*/catalog.json')):
        c=read(p);songs.append(dict(name=Path(c['audio_path']).stem,audio_sha256=c['audio_sha256'],duration_seconds=c['duration_seconds'],audio_samples=c['audio_samples'],sample_rate=c['sample_rate'],frames=c['frames'],
            event_counts=dict(Counter(e['source']+':'+e['kind'] for e in c['events'])),comparison=c['comparison'],stability=c['stability'],phrase_status=c['phrase_status'],accuracy=c['accuracy'],runtime=read(p.parent/'runtime.json')))
    write(out/'songs.json',songs)
    for source,target in [('paired/scores.json','paired_scores.json'),('paired/summary.json','paired_summary.json'),('saved_motion_diagnostics/scores.json','saved_motion_scores.json'),('saved_motion_diagnostics/qualification.json','saved_motion_qualification.json'),
                          ('receipts/independent_metrics.json','independent_metrics.json'),('replay/independent.json','full_model_replay.json'),('receipts/environment.json','environment.json'),('receipts/preview_browser_acceptance.json','preview_browser_acceptance.json'),
                          ('main/manifest.json','main_manifest.json'),('assets/official_weights_lock.json','official_weights_lock.json'),('assets/demucs_lock.json','demucs_lock.json'),('receipts/natten_download.json','natten_download.json'),('viewer_delivery.json','viewer_delivery.json'),('receipts/execution_commit.txt','main_execution_commit.txt'),('receipts/madmom_commit.txt','madmom_commit.txt'),('receipts/madmom_submodules.txt','madmom_submodules.txt')]:
        shutil.copyfile(ROOT/source,out/target)
    shutil.copyfile(Path(__file__).with_name('protocol.json'),out/'protocol.json')
    shutil.copyfile(ROOT/'receipts/pip_freeze.txt',out/'pip_freeze.txt')
    pop=read(ROOT/'paired/population.json');write(out/'population.json',dict(selection=pop['selection'],exposure=pop['exposure'],requested_groups=pop['requested_groups'],registered_groups=pop['registered_groups'],sources=[{k:s.get(k) for k in ['source_uid','dataset','role','audio_sha256','clock_scope','status','legacy_error']} for s in pop['sources']]))
    write(out/'qualification.json',dict(automatic_candidate_catalog=True,full_song_inference=len(songs),independent_human_records=0,music_structure_accuracy='NOT_EVALUABLE',music_quality_gain='NOT_EVALUABLE',phrase_truth='UNKNOWN',Native24='NOT_EVALUABLE',can_route=False,planner_upgraded=False,training_steps=0,source_binding=sha(ROOT/'main/manifest.json')))
    for p in out.iterdir():
        if p.is_file():p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},media_copied=False,weights_copied=False))
    print('MUSIC_EVENTS_PUBLIC_EVIDENCE_READY',len(songs),flush=True)

if __name__=='__main__':run()
