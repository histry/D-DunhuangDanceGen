"""Fresh-process evidence check, including independent saved-clock score arithmetic."""
from pathlib import Path
import numpy as np
from .infer import ROOT
from dunhuang.research.local_timing_r68.common import read,write,sha


def score(points,events,offset=0.):
    if not len(points) or not len(events):return None
    errors=[min(abs(float(p)+offset-float(e)) for e in events) for p in points]
    return float(np.mean(np.exp(-np.square(errors)/.02)))


def run():
    manifest=read(ROOT/'main/manifest.json')
    for name,h in manifest['files'].items():assert sha(ROOT/'main'/name)==h,name
    replay=read(ROOT/'replay/independent.json');assert replay['fresh_process'] and replay['exact'] and replay['differences']==[]
    catalogs=list((ROOT/'main').glob('*/catalog.json'))+list((ROOT/'paired').glob('*/catalog.json'))
    event_count=0
    for path in catalogs:
        c=read(path);assert sha(c['audio_path'])==c['audio_sha256'];n=c['audio_samples'];sr=c['sample_rate']
        ids=set()
        for e in c['events']:
            t=e['seconds'];assert e['uid'] not in ids;ids.add(e['uid'])
            assert e['seconds_supported']==(0<=t<n/sr)
            assert e['nearest_frame']==int(np.rint(30*t)) and e['frame_float']==30*t
            assert e['audio_sample_position']==t*sr
            assert e['nearest_frame_supported']==(0<=e['nearest_frame']<(n*30+sr-1)//sr)
            assert e['annotation_status']=='AUTOMATIC_UNVALIDATED' and e['semantic_truth']=='UNKNOWN'
        assert c['can_route'] is False and c['independent_annotations'] is None
        event_count+=len(ids)
    pop=read(ROOT/'paired/population.json')['sources'];rows=read(ROOT/'paired/scores.json');checks=0
    cats={s['source_uid']:read(ROOT/'paired'/str(i)/'catalog.json') for i,s in enumerate(pop) if s['status']=='SUPPORTED'}
    byhash={c['audio_sha256']:c for c in cats.values()}
    for r in rows:
        if 'scores' not in r:continue
        s=next(s for s in pop if s['source_uid']==r['source_uid']);c=cats[r['source_uid']]
        points=[p for p in s['motion_seconds'] if p>=.5 and p+.5<c['duration_seconds']]
        assert len(points)==r['retained_events'] and r['circular_shift'] is False
        for key,values in r['scores'].items():
            if key.startswith('R64_'):events=s['legacy_events'][key[4:]]
            else:
                model,kind=next((m,k) for m,k in [('beat_this','beat'),('beat_this','downbeat'),('allin1','beat'),('allin1','downbeat'),('acoustic','accent_candidate')] if m+'_'+k==key)
                events=[e['seconds'] for e in c['events'] if e['source']==model and e['kind']==kind and e['seconds_supported']]
            for shift,actual in zip(r['offsets'],values):
                expected=score(points,events,shift)
                assert (actual is None and expected is None) or abs(actual-expected)<1e-12,(r['source_uid'],key,shift)
                checks+=1
        donor=byhash.get(r['donor_audio_sha256']);end=min(c['duration_seconds'],donor['duration_seconds']) if donor else 0
        points=[p for p in s['motion_seconds'] if p>=.5 and p+.5<end];assert len(points)==r['wrong_song_same_event_count']
        for key,values in r['wrong_song'].items():
            model,kind=next((m,k) for m,k in [('beat_this','beat'),('allin1','beat'),('acoustic','accent_candidate')] if m+'_'+k==key)
            for name,cat in [('correct',c),('wrong',donor)]:
                events=[e['seconds'] for e in cat['events'] if e['source']==model and e['kind']==kind and e['seconds_supported']]
                expected=score(points,events);actual=values[name]
                assert (actual is None and expected is None) or abs(actual-expected)<1e-12
                checks+=1
    # Independently rebuild profiles from the frozen actual R69/GTW arrays, then score with scalar arithmetic.
    from dunhuang.research.paired_event_r41.events import profiles,phases
    from dunhuang.research.warp_coupling_r53.common import array_hash
    motion_rows=read(ROOT/'saved_motion_diagnostics/scores.json');lookup={(r['study'],r['unit'],r['method']):r for r in motion_rows}
    qualification=read(ROOT/'saved_motion_diagnostics/qualification.json');cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json')
    music={read(p)['audio_sha256']:read(p) for p in (ROOT/'main').glob('*/catalog.json')};cache={};saved_checks=0;missing=0
    for study,folder,registration in [('R69','outputs/retiming_baselines_r69_01/main','population.json'),('GTW','outputs/gtw_baseline_01/controls','registration.json')]:
        parent=Path(folder);assert sha(parent/'manifest.json')==qualification['source_bindings'][study]
        cases=read(parent/registration)['cases'];bykey={c['key']:c for c in cases}
        with np.load(parent/'arrays.npz') as z:
            for record in read(parent/'records.json'):
                case=bykey[record['case_key']] if study=='R69' else cases[record['case_index']]
                unit=record['case_key'] if study=='R69' else record['unit'];row=lookup[(study,unit,record['method'])];name=unit+'__'+record['method']+'__full_R'
                if name not in z:assert row['status']=='MISSING_EXECUTION';missing+=1;continue
                R=z[name];h=array_hash(R);assert h==row['actual_motion_sha256'] and len(R)==row['output_frames']
                if h not in cache:cache[h]=[e['frame']/30 for e in phases(profiles(R),cfg) if e['body_region']=='upper' and e['type']=='speed_peak']
                assert len(cache[h])==row['motion_event_count'];points=[v+case['pair_music_interval'][0]/30 for v in cache[h]];c=music[case['audio_sha256']]
                for model,kind in [('beat_this','beat'),('beat_this','downbeat'),('allin1','beat'),('allin1','downbeat'),('acoustic','accent_candidate'),('R64','beat'),('R64','accent_candidate')]:
                    events=[e['seconds'] for e in c['events'] if e['source']==model and e['kind']==kind and e['seconds_supported']]
                    expected=score(points,events);actual=row['scores'][model+'_'+kind]
                    assert (actual is None and expected is None) or abs(actual-expected)<1e-12,(study,unit,model,kind)
                    saved_checks+=1
    assert len(cache)==qualification['unique_motion_profiles']
    outputs=['main/manifest.json','replay/independent.json','paired/population.json','paired/scores.json','paired/summary.json','saved_motion_diagnostics/scores.json','saved_motion_diagnostics/qualification.json']
    write(ROOT/'receipts/independent_metrics.json',dict(status='PASS',fresh_process=True,main_full_model_replay=True,paired_model_replayed=False,paired_saved_event_score_recalculation=True,
        catalogs=len(catalogs),events=event_count,independent_score_checks=checks+saved_checks,paired_score_checks=checks,saved_motion_score_checks=saved_checks,saved_motion_profiles_recomputed=len(cache),missing_executions_retained=missing,bindings={n:sha(ROOT/n) for n in outputs},optimizer_calls=0,training_steps=0,quality='NOT_EVALUABLE',can_route=False))
    print('INDEPENDENT_MUSIC_METRICS_PASS',checks+saved_checks,flush=True)

if __name__=='__main__':run()
