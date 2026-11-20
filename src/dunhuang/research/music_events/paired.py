"""Limited historical paired-clock diagnostic; no artistic correctness claims."""
from pathlib import Path
from collections import defaultdict
import sys,json
import numpy as np
from .infer import ROOT,analyze,CFG
from .catalog import reward
from dunhuang.research.local_timing_r68.common import read,write,sha

def prepare():
    from dunhuang.research.timing_validity_r66 import population,engine
    from dunhuang.research.timing_validity_r66.common import CFG as old
    assert sha(old['data'])==old['data_sha256'];pop=population.select(read(old['data']),old);chosen=[]
    for dataset in ['AIST++','FineDance']:
        seen=set();own=[]
        for source in sorted(pop['sources'].values(),key=lambda s:s['source_uid']):
            if source['dataset']!=dataset or 'train' not in source['role'].lower() or source['audio_sha256'] in seen:continue
            seen.add(source['audio_sha256']);own.append(source)
            if len(own)==3:break
        chosen.extend(own)
    if not chosen:raise ValueError('No actual registered Train recordings available')
    out=ROOT/'paired';out.mkdir(exist_ok=False);rows=[]
    for i,s in enumerate(chosen):
        row=dict(source_uid=s['source_uid'],dataset=s['dataset'],role=s['role'],group=s['audio_sha256'],audio=s['audio'],audio_sha256=s['audio_sha256'],clock_scope=s['clock_scope'],status='REGISTERED')
        try:
            R,proof=engine.load_motion(s);events,evproof=engine.motion_events(R)
            lo,hi=proof['shared_source_interval']
            legacy=None;legacy_error=None
            try:
                oldclock=engine.audio_clock(s['audio']['path'])
                legacy=dict(beat=(np.asarray(oldclock['beat_frames'])/30).tolist(),accent_candidate=(np.asarray(oldclock['accent_frames'])/30).tolist())
            except Exception as error:legacy_error=repr(error)
            rows.append(dict(row,status='SUPPORTED',motion_seconds=[p['frame']/30+s['audio_offset_frames']/30 for p in events if p['body_region']=='upper' and p['evidence_span'][0]>=lo and p['evidence_span'][1]<=hi],proof=proof,event_proof=evproof,legacy_events=legacy,legacy_error=legacy_error))
        except Exception as e:rows.append(dict(row,status='RETAINED_CONTRACT_FAILURE',reason=repr(e)))
    write(out/'population.json',dict(sources=rows,selection='LEX_FIRST3_TRAIN_DISTINCT_AUDIO_GROUPS_PER_DATASET_NO_REPLACEMENT',requested_groups=6,registered_groups=len(rows),exposure='HISTORICALLY_EXPOSED_NOT_NEW_GENERALIZATION',human_reviewed=False))

def run():
    receipt=read(ROOT/'receipts/preview_browser_acceptance.json');assert receipt['status']=='PASS' and receipt['manifest_sha256']==sha(ROOT/'main/manifest.json')
    pop=read(ROOT/'paired/population.json');records=[]
    for i,s in enumerate(pop['sources']):
        if s['status']!='SUPPORTED':records.append(s);continue
        audio=s['audio']['path'];c=analyze(audio,ROOT/'paired'/str(i),dict(audio_sha256=s['audio_sha256']))
        p=np.asarray(s['motion_seconds']);p=p[(p>=.5)&(p+.5<c['duration_seconds'])];scores={}
        for model,kind in [('beat_this','beat'),('beat_this','downbeat'),('allin1','beat'),('allin1','downbeat'),('acoustic','accent_candidate')]:
            e=[r['seconds'] for r in c['events'] if r['source']==model and r['kind']==kind and r['seconds_supported']]
            scores[model+'_'+kind]=[reward(p,e,d) for d in CFG['paired_offsets_seconds']]
        if s['legacy_events']:
            for kind,e in s['legacy_events'].items():scores['R64_'+kind]=[reward(p,e,d) for d in CFG['paired_offsets_seconds']]
        # Wrong-song uses the next registered distinct audio in the SAME dataset. No score-based donor search.
        donor=next((d for d in pop['sources'][i+1:]+pop['sources'][:i] if d['dataset']==s['dataset'] and d['audio_sha256']!=s['audio_sha256'] and d['status']=='SUPPORTED'),None)
        records.append(dict(source_uid=s['source_uid'],dataset=s['dataset'],group=s['group'],clock_scope=s['clock_scope'],retained_events=len(p),scores=scores,donor_audio_sha256=None if donor is None else donor['audio_sha256'],
            offsets=CFG['paired_offsets_seconds'],same_support_all_offsets=True,circular_shift=False,quality='NOT_EVALUABLE'))
        print('PAIRED_AUDIO_COMPLETE',i+1,len(pop['sources']),flush=True)
    # Wrong-song correspondence uses the overlap supported by both actual waveforms.
    cat={s['audio_sha256']:read(ROOT/'paired'/str(i)/'catalog.json') for i,s in enumerate(pop['sources']) if s['status']=='SUPPORTED'}
    for r in records:
        if 'scores' not in r:continue
        donor=cat.get(r['donor_audio_sha256']);source=next(s for s in pop['sources'] if s['source_uid']==r['source_uid']);own=cat[r['group']]
        p=np.asarray(source['motion_seconds']);end=min(own['duration_seconds'],donor['duration_seconds']) if donor else 0;p=p[(p>=.5)&(p+.5<end)]
        r['wrong_song_same_event_count']=len(p);r['wrong_song']={}
        if donor:
            for model,kind in [('beat_this','beat'),('allin1','beat'),('acoustic','accent_candidate')]:
                key=model+'_'+kind
                a=[e['seconds'] for e in own['events'] if e['source']==model and e['kind']==kind and e['seconds_supported']]
                b=[e['seconds'] for e in donor['events'] if e['source']==model and e['kind']==kind and e['seconds_supported']]
                r['wrong_song'][key]=dict(correct=reward(p,a),wrong=reward(p,b))
    write(ROOT/'paired/scores.json',records)
    summary=[];rng=np.random.default_rng(76001)
    for dataset in ['AIST++','FineDance']:
        own=[r for r in records if r.get('dataset')==dataset and 'scores' in r]
        for key in sorted({k for r in own for k in r['scores']}):
            values=[]
            for r in own:
                scores=r['scores'][key]
                if all(v is not None for v in scores):values.append(float(np.mean([1 if scores[2]>scores[j]+1e-12 else .5 if abs(scores[2]-scores[j])<=1e-12 else 0 for j in [0,1,3,4]])))
            ci=None if len(values)<2 else np.quantile(rng.choice(values,(2000,len(values)),replace=True).mean(1),[.025,.975]).tolist()
            summary.append(dict(dataset=dataset,metric=key,groups=len(values),mean_paired_win=float(np.mean(values)) if values else None,group_bootstrap_95=ci))
    write(ROOT/'paired/summary.json',dict(results=summary,scope='VERY_SMALL_EXPOSED_TRAIN_CLOCK_DIAGNOSTIC_NOT_ARTISTIC_TRUTH',independent_accuracy='NOT_EVALUABLE',training_steps=0,can_route=False))

if __name__=='__main__':prepare() if '--prepare' in sys.argv else run()
