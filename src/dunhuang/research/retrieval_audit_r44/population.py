"""Expand query coverage without changing source rows, labels or Val tasks."""
import argparse,collections,hashlib
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,identity
from dunhuang.research.sdmr_event_rank.population import relation

def build(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh population')
    out.mkdir(parents=True);path=Path('outputs/sdmr_event_rank_01/benchmark/manifest.json');parent=read(path);rows=parent['rows'];cfg=parent['config']
    from dunhuang.research.paired_event_r41.events import validate_roles
    validate_roles(rows)
    groups=sorted({r['source_group'] for r in rows if r['role']=='train'});assert len(groups)==48
    ordered=sorted(groups,key=lambda g:hashlib.sha256(('R44_44041:'+g).encode()).hexdigest());cal=ordered[:24];probe=ordered[24:];queries=[];missing=[]
    for g in groups:
        sources=sorted({r['raw_motion_sha256'] for r in rows if r['role']=='train' and r['source_group']==g})[:2]
        for s in sources:
            for length in cfg['lengths']:
                opts=sorted([i for i,r in enumerate(rows) if r['role']=='train' and r['source_group']==g and r['raw_motion_sha256']==s and r['frames']==length],key=lambda i:(rows[i]['source_interval'],rows[i]['uid']))
                if opts:queries.append(opts[len(opts)//2])
                else:missing.append(dict(group=g,raw_motion_sha256=s,frames=length))
    tasks=[];unavailable=[];mu=np.array(parent['energy_calibration']['mean']);sd=np.array(parent['energy_calibration']['std'])
    for q in queries:
        row=rows[q];candidates=[i for i,r in enumerate(rows) if r['role']=='train' and r['frames']==row['frames'] and r['raw_motion_sha256']!=row['raw_motion_sha256']];positive=sorted([i for i in candidates if relation(row,rows[i],cfg)=='CLOCK_POSITIVE'],key=lambda i:rows[i]['uid'])[:cfg['maximum_positives']];negative=[i for i in candidates if relation(row,rows[i],cfg)=='CLOCK_NEGATIVE'];distance=lambda i:abs((row['music_scalar']-mu[0])/sd[0]-(rows[i]['motion_scalar']-mu[1])/sd[1])
        for pool in ('same_music','energy_matched','cross_music'):
            neg=[i for i in negative if (rows[i]['audio_sha256']==row['audio_sha256'] if pool=='same_music' else rows[i]['audio_sha256']!=row['audio_sha256'])]
            if pool=='energy_matched':neg=[i for i in neg if distance(i)<=1]
            neg=sorted(neg,key=lambda i:(distance(i) if pool!='same_music' else rows[i]['uid'],rows[i]['uid']))[:cfg['maximum_negatives']]
            if not positive or not neg:unavailable.append(dict(query=q,pool=pool,positives=len(positive),negatives=len(neg)));continue
            ids=sorted(positive+neg,key=lambda i:rows[i]['uid']);tasks.append(dict(uid=identity(dict(query=row['uid'],pool=pool)),query=q,candidates=ids,positive=[i in positive for i in ids],role='train',source_group=row['source_group'],pool=pool,frames=row['frames'],scope='PHYSICAL_AUDIO_CLOCK_NOT_ARTISTIC_TRUTH'))
    tasks.extend(t for t in parent['tasks'] if t['role']=='val');queries.extend(q for q in parent['queries'] if rows[q]['role']=='val')
    for t in tasks:
        for i,p in zip(t['candidates'],t['positive']):
            assert rows[i]['role']==rows[t['query']]['role'] and rows[i]['raw_motion_sha256']!=rows[t['query']]['raw_motion_sha256'];assert relation(rows[t['query']],rows[i],cfg)==('CLOCK_POSITIVE' if p else 'CLOCK_NEGATIVE')
    value=dict(parent,config=dict(cfg,maximum_train_music_groups=48,queries_per_source_length=1),queries=queries,tasks=tasks,unavailable=unavailable,missing_query_support=missing,parent_benchmark_sha256=sha(path),calibration_groups=cal,probe_groups=probe,query_rule='ALL48TRAIN_TWO_RAW_SOURCES_METADATA_MIDDLE_WINDOW_PER_LENGTH; OLD_VAL_UNCHANGED',optimizer_steps=0)
    write(out/'manifest.json',value);write(out/'population_audit.json',dict(train_groups=len(groups),train_queries=sum(rows[q]['role']=='train' for q in queries),val_queries=sum(rows[q]['role']=='val' for q in queries),tasks=len(tasks),unavailable=unavailable,missing=missing,same_choreography_positive_pairs=sum(rows[t['query']]['choreography_uid']==rows[i]['choreography_uid'] for t in tasks for i,p in zip(t['candidates'],t['positive']) if p),same_choreography_negative_pairs=sum(rows[t['query']]['choreography_uid']==rows[i]['choreography_uid'] for t in tasks for i,p in zip(t['candidates'],t['positive']) if not p),independence='DIFFERENT_RAW_RECORDINGS; SHARED_CHOREOGRAPHY_POSSIBLE; VAL_HISTORICALLY_EXPOSED',test_tensor_reads=0,can_route=False));print('R44_POPULATION',len(queries),len(tasks),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();build(a.out)
