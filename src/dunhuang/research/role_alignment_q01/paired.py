"""Metadata-isolated new-to-role clock task; not unseen semantic truth."""
from pathlib import Path
import sys
import numpy as np
from .common import ROOT,CFG,read,write,sha,fresh,prepare
from .events import motion_catalog
from .alignment import score,shuffled
from dunhuang.research.timing_validity_r66 import engine
from dunhuang.research.timing_validity_r66.common import CFG as LEGACY
from dunhuang.research.paper1_auto_eval.q02_common import rank
from dunhuang.research.warp_coupling_r53.common import array_hash,identity

def register():
    assert sha(LEGACY['data'])==LEGACY['data_sha256'];meta=read(LEGACY['data']);past=read('outputs/music_events_01/paired/population.json')['sources'];p=Path('reports/paper1_auto_eval_q02_release/registration.json');assert sha(p)==CFG['past_paired_registration_sha256'];past+=read(p)['paired']['sources'];ids={r['source_uid'] for r in past};fields=['source_uid','audio_sha256','raw_motion_sha256','group','recording_uid','choreography_uid'];prior=[s for s in meta['sources'] if s['source_uid'] in ids];blocked={k:{s[k] for s in prior if s.get(k) not in [None,'UNKNOWN']} for k in fields};sources=[]
    for dataset,role in [('AIST++','train'),('FineDance','genre_train_diagnostic')]:
        seen={k:set() for k in fields};n=0
        for s in sorted(meta['sources'],key=lambda s:s['source_uid']):
            if (s['dataset'],s['role'])!=(dataset,role):continue
            if any(s.get(k) not in [None,'UNKNOWN'] and (s[k] in blocked[k] or s[k] in seen[k]) for k in fields):continue
            sources.append(s);n+=1
            for k in fields:
                if s.get(k) not in [None,'UNKNOWN']:seen[k].add(s[k])
            if n==CFG['paired_groups_per_domain']:break
    return dict(sources=sources,excluded_prior_source_uids=sorted(ids),new_to_role_task=True,independent_new_music_certified=False,historical_project_exposure='R59_OR_EARLIER_POSSIBLE',selection=CFG['paired_selection'],source_manifest_sha256=sha(LEGACY['data']))

def infer(replay=False):
    import torch,soundfile as sf
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    from dunhuang.research.music_events.infer import assets,acoustic
    out=fresh(ROOT/('audio_replay' if replay else 'audio'));reg=read(ROOT/'preview/registration.json')['paired'];model=assets();torch.manual_seed(76001);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;tracker=Audio2Frames(model['beat_this_checkpoint'],device='cuda',float16=False);rows=[]
    for s in reg['sources']:
        try:
            path=s['audio']['path'];assert sha(path)==s['audio_sha256'];y,sr=sf.read(path,always_2d=True,dtype='float32');logits=tracker(y.mean(1),sr);beat,down=Postprocessor(type='minimal',fps=50)(*logits);ac,_=acoustic(path);events=[]
            for role,values in [('BEAT',beat),('DOWNBEAT',down),('ACCENT_PROXY',ac['accent_frames']/30),('STRUCTURE_PROXY',ac['chroma_frames']/30)]:
                for i,t in enumerate(values):
                    if 0<=t<len(y)/sr:events.append(dict(uid=identity([s['audio_sha256'],role,i,float(t)]),role=role,time=float(t),semantic_role='UNKNOWN'))
            rows.append(dict(source_uid=s['source_uid'],status='OBSERVED',audio_sha256=s['audio_sha256'],audio_samples=len(y),sample_rate=sr,duration_seconds=len(y)/sr,events=events,activation_sha256=[array_hash(t.sigmoid().cpu().numpy()) for t in logits],model_manifest_sha256=sha('outputs/music_events_01/assets/model_manifest.json')))
        except Exception as e:rows.append(dict(source_uid=s['source_uid'],status='CONTRACT_FAILURE',error=repr(e),replacement_used=False))
        print('ROLE_AUDIO',s['source_uid'],rows[-1]['status'],flush=True)
    write(out/'events.json',rows)
    if replay:
        differences=[] if sha(out/'events.json')==sha(ROOT/'audio/events.json') else ['events.json'];write(ROOT/'receipts/audio_replay.json',dict(status='PASS' if not differences else 'FAIL',differences=differences,fresh_process=True));assert not differences,differences

def evaluate(out):
    reg=read(ROOT/'preview/registration.json')['paired'];cats={r['source_uid']:r for r in read(ROOT/'audio/events.json')};results=[];catalogs=[]
    for i,s in enumerate(reg['sources']):
        try:
            own=cats[s['source_uid']];assert own['status']=='OBSERVED';donor=next(d for d in reg['sources'][i+1:]+reg['sources'][:i] if d['dataset']==s['dataset'] and d['audio_sha256']!=s['audio_sha256']);wrong=cats[donor['source_uid']];assert wrong['status']=='OBSERVED';R,proof=engine.load_motion(s);offset=s['audio_offset_frames']/30;cat=motion_catalog(R,s['source_uid'],offset);lo,hi=proof['shared_source_interval'];start=max(0,lo/30+offset)+.5;end=min((hi-1)/30+offset,own['duration_seconds'],wrong['duration_seconds'])-.5;assert end>start;query=[e for e in cat['events'] if e['supported'] and start<=e['time']<end and e['evidence_span'][0]>=lo and e['evidence_span'][1]<=hi];music=[e for e in own['events'] if start<=e['time']<end];wm=[e for e in wrong['events'] if start<=e['time']<end];catalogs.append(cat);density=[]
            for role in CFG['role_map'].values():
                events=[e for e in music if e['role']==role]
                density.extend(dict(e,time=start+(j+.5)*(end-start)/len(events)) for j,e in enumerate(events))
            positions=[dict(e,time=start+(j+.5)*(end-start)/len(query)) for j,e in enumerate(query)] if query else [];scores={}
            for method in CFG['paired_methods']:
                values=[]
                for delta in CFG['clock_offsets_seconds']:
                    mm=shuffled(music) if method=='TypeShuffled' else density if method=='MusicDensity' else music;mm=[dict(e,time=e['time']+delta) for e in mm];qq=positions if method=='PositionOnly' else query;v=0. if method=='NoMusic' else score(qq,mm,typed=method!='NoTypes');values.append(v)
                scores[method]=dict(values=values,ranking=rank(values))
            results.append(dict(source_uid=s['source_uid'],dataset=s['dataset'],role=s['role'],dancer_role=s.get('dancer_role','UNKNOWN'),status='OBSERVED',clock_scope=s['clock_scope'],proof=proof,audio_group=s['audio_sha256'],wrong_audio_group=donor['audio_sha256'],common_support_seconds=[start,end],query_events=len(query),music_events=len(music),query_role_counts={k:sum(e['role']==k for e in query) for k in CFG['role_map']},music_role_counts={k:sum(e['role']==k for e in music) for k in CFG['role_map'].values()},scores=scores,correct_vs_wrong_role_scores=[score(query,music),score(query,wm)],semantic_accuracy='NOT_EVALUABLE'))
        except Exception as e:results.append(dict(source_uid=s['source_uid'],dataset=s['dataset'],status='CONTRACT_FAILURE',error=repr(e),replacement_used=False))
        print('ROLE_PAIRED',s['source_uid'],results[-1]['status'],flush=True)
    summary=[];rng=np.random.default_rng(CFG['seed'])
    for dataset in ['AIST++','FineDance']:
        own=[r for r in results if r['dataset']==dataset and r['status']=='OBSERVED'];valid=[r for r in own if all(r['scores'][m]['ranking']['status']=='EVALUABLE' for m in CFG['paired_methods'])];stats={m:dict(top1=sum(r['scores'][m]['ranking']['strict_top1'] for r in valid),tied_top1=None if not valid else float(np.mean([r['scores'][m]['ranking']['tied_top1_credit'] for r in valid])),paired_win=None if not valid else float(np.mean([r['scores'][m]['ranking']['paired_win'] for r in valid]))) for m in CFG['paired_methods']};deltas={}
        for m in CFG['paired_methods'][1:]:
            values=[r['scores']['RoleOrdered']['ranking']['tied_top1_credit']-r['scores'][m]['ranking']['tied_top1_credit'] for r in valid];deltas[m]=None if len(values)<2 else np.quantile(rng.choice(values,(CFG['bootstrap_draws'],len(values)),replace=True).mean(1),[.025,.975]).tolist()
        qualifies=len(valid)==CFG['paired_groups_per_domain'] and all(v is not None and v[0]>0 for v in deltas.values());summary.append(dict(dataset=dataset,registered=sum(s['dataset']==dataset for s in reg['sources']),observed=len(own),complete_cases=len(valid),methods=stats,Role_minus_null_top1_95=deltas,clock_gate_pass=qualifies,semantic_role_gate='NOT_EVALUABLE_NO_INDEPENDENT_RECORDS',independent_new_music_certified=False))
    write(out/'paired_results.json',results);write(out/'paired_motion_catalogs.json',catalogs);write(out/'paired_summary.json',summary);return summary

if __name__=='__main__':infer('--replay' in sys.argv)
