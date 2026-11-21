"""Fixed new-to-BAS population, frozen Beat This!, top-1 and null controls."""
from pathlib import Path
import numpy as np
from .q02_common import ROOT,CFG,Q01,read,write,sha,prepare,rank
from . import music
from dunhuang.research.timing_validity_r66 import engine
from dunhuang.research.timing_validity_r66.common import CFG as LEGACY
from dunhuang.research.complete_r36.core import relative_fk
from dunhuang.research.warp_coupling_r53.common import array_hash

IDENTITIES=['source_uid','audio_sha256','raw_motion_sha256','group','recording_uid','choreography_uid']
def select(meta,old):
    olduids={r['source_uid'] for r in old['sources']};past=[s for s in meta['sources'] if s['source_uid'] in olduids]
    blocked={k:{s[k] for s in past if s.get(k) not in [None,'UNKNOWN']} for k in IDENTITIES};chosen=[];audit=[]
    for dataset,role in CFG['roles'].items():
        seen={k:set() for k in IDENTITIES};n=0
        for s in sorted(meta['sources'],key=lambda s:s['source_uid']):
            if (s['dataset'],s['role'])!=(dataset,role):continue
            conflicts=[k for k in IDENTITIES if s.get(k) not in [None,'UNKNOWN'] and (s[k] in blocked[k] or s[k] in seen[k])]
            if conflicts:audit.append(dict(source_uid=s['source_uid'],exclusion=conflicts));continue
            chosen.append(s);n+=1
            for k in IDENTITIES:
                if s.get(k) not in [None,'UNKNOWN']:seen[k].add(s[k])
            if n==CFG['groups_per_domain']:break
    return dict(sources=chosen,selection=CFG['selection'],exclusions=audit,requested_per_domain=CFG['groups_per_domain'],old_source_uids=sorted(olduids),known_link_isolation=True,unknown_links_certified=False,independent_generalization_certified=False,historical_exposure=CFG['historical_exposure'],source_metadata_sha256=sha(LEGACY['data']))

def register():
    assert sha(LEGACY['data'])==LEGACY['data_sha256'];return select(read(LEGACY['data']),read('outputs/music_events_01/paired/population.json'))

def infer(replay=False):
    import os,torch,soundfile as sf
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    from dunhuang.research.music_events.infer import assets
    dst=ROOT/('audio_replay' if replay else 'audio');dst.mkdir(exist_ok=False);reg=read(ROOT/'registration.json')['paired'];model=assets();torch.manual_seed(76001);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tracker=Audio2Frames(model['beat_this_checkpoint'],device='cuda',float16=False);rows=[]
    for s in reg['sources']:
        try:
            path=s['audio']['path'];assert sha(path)==s['audio_sha256'];y,sr=sf.read(path,always_2d=True,dtype='float32');logits=tracker(y.mean(1),sr);beat,down=Postprocessor(type='minimal',fps=50)(*logits)
            rows.append(dict(source_uid=s['source_uid'],audio_sha256=s['audio_sha256'],sample_rate=sr,audio_samples=len(y),duration_seconds=len(y)/sr,beat=np.asarray(beat).tolist(),downbeat=np.asarray(down).tolist(),activation_sha256=[array_hash(t.sigmoid().cpu().numpy()) for t in logits],model_manifest_sha256=sha('outputs/music_events_01/assets/model_manifest.json'),status='OBSERVED'))
        except Exception as e:rows.append(dict(source_uid=s['source_uid'],status='CONTRACT_FAILURE',error=repr(e),replacement_used=False))
        print('Q02_BEAT_INFER',s['source_uid'],rows[-1]['status'],flush=True)
    write(dst/'events.json',rows)
    if replay:
        diff=[] if sha(dst/'events.json')==sha(ROOT/'audio/events.json') else ['events.json'];write(ROOT/'receipts/audio_replay.json',dict(status='PASS' if not diff else 'FAIL',differences=diff,fresh_process=True));assert not diff,diff

def energy_scores(s,P,start,end,donor):
    import soundfile as sf
    t=np.arange(np.ceil(start*30),np.floor(end*30)+1)/30;speed=np.linalg.norm(np.diff(P,axis=0),axis=-1).mean(-1)*30;offset=s['audio_offset_frames']/30
    v=np.interp(t,(np.arange(len(speed))+.5)/30+offset,speed)
    def series(path):
        y,sr=sf.read(path,always_2d=True,dtype='float32');y=y.mean(1);hop=max(1,int(sr/30));n=len(y)//hop;rms=np.sqrt(np.mean(y[:n*hop].reshape(n,hop)**2,axis=1));return (np.arange(n)+.5)*hop/sr,rms
    at,amp=series(s['audio']['path']);dt,damp=series(donor['audio']['path'])
    def corr(a):
        if np.std(a)<1e-12 or np.std(v)<1e-12:return None
        return float(np.corrcoef(a,v)[0,1])
    return [corr(np.interp(t+d,at,amp)) for d in CFG['offsets']],corr(np.interp(t,dt,damp))

def evaluate(out):
    ctx=prepare();reg=read(ROOT/'registration.json')['paired'];cats={r['source_uid']:r for r in read(ROOT/'audio/events.json')};rows=[]
    for i,s in enumerate(reg['sources']):
        try:
            own=cats[s['source_uid']];assert own['status']=='OBSERVED';R,proof=engine.load_motion(s);P=relative_fk(R,ctx['J'],ctx['parents']);beats=music.motion_beats(P)+s['audio_offset_frames']/30
            donor=next(v for v in reg['sources'][i+1:]+reg['sources'][:i] if v['dataset']==s['dataset'] and v['audio_sha256']!=s['audio_sha256']);wrong=cats[donor['source_uid']];assert wrong['status']=='OBSERVED'
            lo,hi=proof['shared_source_interval'];start=max(0,lo/30+s['audio_offset_frames']/30)+.5;end=min((hi-1)/30+s['audio_offset_frames']/30,own['duration_seconds'],wrong['duration_seconds'])-.5
            assert end>start;beats=beats[(beats>=start-.5)&(beats<=end+.5)];query=np.asarray(own['beat']);query=query[(query>=start)&(query<end)];w=np.asarray(wrong['beat']);w=w[(w>=start)&(w<end)]
            scores=[music.gaussian_match(query+d,beats) for d in CFG['offsets']];wrongscore=music.gaussian_match(w,beats);record=dict(source_uid=s['source_uid'],dataset=s['dataset'],role=s['role'],group=s['group'],audio_group=s['audio_sha256'],choreography_uid=s.get('choreography_uid','UNKNOWN'),clock_scope=s['clock_scope'],dancer_role=s.get('dancer_role','UNKNOWN'),status='OBSERVED',source_proof=proof,motion_sha256=array_hash(R),support_seconds=[start,end],query_count=len(query),motion_minima_count=len(beats),wrong_query_count=len(w),wrong_audio_sha256=donor['audio_sha256'],offsets=CFG['offsets'],scores=scores,ranking=rank(scores),six_candidate_ranking=rank(scores+[wrongscore]),wrong_song_score=wrongscore,correct_vs_wrong=None if scores[2] is None or wrongscore is None else 1 if scores[2]>wrongscore+1e-12 else .5 if abs(scores[2]-wrongscore)<=1e-12 else 0)
            dt=None if len(query)<2 else float(np.median(np.diff(query)));record['median_beat_period_seconds']=dt;record['period_ambiguity']=[] if dt is None else [dict(offset=d,distance_to_integer_beat_cycles_seconds=float(abs(d-dt*np.rint(d/dt))),within_0_1_second=bool(abs(d-dt*np.rint(d/dt))<=.1)) for d in CFG['offsets'] if d]
            # No null is fitted: all four grid phases are retained, same motion-event count.
            grids=[]
            for phase in CFG['periodic_grid_phases']:
                grid=np.array([]) if not len(beats) else start-.5+(np.arange(len(beats))+phase)*(end-start+1)/len(beats)
                ss=[music.gaussian_match(query+d,grid) for d in CFG['offsets']];grids.append(dict(phase=phase,scores=ss,ranking=rank(ss),motion_event_count=len(grid)))
            energy,ew=energy_scores(s,P,start,end,donor);record.update(equal_density_grid=grids,energy_scores=energy,energy_ranking=rank(energy),wrong_song_energy=ew,artistic_quality='NOT_EVALUABLE');rows.append(record)
        except Exception as e:rows.append(dict(source_uid=s['source_uid'],dataset=s['dataset'],status='CONTRACT_FAILURE',error=repr(e),replacement_used=False))
        print('Q02_CLOCK_GROUP',s['source_uid'],rows[-1]['status'],flush=True)
    summary=[];rng=np.random.default_rng(CFG['bootstrap_seed'])
    for dataset in CFG['roles']:
        own=[r for r in rows if r['dataset']==dataset and r['status']=='OBSERVED'];valid=[r for r in own if r['ranking']['status']=='EVALUABLE' and r['energy_ranking']['status']=='EVALUABLE' and r['equal_density_grid'][0]['ranking']['status']=='EVALUABLE']
        ci=lambda v:None if len(v)<2 else np.quantile(rng.choice(v,(CFG['bootstrap_draws'],len(v)),replace=True).mean(1),[.025,.975]).tolist()
        mean=lambda v:None if not len(v) else float(np.mean(v));diff_e=[r['ranking']['tied_top1_credit']-r['energy_ranking']['tied_top1_credit'] for r in valid];diff_g=[r['ranking']['tied_top1_credit']-r['equal_density_grid'][0]['ranking']['tied_top1_credit'] for r in valid];ce,cg=ci(diff_e),ci(diff_g)
        summary.append(dict(dataset=dataset,requested=CFG['groups_per_domain'],registered=sum(s['dataset']==dataset for s in reg['sources']),observed=len(own),complete_cases=len(valid),strict_top1_correct=sum(r['ranking']['strict_top1'] for r in valid),tied_top1_mean=mean([r['ranking']['tied_top1_credit'] for r in valid]),paired_win_mean=mean([r['ranking']['paired_win'] for r in valid]),energy_top1=mean([r['energy_ranking']['tied_top1_credit'] for r in valid]),grid_phase0_top1=mean([r['equal_density_grid'][0]['ranking']['tied_top1_credit'] for r in valid]),top1_minus_energy_95=ce,top1_minus_grid_95=cg,qualifies=bool(len(valid)==CFG['groups_per_domain'] and ce is not None and cg is not None and ce[0]>0 and cg[0]>0),independent_generalization_certified=False))
    write(out/'paired_scores.json',rows);write(out/'paired_summary.json',summary);return summary

if __name__=='__main__':
    import sys
    infer('--replay' in sys.argv)
