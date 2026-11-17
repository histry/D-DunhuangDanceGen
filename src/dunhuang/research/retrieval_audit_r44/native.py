"""Frozen Native candidates: cost-scale audit, tie-aware hubness, blinded preparation."""
import argparse,collections,hashlib
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from .path import partial

def audit(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh Native audit')
    out.mkdir(parents=True);root=Path('outputs/sdmr_event_rank_01/change_shadow_v3');meta=read(root/'result.json');z=Arrays(root/'arrays.npz');cfg=read('src/dunhuang/research/sdmr_event_rank/config.json');available=collections.Counter();tops={m:collections.Counter() for m in ('Global','Diagonal','Offset','Partial','Hybrid')};records=[];blind=[];admin=[];arrays={}
    for row in meta['records']:
        key=row['key'];table=np.asarray(row['variants']['actual']);q={n:z[key+'_c0_'+n] for n in ('mt','dt','mv','dv','positions')};v=[];similarities=[]
        for j,candidate in enumerate(row['candidates']):
            c={n:z[key+f'_c{j}_'+n] for n in ('mt','dt','mv','dv','positions')};score,cov,cost=partial(q,c,cfg);assert abs(score-table[j,3])<1e-6 and abs(cov-table[j,5])<1e-12
            stride=cfg['token_stride'];s=q['mt'][::stride]@c['dt'][::stride].T;distance=abs(q['positions'][::stride,None]-c['positions'][None,::stride]);mask=(distance<=cfg['path_band_frames'])&q['mv'][::stride,None]&c['dv'][None,::stride];sim=s[mask];similarities.extend(sim.tolist());v.append([score,cov,cost,float(np.mean(sim>.4))]);available[candidate['source_uid']]+=1
            blind_id=hashlib.sha256(('R44_BLIND:'+key+':'+candidate['event_uid']).encode()).hexdigest()[:20];blind.append(dict(blind_item_id=blind_id,motion_array_key=blind_id,query_audio_sha256=row['query_audio']['sha256'],audio_frame_interval=row['audio_frame_interval'],tasks=['EVENT','POSE','MUSIC_MATCH'],decision=None,status='UNREVIEWED',human_reviewed=False));admin.append(dict(blind_item_id=blind_id,key=key,candidate=j,provenance=candidate));arrays[blind_id]=z[key+f'_c{j}_xyz']
        top={}
        for j,m in enumerate(tops):
            ids=np.flatnonzero(np.isclose(table[:,j],table[:,j].max(),atol=1e-7,rtol=0));top[m]=dict(tied_candidates=ids.tolist(),coverage=float(table[ids,5].mean()))
            for i in ids:tops[m][row['candidates'][int(i)]['source_uid']]+=1/len(ids)
        records.append(dict(key=key,candidates=len(row['candidates']),within_band_similarity_quantiles=np.quantile(similarities,[0,.05,.5,.95,1]).tolist(),fraction_similarity_above_local_match_threshold=float(np.mean(np.array(similarities)>.4)),all_null_candidates=sum(vv[1]==0 for vv in v),top=top));arrays[key+'_path_audit']=np.asarray(v)
    hub={m:dict(fractional_top1=dict(count),top1_per_available_candidate={u:count[u]/available[u] for u in available},mean_selected_coverage=float(np.mean([r['top'][m]['coverage'] for r in records]))) for m,count in tops.items()}
    np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,hubness=hub,availability=dict(available),parent_manifest_sha256=sha(root/'result.json'),parent_arrays_sha256=sha(root/'arrays.npz'),local_indifference_similarity=.4,costs=dict(match='1-similarity',drop_music=.3,drop_motion=.3),recurrence_matches_frozen_all_candidates=True,scope='LOCAL_THRESHOLD_NOT_A_FULL_PATH_SUFFICIENCY_RULE; NO_SEMANTIC_ACCURACY',optimizer_steps=0,can_route=False,human_reviewed=False,operational_qualified=False));write(out/'blind_items.json',dict(status='PREPARATION_ONLY',human_reviewed=False,semantic_event_truth='UNKNOWN',items=sorted(blind,key=lambda r:r['blind_item_id']),instructions='Two independent reviewers; hide model/source/role/scores; allow UNSURE; retain raw reviews and adjudication; source map kept separately. No generated label is human truth.'));write(out/'admin_map.json',admin);z.close();print('R44_NATIVE',len(records),len(admin),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();audit(a.out)
