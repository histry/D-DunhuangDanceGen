"""Verified-review importer and tie-aware human metrics. Missing/UNSURE never become negatives."""
import argparse,collections,datetime,math,shutil
from pathlib import Path
import numpy as np
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.sdmr_event_rank.evaluate import group_ci

DECISIONS=dict(EVENT={'VALID','INVALID','UNSURE'},POSE={'VALID','INVALID','UNSURE'},MUSIC_MATCH={'HIGH','ACCEPTABLE','INCOMPATIBLE','UNSURE'},QUERY_MATCH={'HAS_SUITABLE_CANDIDATE','NO_SUITABLE_CANDIDATE','UNSURE'})

def validate(record,items,queries,roster):
    if record.get('demonstration') or record.get('status')!='UNVERIFIED_CLIENT_RECORDS':raise ValueError('Demonstration or unsupported client record')
    task=record.get('task_type');reviewer=roster.get(record.get('reviewer_id'))
    if task not in DECISIONS or record.get('decision') not in DECISIONS[task]:raise ValueError('Task/decision scope')
    if not reviewer or not reviewer.get('identity_verified') or not reviewer.get('independent_from_model_development') or not reviewer.get('reviewer_roster_ref'):raise ValueError('Unaudited reviewer identity/independence')
    if not record.get('review_id') or record.get('confidence') not in (1,2,3) or not record.get('reason_code'):raise ValueError('Missing review evidence')
    if record['decision']=='UNSURE' and record['reason_code']=='OBSERVABLE_EVIDENCE':raise ValueError('UNSURE needs an uncertainty reason')
    if record['reason_code']=='OTHER' and not record.get('note','').strip():raise ValueError('OTHER needs a reason note')
    when=datetime.datetime.fromisoformat(record['reviewed_at'].replace('Z','+00:00'))
    if when.tzinfo is None or when>datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=1):raise ValueError('Auditable review timestamp')
    item=queries.get(record['blind_item_id']) if task=='QUERY_MATCH' else items.get(record['blind_item_id'])
    if item is None:raise ValueError('Unknown blind identity')
    expected=item['query_media_ref'] if task=='QUERY_MATCH' else item['media_ref'][task]
    if record.get('media_ref')!=expected:raise ValueError('Media evidence hash mismatch')
    if record.get('review_protocol_version')!='DUNHUANG_BLIND_REVIEW_V1_R45':raise ValueError('Protocol version')
    return record

def resolve(records,adjudication=None):
    ids=[r['reviewer_id'] for r in records]
    if len(ids)!=len(set(ids)):return dict(state='PARTIAL_REVIEW',reason='MULTIPLE_UNRESOLVED_VERSIONS_FROM_SAME_REVIEWER',decision=None)
    if len(ids)<2:return dict(state='PARTIAL_REVIEW',reason='NEEDS_TWO_INDEPENDENT_REVIEWERS',decision=None)
    decisions=[r['decision'] for r in records]
    if len(set(decisions))==1:return dict(state='UNSURE' if decisions[0]=='UNSURE' else 'HUMAN_REVIEW_VERIFIED',decision=decisions[0],review_ids=[r['review_id'] for r in records])
    if adjudication and adjudication['reviewer_id'] not in ids and set(adjudication.get('resolves_review_ids',[]))=={r['review_id'] for r in records}:
        return dict(state='UNSURE' if adjudication['decision']=='UNSURE' else 'HUMAN_REVIEW_VERIFIED',decision=adjudication['decision'],review_ids=[r['review_id'] for r in records],adjudication_id=adjudication['review_id'])
    return dict(state='PARTIAL_REVIEW',reason='DISAGREEMENT_NEEDS_THIRD_INDEPENDENT_ADJUDICATION',decision=None,review_ids=[r['review_id'] for r in records])

def ndcg(scores,relevance,k):
    scores=np.asarray(scores,float);rel=np.asarray(relevance,float)
    if scores.shape!=rel.shape or scores.ndim!=1 or not np.isfinite(scores).all() or np.any(rel<0):raise ValueError('Finite complete judged ranking')
    k=min(k,len(scores));discount=1/np.log2(np.arange(2,k+2));ideal=float(np.sum((2**np.sort(rel)[::-1][:k]-1)*discount))
    if ideal==0:return None
    order=np.argsort(-scores,kind='stable');gain=2**rel-1;dcg=0.;rank=0
    while rank<len(order):
        tail=order[rank:];tie=tail[np.isclose(scores[tail],scores[order[rank]],atol=1e-7,rtol=0)];dcg+=float(gain[tie].mean()*discount[rank:min(rank+len(tie),k)].sum()) if rank<k else 0.;rank+=len(tie)
    return dcg/ideal

def run(package,scores,out,reviews=None,roster=None,adjudications=None):
    package=Path(package);scores=Path(scores);out=Path(out)
    if out.exists():raise ValueError('Fresh review evaluation')
    out.mkdir(parents=True);meta=read(package/'admin/manifest.json');sm=read(scores/'result.json');assert sm['review_manifest_sha256']==sha(package/'admin/manifest.json');items={r['blind_item_id']:r for r in meta['items']};queries={r['query_id']:r for r in meta['queries']};scored={r['key']:r for r in sm['records']};z=Arrays(scores/'arrays.npz');hub={};availability=collections.Counter()
    for pool in ('full','balanced'):
        counts={m:collections.Counter() for m in sm['methods']};available=collections.Counter();ties={m:0 for m in sm['methods']}
        for r in sm['records']:
            ids=np.arange(len(r['candidate_source_uids'])) if pool=='full' else np.array(r['balanced_candidates']);table=z[r['key']+'_scores'][ids]
            for i in ids:available[r['candidate_source_uids'][int(i)]]+=1
            for j,m in enumerate(sm['methods']):
                top=np.flatnonzero(np.isclose(table[:,j],table[:,j].max(),atol=1e-7,rtol=0));ties[m]+=len(top)>1
                for i in top:counts[m][r['candidate_source_uids'][int(ids[i])]]+=1/len(top)
        hub[pool]=dict(availability=dict(available),methods={m:dict(fractional_top1=dict(v),top1_per_available_candidate={s:v[s]/available[s] for s in available},tied_queries=ties[m]) for m,v in counts.items()})
    raw=[];valid=[];quarantined=[];resolved={};agreement={};metrics=[];raw_hash=None
    if reviews:
        if not roster:raise ValueError('Real reviews need an independently auditable roster')
        registry=read(roster);people={r['reviewer_id']:r for r in registry['reviewers']}
        for p in people.values():checked(p['authentication_evidence'])
        raw=read(reviews)['records'];raw_hash=sha(reviews);shutil.copyfile(reviews,out/'raw_reviews.json');seen=set()
        for r in raw:
            try:
                if r.get('review_id') in seen:raise ValueError('Duplicate review ID')
                seen.add(r.get('review_id'));valid.append(validate(r,items,queries,people))
            except (ValueError,KeyError,TypeError) as e:quarantined.append(dict(review_id=r.get('review_id'),reason=str(e)))
        grouped=collections.defaultdict(list)
        for r in valid:grouped[r['blind_item_id'],r['task_type']].append(r)
        adj={}
        if adjudications:
            for r in read(adjudications)['records']:
                validate(r,items,queries,people);adj[r['blind_item_id'],r['task_type']]=r
        for key,value in grouped.items():resolved['/'.join(key)]=resolve(value,adj.get(key))
        for task in DECISIONS:
            pairs=[]
            for (item,t),value in grouped.items():
                if t==task:
                    unique={r['reviewer_id']:r for r in value}
                    for i,a in enumerate(unique.values()):
                        for b in list(unique.values())[i+1:]:pairs.append(a['decision']==b['decision'])
            agreement[task]=dict(pair_comparisons=len(pairs),observed_agreement=float(np.mean(pairs)) if pairs else None,unsure_records=sum(r['task_type']==task and r['decision']=='UNSURE' for r in valid))
        byquery=collections.defaultdict(list)
        for item in meta['items']:byquery[item['query_id']].append(item)
        for qid,part in byquery.items():
            judgments=[resolved.get(p['blind_item_id']+'/MUSIC_MATCH') for p in part]
            if not all(v and v['state']=='HUMAN_REVIEW_VERIFIED' for v in judgments):continue
            rel=np.array([{'HIGH':2,'ACCEPTABLE':1,'INCOMPATIBLE':0}[v['decision']] for v in judgments]);row=scored[part[0]['key']];table=z[row['key']+'_scores'][[p['candidate'] for p in part]];none=resolved.get(qid+'/QUERY_MATCH');conflict=bool(none and none['state']=='HUMAN_REVIEW_VERIFIED' and ((none['decision']=='NO_SUITABLE_CANDIDATE')!=bool(rel.max()==0)))
            if conflict:continue
            result={}
            for j,m in enumerate(sm['methods']):
                pref=[]
                for a in range(len(rel)):
                    for b in range(a+1,len(rel)):
                        if rel[a]!=rel[b]:pref.append(.5 if abs(table[a,j]-table[b,j])<=1e-7 else float((table[a,j]-table[b,j])*(rel[a]-rel[b])>0))
                result[m]=dict(ndcg_at1=ndcg(table[:,j],rel,1),ndcg_at3=ndcg(table[:,j],rel,3),ndcg_at5=ndcg(table[:,j],rel,5),preference_agreement=float(np.mean(pref)) if pref else None)
            metrics.append(dict(query_id=qid,song=row['song'],methods=result,none_suitable_truth=none['decision'] if none and none['state']=='HUMAN_REVIEW_VERIFIED' else None,partial_all_null_rejection=all(row['partial_coverage'][p['candidate']]==0 for p in part)))
    summary={}
    for method in sm['methods']:
        for metric in ('ndcg_at1','ndcg_at3','ndcg_at5','preference_agreement'):
            groups=collections.defaultdict(list)
            for r in metrics:
                v=r['methods'][method][metric]
                if v is not None:groups[r['song']].append(v)
            values={g:float(np.mean(v)) for g,v in groups.items()};summary[method+'/'+metric]=dict(song_means=values,mean=float(np.mean(list(values.values()))) if values else None,ci95=group_ci(list(values.values()),45045,5000))
    refusal=[r for r in metrics if r['none_suitable_truth'] is not None];tp=sum(r['partial_all_null_rejection'] and r['none_suitable_truth']=='NO_SUITABLE_CANDIDATE' for r in refusal);fp=sum(r['partial_all_null_rejection'] and r['none_suitable_truth']!='NO_SUITABLE_CANDIDATE' for r in refusal);fn=sum(not r['partial_all_null_rejection'] and r['none_suitable_truth']=='NO_SUITABLE_CANDIDATE' for r in refusal)
    coverage={}
    for task in DECISIONS:
        expected=queries if task=='QUERY_MATCH' else items;states=collections.Counter(resolved.get(i+'/'+task,dict(state='UNREVIEWED'))['state'] for i in expected);coverage[task]=dict(registered=len(expected),states=dict(states))
    all_tasks=all(coverage[t]['states'].get('HUMAN_REVIEW_VERIFIED',0)==len(items) for t in ('EVENT','POSE','MUSIC_MATCH'))
    write(out/'result.json',dict(status='HUMAN_METRICS_PARTIAL_COVERAGE' if metrics else 'NOT_EVALUABLE_NO_VERIFIED_COMPLETE_MUSIC_JUDGMENTS',registered_items=len(items),registered_queries=len(queries),coverage=coverage,source_balanced=True,hubness=hub,raw_review_count=len(raw),valid_independent_records=len(valid),quarantined=quarantined,consensus=resolved,agreement=agreement,metrics=metrics,summary=summary,refusal=dict(evaluable_queries=len(refusal),rule='REGISTERED_ALL_NULL_PARTIAL_COVERAGE_ONLY; NOT_FITTED_TO_LABELS',precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None),raw_reviews_sha256=raw_hash,raw_records_preserved=True,unknown_not_negative=True,human_reviewed=bool(metrics),whole_dataset_human_qualified=all_tasks and len(metrics)==len(queries),new_independent_music_count=0,optimizer_steps=0,can_route=False,operational_qualified=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--scores',required=True);p.add_argument('--out',required=True);p.add_argument('--reviews');p.add_argument('--roster');p.add_argument('--adjudications');a=p.parse_args();run(a.package,a.scores,a.out,a.reviews,a.roster,a.adjudications)
