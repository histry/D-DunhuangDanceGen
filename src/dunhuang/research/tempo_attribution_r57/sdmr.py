"""R55 score-base/gradient audit, no new parameter update or unqualified routing."""
from collections import defaultdict
from .common import *
def preserved_scores(raw,base,weight=.25):return base.detach()+weight*(raw-base.detach())
def run(out):
 import torch
 from torch.nn import functional as F
 from dunhuang.research.coverage_r34.common import forbid_optimizers
 from dunhuang.research.clock_residual_r55.study import Study as Data
 from dunhuang.research.clock_residual_r55.common import CFG as OLD,ranking_loss,ROOT as OLDROOT
 from dunhuang.research.paired_event_r41.train import seeded
 from dunhuang.research.sdmr_event_rank.scorer import retrieval_metrics
 from dunhuang.research.residual_sdmr_r54.common import relation
 out=fresh(out);forbid_optimizers();s=Data();params=[n for n,p in s.model().named_parameters() if p.requires_grad];tasks=s.meta['tasks'];chosen=[]
 for group in sorted(s.split['optimization']):
  options=[i for i,t in enumerate(tasks) if t['group']==group and t['training_admitted']];chosen.append(min(options,key=lambda i:tasks[i]['uid']))
 records=[];arrays={};models={}
 for arm in CFG['gradient_models']:
  for step in ([0] if arm=='Frozen' else CFG['gradient_steps']):
   net=s.model();checkpoint=OLDROOT/('Conditional' if arm=='Frozen' else arm)/f'step{step}_pre.pt';net.load_state_dict(torch.load(checkpoint,map_location=s.device,weights_only=True)['state']);net.eval();models[arm+str(step)]=dict(path=str(checkpoint),sha256=sha(checkpoint));ps=[p for p in net.parameters() if p.requires_grad]
   for task_id in chosen:
    t=tasks[task_id];seeded(OLD['seed']+step);raw=s.raw(net,t);base=s.base_scores(t);rank=ranking_loss(raw,base,t['heuristic'],t['positive']);p=torch.tensor(t['positive'],device=s.device);score=preserved_scores(raw,base);teacher=F.softplus(-(score[p,None]-score[None,~p])/OLD['temperature']).mean()+OLD['residual_penalty']*(raw-base).square().mean();shift=OLD['offsets'][(step+t['query'])%len(OLD['offsets'])];truth=OLD['offsets'].index(-shift);cr=s.clock(net,t['query'],shift);cb=s.clock_base(t['query'],shift);clock=ranking_loss(cr,cb,np.zeros(len(cb)),np.arange(len(cb))==truth)
    def grad(loss,retain):
     gs=torch.autograd.grad(loss,ps,retain_graph=retain,allow_unused=True);return torch.cat([(torch.zeros_like(q) if g is None else g).reshape(-1) for q,g in zip(ps,gs)]).detach().double()
    gr=grad(rank,True);gt=grad(teacher,False);gc=grad(clock,False)
    def compare(g):
     a=float(g.norm());b=float(gc.norm());return dict(rank_norm=a,clock_norm=b,weighted_clock_to_rank=OLD['clock_weight']*b/a if a>1e-12 else None,cosine=float(torch.dot(g,gc)/(g.norm()*gc.norm())) if a*b>1e-12 else None)
    records.append(dict(arm=arm,step=step,group=t['group'],task_uid=t['uid'],task_id=task_id,shift=shift,frames=t['frames'],legacy_rank=compare(gr),teacher_preserved_rank=compare(gt),loss_rank=float(rank.detach()),loss_clock=float(clock.detach()),gradient_sha256={k:array_hash(g.cpu().numpy()) for k,g in [('rank',gr),('teacher_rank',gt),('clock',gc)]}))
   print('R57_SDMR_GRAD',arm,step,len(chosen),flush=True);del net
 # Exact algebraic rescore of SAVED R55 outputs. This is NOT retraining.
 oldnpz=np.load(OLDROOT/'evaluation/scores.npz') if (OLDROOT/'evaluation/scores.npz').exists() else None
 if oldnpz is None:
  files=[p for p in (OLDROOT/'evaluation').glob('*.npz') if 'calibration' not in p.name];assert len(files)==1;oldnpz=np.load(files[0])
 groups=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(list))))
 for i,t in enumerate(tasks):
  H=oldnpz[f'Heuristic_{i}'];base=oldnpz[f'Frozen_{i}'];scores=dict(Heuristic=H,Frozen=base)
  for arm in ('Conditional','NoClock'):
   legacy=oldnpz[f'{arm}_{i}'];scores[arm]=legacy;scores[arm+'_TeacherRescore']=base+(legacy-H)
  for name,values in scores.items():
   arrays[f'{name}_{i}']=values;metric=retrieval_metrics(values,t['positive']);role=next(k for k,v in s.split.items() if t['group'] in v);groups[role+'_'+t['family']][name][t['group']][t['recording']].append(metric)
 rescore={}
 for role,methods in groups.items():
  rescore[role]={name:{g:{key:float(np.mean([np.mean([v[key] for v in values]) for values in recordings.values()])) for key in ('recall_at1','recall_at5','mrr')} for g,recordings in by.items()} for name,by in methods.items()}
 # Candidate pool depends only on role, frame count and available pre-existing tensors.
 # It does NOT use positive bins. UNKNOWN relations remain in ranked pool.
 available=sorted(int(k[1:]) for k in s.z.files if k.startswith('M') and k[1:].isdigit());openrows=[]
 for group in sorted(s.split['val']):
  for length in (90,150,240):
   queries=[i for i in available if s.meta['rows'][i]['source_group']==group and s.meta['rows'][i]['frames']==length]
   if not queries:continue
   q=min(queries,key=lambda i:s.meta['rows'][i]['uid']);qr=s.meta['rows'][q];pool=[i for i in available if s.meta['rows'][i]['role']=='val' and s.meta['rows'][i]['frames']==length and s.meta['rows'][i]['raw_motion_sha256']!=qr['raw_motion_sha256']];assert pool;relations=[relation(qr,s.meta['rows'][j]) for j in pool];positive=np.array([x=='CLOCK_POSITIVE' for x in relations]);task=dict(query=q,candidates=pool,donor=q);h=-np.abs(qr['music_activity']-np.array([s.meta['rows'][j]['body_activity'] for j in pool]));scores={}
   with torch.no_grad():scores['Frozen']=s.raw(s.base,task).cpu().numpy();scores['Heuristic']=h
   for name,v in scores.items():
    order=np.argsort(-v,kind='stable');arrays[f'open_{q}_{name}']=v;openrows.append(dict(query=q,group=group,frames=length,method=name,pool=pool,pool_size=len(pool),positive_count=int(positive.sum()),unknown_count=relations.count('UNKNOWN'),top1_relation=relations[int(order[0])],source_clock_target_metrics=retrieval_metrics(v,positive) if positive.any() and not positive.all() else None,shortlist_5_clock_positive_coverage=float(positive[order[:5]].any()) if positive.any() else None,ground_truth_scope='SOURCE_CLOCK_TARGET_NOT_ARTISTIC_CORRECTNESS',pool_scope='ALL_AVAILABLE_R54_TENSORS; HISTORICAL_TENSOR_POPULATION_WAS_TASK_SELECTED; NOT_FULL_RAW_DATABASE',positive_identity_used_for_candidate_selection=False))
   print('R57_OPEN_POOL',group,length,len(pool),flush=True)
 summary={}
 for arm in CFG['gradient_models']:
  for step in ([0] if arm=='Frozen' else CFG['gradient_steps']):
   rr=[r for r in records if r['arm']==arm and r['step']==step];summary[arm+str(step)]={kind:dict(groups=len(rr),negative_cosine_fraction=float(np.mean([r[kind]['cosine']<0 for r in rr if r[kind]['cosine'] is not None])),median_cosine=float(np.median([r[kind]['cosine'] for r in rr if r[kind]['cosine'] is not None]))) for kind in ('legacy_rank','teacher_preserved_rank')}
 result=dict(binding=binding(),parent_checkpoint_models=models,parent_evaluation_sha256=sha(OLDROOT/'evaluation/result.json'),parent_population_audit=s.population_audit,partition=s.split,parameter_names=params,gradient_records=records,gradient_summary=summary,rescore_group_metrics=rescore,open_pool=openrows,optimizer_calls=0,training_updates=0,gradient_conflict_is_not_causal_proof=True,teacher_rescore_not_retrained=True,new_music_independence=False,FineDance_admitted=False,can_route=False,semantic_event_truth='UNKNOWN');write(out/'result.json',result);save_arrays(out/'arrays.npz',arrays);return result
def verify(out,source):
 out=fresh(out);old=Path(source);expected=read(old/'result.json');before=binding();assert before['code']==expected['binding']['code'];actual=run(out/'rebuild');actual['binding']['commit']=expected['binding']['commit'];assert actual==expected;assert sha(old/'arrays.npz')==sha(out/'rebuild/arrays.npz');write(out/'result.json',dict(result_exact=True,arrays_bytes_exact=True,all_gradient_hashes_exact=True,optimizer_calls=0,source_commit=expected['binding']['commit'],replay_commit=before['commit']))
