"""All models frozen before historical Val is opened; music-group inference only."""
from collections import defaultdict
import torch
from .common import *
from .learn import Study
from dunhuang.research.sdmr_event_rank.scorer import retrieval_metrics

def summarize(tasks,scores):
 results={}
 for role in ('train','val'):
  for family in ('cross_music','same_music'):
   ids=[i for i,t in enumerate(tasks) if t['role']==role and t['family']==family];groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)));individual={}
   for method in scores:
    for i in ids:
     t=tasks[i];v=retrieval_metrics(scores[method][i],t['positive']);individual[method,i]=v
     for metric in ('recall_at1','recall_at5','mrr','uniform_candidate_chance'):groups[method][metric][t['group']].append((t['recording'],v[metric]))
   means={}
   for method,metrics in groups.items():
    means[method]={}
    for metric,gg in metrics.items():
     gm={}
     for group,items in gg.items():
      rec=defaultdict(list)
      for uid,v in items:rec[uid].append(v)
      gm[group]=float(np.mean([np.mean(v) for v in rec.values()]))
     means[method][metric]=gm
   def ci(values):
    if len(values)<2:return [None,None]
    values=np.array(values);rng=np.random.default_rng(CFG['seed']);v=values[rng.integers(0,len(values),(CFG['bootstrap_draws'],len(values)))].mean(1);return np.quantile(v,[.025,.975]).tolist()
   aggregate={m:{k:dict(mean=float(np.mean(list(g.values()))),ci95=ci(list(g.values())),music_groups=len(g)) for k,g in mm.items()} for m,mm in means.items()}
   comparisons={}
   for method in ('Heuristic','Frozen','HybridFrozen','Unconditional','Shuffled','WrongMusic'):
    if method not in means:continue
    a,b=means['Conditional']['recall_at1'],means[method]['recall_at1'];keys=sorted(a.keys()&b.keys());delta=[a[g]-b[g] for g in keys];comparisons[method]=dict(delta=float(np.mean(delta)) if delta else None,ci95=ci(delta),music_groups=len(keys))
   results[role+'_'+family]=dict(tasks=len(ids),methods=aggregate,paired_differences=comparisons,group_R1={m:r['recall_at1'] for m,r in means.items()})
 return results

def run(out):
 out=fresh(out)
 for arm in ('Conditional','Unconditional','Shuffled'):assert read(ROOT/arm/'result.json')['updates']==CFG['steps']
 s=Study();s.frozen_inference=True;tasks=s.meta['tasks'];scores=defaultdict(dict);arrays={};models={}
 for arm in ('Conditional','Unconditional','Shuffled'):
  net=s.model();p=ROOT/arm/'model.pt';assert sha(p)==read(ROOT/arm/'result.json')['model_sha256'];net.load_state_dict(torch.load(p,map_location=s.device,weights_only=True)['state']);models[arm]=net.eval()
 assert len({read(ROOT/a/'result.json')['initial_digest'] for a in models})==1
 assert len({read(ROOT/a/'result.json')['schedule_sha256'] for a in models})==1
 for i,t in enumerate(tasks):
  for arm,net in models.items():
   actual=s.evaluate(net,t);scores[arm][i]=actual['Conditional']
   if arm=='Conditional':
    for method in ('Heuristic','Frozen','HybridFrozen'):scores[method][i]=actual[method]
    scores['ResidualOnly'][i]=actual['Residual'];scores['WrongMusic'][i]=s.evaluate(net,t,'wrong_music')['Conditional']
  for method,bytask in scores.items():arrays[method+'_'+str(i)]=bytask[i]
  if (i+1)%100==0:print('R54_EVAL',i+1,len(tasks),flush=True)
 summary=summarize(tasks,scores);main=summary['val_cross_music'];paired=main['paired_differences'];gate=len(s.meta['groups']['val'])>=CFG['minimum_val_groups'] and all(k in paired and paired[k]['ci95'][0] is not None and paired[k]['ci95'][0]>0 for k in ('Heuristic','Frozen','HybridFrozen','Unconditional','Shuffled','WrongMusic'));same=summary['val_same_music'];same_ok=bool(same['tasks'] and same['paired_differences']['Heuristic']['ci95'][0] is not None and same['paired_differences']['Heuristic']['ci95'][0]>0)
 save_arrays(out/'scores.npz',arrays);write(out/'result.json',dict(summary=summary,A3_increment_gate=bool(gate),A4_admitted=bool(gate and same_ok),A5_admitted=False,scope='SOURCE_CLOCK_ASSOCIATION_ON_SIX_EXPOSED_VAL_MUSIC_GROUPS_NOT_DUNHUANG_SEMANTIC_ACCURACY',initial_same=True,schedule_same=True,models={a:sha(ROOT/a/'model.pt') for a in models},test_tensor_reads=0,can_route=False));print('R54_GATE',gate,same_ok,flush=True)
