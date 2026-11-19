"""Frozen-checkpoint readout. No optimizer or trained teacher-preserved model is created."""
import collections
import torch
from .common import *
from .data import Data
from .metrics import candidate_pool,relation,retrieval,clock_metrics,paired_ci
from dunhuang.research.paired_event_r41.model import EventContent
from dunhuang.research.paired_event_r41.train import seeded,digest

def load(name):
 path=seal_member('outputs/paired_event_r41_01','main/model.pt') if name=='Frozen' else seal_member(CFG['r55'],name+'/model.pt');seeded(CFG['seed']);net=EventContent().cpu().eval();net.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['state']);net.requires_grad_(False);assert sum(p.numel() for p in net.parameters())==522632
 return net,dict(path=str(path),sha256=sha(path),state_digest=digest(net),training_updates_this_round=0)

def content(net,inputs,zero=False):
 data=[torch.from_numpy(np.asarray(v).copy())[None] for v in inputs]
 if zero:data[0]=torch.zeros_like(data[0])
 with torch.no_grad():return [v[0].cpu().numpy() for v in net.content(*data)]

def diagonal(query,candidate,reverse=False):
 mt,_,mv,_=query;_,dt,_,dv=candidate
 if reverse:dt=dt[:,::-1];dv=dv[:,::-1]
 mt,dt,mv,dv=mt[::3],dt[:,::3] if dt.ndim==3 else dt[::3],mv[::3],dv[:,::3] if dv.ndim==2 else dv[::3]
 valid=mv[None]&dv
 if not valid.any(-1).all():raise ValueError('No genuine common content support')
 return (np.einsum('ctd,td->ct',dt,mt,optimize=False)*valid).sum(-1)/valid.sum(-1)

def clock_score(values,offsets):
 m,d,mv,dv=values;pad=max(map(abs,offsets));q=np.arange(pad,len(m)-pad);result=[]
 for delta in offsets:
  v=mv[q]&dv[q+delta];assert v.any();result.append(float(((m[q]*d[q+delta]).sum(-1)*v).sum()/v.sum()))
 return np.array(result)

def shift_inputs(inputs,delta,pad=15):
 m,d,dm,mv=inputs;n=len(m)
 if n<=4*pad+12 or abs(delta)>pad:raise ValueError('Non-circular common support missing')
 return [m[pad:n-pad],d[pad+delta:n-pad+delta],dm[pad+delta:n-pad+delta],mv[pad:n-pad]]

def clock_energy(E,A,shift,zero=False):
 n=len(E);m=E[15:n-15].copy();d=A[15+shift:n-15+shift];q=np.arange(15,len(m)-15)
 if zero:m[:]=0
 result=[]
 for o in CFG['offsets']:
  x=m[q]-m[q].mean();y=d[q+o]-d[q+o].mean();den=np.linalg.norm(x)*np.linalg.norm(y);result.append(float(x@y/den) if den>1e-12 else 0.)
 return np.array(result)

def combine(raw,heuristic):
 out=dict(Heuristic=heuristic,Frozen=raw['Frozen'])
 for name in ('Conditional','NoClock'):
  if name in raw:out[name]=heuristic+CFG['residual_weight']*(raw[name]-raw['Frozen']);out[name+'_TeacherRescore']=raw['Frozen']+CFG['residual_weight']*(raw[name]-raw['Frozen'])
 return out

def donor(rows,q):
 a=rows[q];options=[i for i,r in enumerate(rows) if r['dataset']==a['dataset'] and r['role']==a['role'] and r['audio_sha256']!=a['audio_sha256']]
 return min(options,key=lambda i:(abs(rows[i]['music_activity']-a['music_activity']),rows[i]['uid'])) if options else None

def balanced(records,field='metrics'):
 buckets=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in records:
  if r.get(field) is not None:buckets[r['group']][r['recording']].append(r[field])
 return {g:{k:float(np.mean([np.mean([v[k] for v in values]) for values in sources.values()])) for k in next(iter(sources.values()))[0]} for g,sources in buckets.items()}

def summaries(records,clocks):
 result={};by={}
 for key in sorted({(r['dataset'],r['role'],r['task'],r['control'],r['method']) for r in records}):
  rr=[r for r in records if (r['dataset'],r['role'],r['task'],r['control'],r['method'])==key];valid=[dict(r,metrics={k:r['metrics'][k] for k in ('recall_at1','recall_at5','mrr','chance_at1')}) for r in rr if r['metrics'] is not None];group=balanced(valid);name='|'.join(key);by[key]=group;result[name]=dict(queries=len(rr),evaluable_queries=len(valid),target_absent_queries=sum(r['target_count']==0 for r in rr),mean_target_coverage=float(np.mean([r['target_count']>0 for r in rr])),groups=group,mean={k:float(np.mean([v[k] for v in group.values()])) for k in next(iter(group.values()))} if group else None)
 comparisons={}
 for ds,role,task in sorted({k[:3] for k in by}):
  for method in CFG['methods']:
   a=by.get((ds,role,task,'actual',method),{})
   for ctrl,ref in [('actual','Frozen'),('actual','Heuristic'),('music_zero',method),('wrong_music',method),('motion_tokens_reverse',method)]:
    b=by.get((ds,role,task,ctrl,ref),{});common=sorted(a.keys()&b.keys());comparisons['|'.join((ds,role,task,method,'minus_'+ctrl+'_'+ref))]=paired_ci([a[g]['recall_at1']-b[g]['recall_at1'] for g in common],CFG['seed'],CFG['bootstrap_draws'])
 clocksummary={}
 for key in sorted({(r['dataset'],r['role'],r['control'],r['method']) for r in clocks}):
  rr=[r for r in clocks if (r['dataset'],r['role'],r['control'],r['method'])==key];g=balanced(rr);clocksummary['|'.join(key)]=dict(groups=g,mean={k:float(np.mean([v[k] for v in g.values()])) for k in ('accuracy','mae_frames')})
 return dict(retrieval=result,paired_R1_differences=comparisons,clock=clocksummary)

def run(out,stage):
 from dunhuang.research.coverage_r34.common import forbid_optimizers
 forbid_optimizers();out=fresh(out);data=Data(ROOT/'data');rows=data.rows;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);queries=data.meta['queries']
 if stage=='main':
  queries=[]
  for ds,role in [('AIST++','train'),('AIST++','val'),('FineDance','genre_train_diagnostic')]:
   groups=sorted({r['group'] for r in rows if r['dataset']==ds and r['role']==role})[:2]
   for group in groups:queries.append(min([i for i in data.meta['queries'] if rows[i]['group']==group and rows[i]['role']==role],key=lambda i:rows[i]['uid']))
 else:assert read(ROOT/'receipts/main_browser.json')['passed']
 model_names=['Frozen'] if stage=='main' else CFG['models'];models={};encoded={};proof={}
 for name in model_names:
  net,p=load(name);models[name]=net;proof[name]=p;all_values=[]
  for i in range(len(rows)):
   all_values.append(content(net,data.inputs(i)))
   if (i+1)%1500==0:print('R59_ENCODE',name,i+1,len(rows),flush=True)
  encoded[name]=[np.stack([r[k] for r in all_values]) for k in range(4)]
 records=[];clocks=[];arrays={};viewer=[]
 for number,q in enumerate(queries):
  r=rows[q];pool=candidate_pool(rows,q);assert pool;positions={v:i for i,v in enumerate(pool)};don=donor(rows,q);h=-np.abs(r['music_activity']-np.array([rows[j]['body_activity'] for j in pool]));controls=['actual'] if stage=='main' else CFG['controls']
  for control in controls:
   if control=='wrong_music' and don is None:continue
   raw={}
   for name,net in models.items():
    source=don if control=='wrong_music' else q;query=[v[source] for v in encoded[name]] if control!='music_zero' else content(net,data.inputs(q),True);candidates=[v[pool] for v in encoded[name]];raw[name]=diagonal(query,candidates,control=='motion_tokens_reverse')
   H=-np.abs((0. if control=='music_zero' else rows[don]['music_activity'] if control=='wrong_music' else r['music_activity'])-np.array([rows[j]['body_activity'] for j in pool]));scores=combine(raw,H)
   for name,values in scores.items():arrays[f'q{q}_{control}_{name}']=np.asarray(values,dtype='<f8')
   for task in ('open','same_music','activity_hard'):
    ids=candidate_pool(rows,q,task,CFG['hard_pool_size']);local=[positions[j] for j in ids];relations=[relation(r,rows[j],CFG['positive_iou'],CFG['negative_gap']) for j in ids];positive=np.array([v=='CLOCK_TARGET' for v in relations]);truth=int(positive.sum())
    for name,values in scores.items():
     metrics=retrieval(values[local],positive) if ids else None;top=ids[int(np.argmax(values[local]))] if ids else None;records.append(dict(query=q,group=r['group'],recording=r['raw_motion_sha256'],dataset=r['dataset'],role=r['role'],task=task,control=control,method=name,metrics=metrics,target_count=truth,pool_size=len(ids),unknown_count=relations.count('UNKNOWN'),top1=top,top1_relation=relation(r,rows[top]) if top is not None else None,donor=don if control=='wrong_music' else None,pool_sha256=identity([rows[j]['uid'] for j in ids]),positive_used_for_pool=False,ground_truth_scope='RECORDED_AUDIO_CLOCK_TARGET_NOT_UNIQUE_ARTISTIC_DANCE',same_recording_target_in_pool=r['dataset']=='FineDance'))
   if stage=='main' and control=='actual':viewer.append(dict(query=q,pool=pool,top={name:pool[int(np.argmax(values))] for name,values in scores.items()}))
  if stage=='controls':
   inputs=data.inputs(q);z=data.source(r['source_uid']);a,b=r['source_interval'];E,A=z['E'][a:b],z['A'][a:b]
   for shift in CFG['offsets']:
    raw={}
    for control in ('actual','music_zero'):
     for name,net in models.items():raw[name]=clock_score(content(net,shift_inputs(inputs,shift),control=='music_zero'),CFG['offsets'])
     H=clock_energy(E,A,shift,control=='music_zero');scores=combine(raw,H)
     for name,v in scores.items():
      arrays[f'clock{q}_{shift}_{control}_{name}']=v;clocks.append(dict(query=q,group=r['group'],recording=r['raw_motion_sha256'],dataset=r['dataset'],role=r['role'],control=control,method=name,injected_shift=shift,truth=-shift,metrics=clock_metrics(v,CFG['offsets'],-shift)))
  if (number+1)%30==0:print('R59_QUERIES',number+1,len(queries),flush=True)
 for name,net in models.items():assert digest(net)==proof[name]['state_digest']
 summary=summaries(records,clocks) if stage=='controls' else None;result=dict(binding=binding(),data_manifest_sha256=sha(ROOT/'data/manifest.json'),stage=stage,queries=queries,models=proof,records=records,clocks=clocks,summary=summary,viewer=viewer,score_resolution=1e-7,FineDance_scope='PRIMARY_GENRE_TRAIN_DIAGNOSTIC; WITHIN_RECORDING_TARGET_BANK_NOT_CROSS_RECORDING_SOURCE_GENERALIZATION',teacher_rescore='ALGEBRAIC_RESCORING_SAVED_R55_CHECKPOINTS_NOT_NEW_TEACHER_PRESERVED_TRAINING',neural_training_updates=0,optimizer_calls=0,can_route=False,independent_new_music_certified=False,human_reviewed=False)
 save_arrays(out/'scores.npz',arrays);write(out/'result.json',result)
 if stage=='main':
  from .view import viewer as make_viewer
  make_viewer(out,data,result)
 return result

def verify(out,source):
 out=fresh(out);source=Path(source);expected=read(source/'result.json');current=binding();assert current['science']==expected['binding']['science'] and current['dependencies']==expected['binding']['dependencies'];actual=run(out/'rebuild',expected['stage']);actual['binding']['commit']=expected['binding']['commit'];assert actual==expected;assert sha(source/'scores.npz')==sha(out/'rebuild/scores.npz');write(out/'result.json',dict(result_exact=True,score_bytes_exact=True,all_candidates_reencoded=True,optimizer_calls=0,source_commit=expected['binding']['commit'],replay_commit=current['commit']))
