"""Music-group evaluation and Train-only confidence; no ChangE semantic truth."""
from collections import defaultdict
from .common import *
from .study import Study
from dunhuang.research.sdmr_event_rank.scorer import retrieval_metrics
from dunhuang.research.event_alignment_r49.common import multiscale

def selected_clock_rows(s,groups):
 ids=[]
 for g in groups:
  options=sorted({t['query'] for t in s.meta['tasks'] if t['group']==g and t['role']==('val' if g in s.split['val'] else 'train')},key=lambda i:s.meta['rows'][i]['uid'])
  ids.extend(options[:2])
 return ids

def model(s,arm):
 net=s.model();p=ROOT/arm/'model.pt';assert sha(p)==read(ROOT/arm/'result.json')['model_sha256'];net.load_state_dict(torch.load(p,map_location=s.device,weights_only=True)['state']);return net.eval()

def clock_readout(s,net,ids,controls=('actual',)):
 arrays={};records=[]
 with torch.no_grad():
  for i in ids:
   for shift in CFG['offsets']:
    truth=CFG['offsets'].index(-shift)
    for control in controls:
     raw=s.clock(net,i,shift,control);base=s.clock_base(i,shift,control);values={control:residual_scores(raw,base,np.zeros(len(base))).cpu().numpy()}
     if control=='actual':
      values['Frozen']=base.cpu().numpy();values['ClockOnly']=-np.abs(CFG['offsets']).astype(float);E=s.z['E'+str(i)];A=s.z['A'+str(i)];pad=max(map(abs,CFG['offsets']));n=len(E)-2*pad;positions=np.arange(pad+6,n-6-pad)+pad;values['EnergyCurve']=np.array([multiscale(E[positions],A[positions+shift+delta]) for delta in CFG['offsets']])
     for label,v in values.items():
      arrays[f'{i}_{shift}_{label}']=v;best=int(np.argmax(v));p=np.arange(len(v))==truth;sorted_v=np.sort(v);records.append(dict(row=i,group=s.meta['rows'][i]['source_group'],recording=s.meta['rows'][i]['recording_uid'],control=label,shift=shift,truth=-shift,predicted=CFG['offsets'][best],accuracy=hit(v,p),absolute_error_frames=float(np.mean([abs(CFG['offsets'][j]+shift) for j in np.flatnonzero(np.isclose(v,v.max(),rtol=0,atol=1e-7))])),margin=float(sorted_v[-1]-sorted_v[-2])))
 return records,arrays

def confidence(cal,thresholds=None):
 # Synthetic injected-clock correctness only; ties are fractional, UNKNOWN is preserved.
 thresholds=CFG['confidence_thresholds'] if thresholds is None else thresholds;curve=[]
 for threshold in thresholds:
  accepted=[r for r in cal if r['margin']>=threshold];by=defaultdict(list)
  for r in accepted:by[r['group']].append(1-r['accuracy'])
  error=float(np.mean([np.mean(v) for v in by.values()])) if by else None
  curve.append(dict(threshold=threshold,accepted=len(accepted),total=len(cal),coverage=len(accepted)/max(1,len(cal)),group_error=error,music_groups=len(by)))
 valid=[r for r in curve if r['accepted']>=20 and r['music_groups']>=4 and r['group_error']<=CFG['calibration_max_error']]
 return dict(threshold=min(r['threshold'] for r in valid) if valid else None,curve=curve,scope='TRAIN_HELD_MUSIC_GROUPS_INJECTED_CLOCK_NOT_SEMANTIC_CONFIDENCE')

def mean_ci(values):
 values=np.array(values,dtype=float)
 if not len(values):return dict(mean=None,ci95=[None,None],groups=0)
 rng=np.random.default_rng(CFG['seed']);a=values[rng.integers(len(values),size=(CFG['bootstrap_draws'],len(values)))].mean(1);return dict(mean=float(values.mean()),ci95=np.quantile(a,[.025,.975]).tolist(),groups=len(values))

def clock_summary(records):
 out={}
 for control in sorted({r['control'] for r in records}):
  out[control]={}
  for metric in ('accuracy','absolute_error_frames'):
   groups=defaultdict(lambda:defaultdict(list))
   for r in records:
    if r['control']==control:groups[r['group']][r['recording']].append(r[metric])
   out[control][metric]=mean_ci([np.mean([np.mean(x) for x in rec.values()]) for rec in groups.values()])
 return out

def evaluate(out):
 out=fresh(out);s=Study();s.frozen_inference=True;models={a:model(s,a) for a in CFG['arms']};results={a:read(ROOT/a/'result.json') for a in models};assert len({r['initial_digest'] for r in results.values()})==1 and len({r['schedule_sha256'] for r in results.values()})==1
 tasks=s.meta['tasks'];scores=defaultdict(dict);arrays={}
 for i,t in enumerate(tasks):
  for arm,net in models.items():
   v=s.evaluate(net,t);scores[arm][i]=v['Conditional']
   if arm=='Conditional':
    for name in ('Heuristic','Frozen','HybridFrozen'):scores[name][i]=v[name]
    for control in ('wrong_music','music_zero','time_reverse'):
     with torch.no_grad():
      raw=s.raw(net,t,control=control);base=s.base_scores(t,control=control);scores[control][i]=residual_scores(raw,base,t['heuristic']).cpu().numpy()
  scores['EnergyCurve'][i]=np.array([multiscale(s.z['E'+str(t['query'])],s.z['A'+str(j)]) for j in t['candidates']])
  for name in scores:arrays[f'{name}_{i}']=scores[name][i]
  if (i+1)%100==0:print('R55_EVAL',i+1,len(tasks),flush=True)
 summary={}
 for role,groups in s.split.items():
  for family in ('cross_music','same_music'):
   by={};per={}
   for name in scores:
    raw=defaultdict(lambda:defaultdict(list))
    for i,t in enumerate(tasks):
     if t['group'] in groups and t['family']==family:raw[t['group']][t['recording']].append(retrieval_metrics(scores[name][i],t['positive'])['recall_at1'])
    per[name]={g:float(np.mean([np.mean(v) for v in rec.values()])) for g,rec in raw.items()};by[name]=mean_ci(list(per[name].values()))
   comparisons={name:mean_ci([per['Conditional'][g]-per[name][g] for g in per['Conditional'].keys()&per[name].keys()]) for name in per if name!='Conditional'}
   summary[role+'_'+family]=dict(methods=by,paired_differences=comparisons,group_R1=per)
 # Train-only calibration is computed and sealed before ANY Val clock readout.
 cal_ids=selected_clock_rows(s,s.split['calibration']);cal,aa=clock_readout(s,models['Conditional'],cal_ids);calibration=confidence([r for r in cal if r['control']=='actual']);write(out/'calibration.json',calibration);cal_hash=sha(out/'calibration.json');save_arrays(out/'calibration_scores.npz',aa)
 clocks={};allrec={}
 for arm,net in models.items():
  val_ids=selected_clock_rows(s,s.split['val']);rr,aa=clock_readout(s,net,val_ids,('actual','music_zero','both_zero','time_reverse'));clocks[arm]=clock_summary(rr);allrec[arm]=rr
  for key,value in aa.items():arrays[f'clock_{arm}_{key}']=value
 assert sha(out/'calibration.json')==cal_hash
 val_actual=[r for r in allrec['Conditional'] if r['control']=='actual'];curve=confidence(val_actual)['curve'];threshold=calibration['threshold'];accepted=[r for r in val_actual if threshold is not None and r['margin']>=threshold]
 gate=all(summary['val_cross_music']['paired_differences'][m]['ci95'][0] is not None and summary['val_cross_music']['paired_differences'][m]['ci95'][0]>0 for m in ('Heuristic','EnergyCurve','Frozen','Shuffled','wrong_music'))
 save_arrays(out/'scores.npz',arrays);write(out/'clock_records.json',allrec);write(out/'result.json',dict(content=summary,clock=clocks,calibration_sha256=cal_hash,val_coverage_error_curve=curve,accepted_val=len(accepted),total_val=len(val_actual),threshold_frozen_on_train=threshold,A3_increment_gate=bool(gate),A4_A5='NOT_PROMOTED_UNPAIRED_CHANGe_SEMANTIC_TRUTH_UNKNOWN',models={a:sha(ROOT/a/'model.pt') for a in models},code=binding(),can_route=False,human_reviewed=False,Test_reads=0,whole_song_generations=0));print('R55_EVALUATED',gate,flush=True)
