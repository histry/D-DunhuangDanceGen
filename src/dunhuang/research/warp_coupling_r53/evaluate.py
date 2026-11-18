"""Development-group summaries; edges are never treated as independent observations."""
from .common import *

def summarize(root):
 data=read(root/'result.json');cache=read(root/'cache.json')['records'];rows=[]
 for r in data['records']:
  base=r['plans']['B0'];out=dict(key=r['key'],song=r['song'],comparisons={})
  for arm,s in r['plans'].items():
   if arm=='B0':continue
   if s['status']!='NUMERICAL_SHORT_PLAN_NOT_NATIVE' or base['status']!='NUMERICAL_SHORT_PLAN_NOT_NATIVE':out['comparisons'][arm]=dict(status='NOT_BOTH_ADMITTED');continue
   a,b=base['metrics'],s['metrics'];out['comparisons'][arm]=dict(boundary_delta=b['independent_boundary_D3_mean']-a['independent_boundary_D3_mean'],core_ratio=b['core_D3_Q95_mean']/max(a['core_D3_Q95_mean'],1e-12),beat_delta=None if a['beat_proxy'] is None or b['beat_proxy'] is None else b['beat_proxy']-a['beat_proxy'],different_path=s['nodes']!=base['nodes'],source_coverage=len({data['nodes'][k]['source_uid'] for k in s['nodes']}),exact_event_repeats=len(s['nodes'])-len({data['nodes'][k]['event_uid'] for k in s['nodes']}))
  rows.append(out)
 rng=np.random.default_rng(CFG['seed']);music_groups=[];passes=[]
 for song in range(3):
  rr=[r for r in data['records'] if r['song']==song];comparisons={}
  for other in ('B0','B1','B2'):
   good=[r for r in rr if all(r['plans'].get(k,{}).get('status')=='NUMERICAL_SHORT_PLAN_NOT_NATIVE' for k in (other,'B3'))]
   if not good:comparisons[other]=dict(cases=0,qualified=False);continue
   a=np.mean([r['plans'][other]['metrics']['independent_boundary_D3_mean'] for r in good]);b=np.mean([r['plans']['B3']['metrics']['independent_boundary_D3_mean'] for r in good]);core=np.mean([r['plans']['B3']['metrics']['core_D3_Q95_mean']/max(r['plans'][other]['metrics']['core_D3_Q95_mean'],1e-12) for r in good]);beats=[r['plans']['B3']['metrics']['beat_proxy']-r['plans'][other]['metrics']['beat_proxy'] for r in good if r['plans']['B3']['metrics']['beat_proxy'] is not None and r['plans'][other]['metrics']['beat_proxy'] is not None];beat=float(np.mean(beats)) if beats else None
   comparisons[other]=dict(cases=len(good),boundary_ratio=float(b/max(a,1e-12)),core_ratio=float(core),beat_delta=beat,qualified=bool(b<=a*(1-CFG['independent_boundary_improvement']) and core<=1+CFG['maximum_core_D3_degradation'] and beat is not None and beat>=-CFG['maximum_beat_proxy_degradation']))
  ok=all(v['qualified'] for v in comparisons.values());passes.append(ok);music_groups.append(dict(song=song,comparisons=comparisons,registered_proxy_pass=ok))
 bootstrap={}
 for other in ('B0','B1','B2'):
  values=[g['comparisons'][other].get('boundary_ratio') for g in music_groups]
  if all(v is not None for v in values):
   samples=np.asarray(values)[rng.integers(0,3,(5000,3))].mean(1);bootstrap[other]=dict(music_mean_ratio=float(np.mean(values)),exploratory_interval=np.quantile(samples,[.025,.975]).tolist(),independent_music_groups=3,scope='EXPOSED_THREE_GROUP_DESCRIPTIVE_BOOTSTRAP_NOT_GENERALIZATION_CI')
 sensitivity=[]
 for uid in sorted({n['source_uid'] for n in data['nodes'].values()}):
  selected=[r for r in data['records'] if all(r['plans'].get(k,{}).get('status')=='NUMERICAL_SHORT_PLAN_NOT_NATIVE' for k in ('B0','B3')) and all(data['nodes'][n]['source_uid']!=uid for k in ('B0','B3') for n in r['plans'][k]['nodes'])]
  sensitivity.append(dict(excluded_source=uid,cases=len(selected),boundary_delta_mean=float(np.mean([r['plans']['B3']['metrics']['independent_boundary_D3_mean']-r['plans']['B0']['metrics']['independent_boundary_D3_mean'] for r in selected])) if selected else None))
 flips=[c for p in data['full_pair_mechanisms'] for c in p['cells'] if c['admission_flip']];recovered_edges=sum(not p['cells'][0]['admitted'] and any(c['admitted'] for c in p['cells']) for p in data['full_pair_mechanisms']);lost_edges=sum(p['cells'][0]['admitted'] and any(not c['admitted'] for c in p['cells']) for p in data['full_pair_mechanisms']);recovered=sum(r['plans']['B0']['status']!='NUMERICAL_SHORT_PLAN_NOT_NATIVE' and r['plans'].get('B3',{}).get('status')=='NUMERICAL_SHORT_PLAN_NOT_NATIVE' for r in data['records'])
 return dict(cases=rows,music_groups=music_groups,exploratory_bootstrap=bootstrap,leave_source_out=sensitivity,admission_flip_cells=len(flips),fixed_event_pair_count=len(data['full_pair_mechanisms']),fixed_event_pairs_with_recovered_edges=recovered_edges,fixed_event_pairs_with_lost_edges=lost_edges,registered_tasks=len(rows),baseline_failed_tasks=sum(r['plans']['B0']['status']!='NUMERICAL_SHORT_PLAN_NOT_NATIVE' for r in data['records']),recovered_tasks=recovered,recovery_rate_all_registered=recovered/len(rows),G3=bool(sum(passes)>=CFG['G3_required_music_groups']),G3_scope=CFG['G3_scope'],G4=False,G4_reason=CFG['G4'],can_route=False)
