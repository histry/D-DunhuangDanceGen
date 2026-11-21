"""Frozen model evaluation; all paired outputs grouped at music identity, not windows."""
import sys
import numpy as np
import torch
from .common import *
from .data import Store,nuisance
from .train import load
from .network import tensors

def ranking(values):
 x=np.asarray(values);best=x.max();top=np.isclose(x,best,rtol=0,atol=1e-7);return dict(strict_top1=bool(top[2] and top.sum()==1),fractional_top1=float(top[2]/top.sum()),paired_win=float(np.mean((x[2]>np.delete(x,2)+1e-7)+.5*np.isclose(x[2],np.delete(x,2),atol=1e-7,rtol=0))))
def run(out=None,names=None):
 names=names or CFG['models'];out=fresh(out or ROOT/'controls');verify(ROOT/'data');reg=read(ROOT/'data/registration.json');rows=read(ROOT/'data/windows.json');byid={r['uid']:r for r in rows};donors=read(ROOT/'data/donors.json');normal=dict(np.load(ROOT/'data/normalizer.npz'));store=Store(reg);results=[]
 # Content-independent baseline uses the same actual clips and nuisance rule, no delta labels as inputs.
 for name in names:
  net=load(name)
  for r in rows:
   donor=byid[donors[r['uid']]['donor_uid']];clips=[store.crop(r,d,normal=normal) for d in CFG['clock_offsets']]+[store.crop(r,donor=donor,normal=normal)]
   with torch.inference_mode():
    actual=net(*tensors(clips),constrained=True).cpu().numpy().astype(float).tolist();zero=net(*tensors([store.crop(r,d,normal=normal,zero=True) for d in CFG['clock_offsets']]),constrained=True).cpu().numpy().astype(float).tolist();reverse=net(*tensors([store.crop(r,d,normal=normal,reverse=True) for d in CFG['clock_offsets']]),constrained=True).cpu().numpy().astype(float).tolist()
   raw=store.crop(r);Dproxy=nuisance(raw[1][:,:272])[:2];density=[-float(np.linalg.norm((nuisance(store.crop(r,d)[0])[:2]-Dproxy)/[60,2])) for d in CFG['clock_offsets']]
   results.append(dict(uid=r['uid'],source_uid=r['source_uid'],split=r['split'],dataset=r['dataset'],music_group=r['group'],model=name,scores=actual,clock=ranking(actual[:5]),wrong_music_win=bool(actual[2]>actual[5]+1e-7),zero_scores=zero,zero_clock=ranking(zero),reverse_scores=reverse,reverse_clock=ranking(reverse),density_scores=density,density_clock=ranking(density),position_clock=ranking([0]*5),donor=donors[r['uid']],role_truth='UNKNOWN',can_route=False))
  print('FG_EVALUATED',name,len(results),flush=True)
 summaries=[];rng=np.random.default_rng(CFG['seed'])
 for split in ['train','dev','holdout','fine_diagnostic']:
  for name in names:
   own=[r for r in results if r['split']==split and r['model']==name];groups=sorted({r['music_group'] for r in own});means=lambda k:[float(np.mean([r[k]['fractional_top1'] for r in own if r['music_group']==g])) for g in groups]
   summaries.append(dict(split=split,model=name,windows=len(own),music_groups=len(groups),strict_top1=sum(r['clock']['strict_top1'] for r in own),group_balanced_clock_top1=float(np.mean(means('clock'))),group_balanced_zero_top1=float(np.mean(means('zero_clock'))),group_balanced_density_top1=float(np.mean(means('density_clock'))),correct_vs_wrong_music=float(np.mean([r['wrong_music_win'] for r in own])),artistic_quality='NOT_EVALUABLE'))
 comparisons=[]
 for split in ['dev','holdout','fine_diagnostic']:
  own=[r for r in results if r['split']==split and r['model']=='Feasible'];groups=sorted({r['music_group'] for r in own})
  for control in ['OrdinaryPostFilter','ShuffledPair','MusicDensity','PositionOnly','MusicZero']:
   if control in CFG['models'] and control not in names:continue
   delta=[]
   for group in groups:
    vals=[]
    for r in [q for q in own if q['music_group']==group]:
     other=next((q for q in results if q['model']==control and q['uid']==r['uid']),None);null=other['clock']['fractional_top1'] if other else r[{'MusicDensity':'density_clock','PositionOnly':'position_clock','MusicZero':'zero_clock'}[control]]['fractional_top1'];vals.append(r['clock']['fractional_top1']-null)
    delta.append(np.mean(vals))
   ci=np.quantile(rng.choice(delta,(CFG['bootstrap_draws'],len(delta)),replace=True).mean(1),[.025,.975]).tolist();comparisons.append(dict(split=split,control=control,groups=len(groups),mean_delta=float(np.mean(delta)),group_bootstrap95=ci))
 qualified=all(r['group_bootstrap95'][0]>0 for r in comparisons if r['split']=='holdout') and len([r for r in comparisons if r['split']=='holdout'])==5
 write(out/'results.json',results);write(out/'summary.json',summaries);write(out/'comparisons.json',comparisons);write(out/'qualification.json',dict(content_and_feasibility_gate=qualified,finite_pilot=True,new_unseen_music_certified=False,semantic_role_accuracy='NOT_EVALUABLE',independent_role_records=0,can_route=False,whole_song='NOT_STARTED',next_step='INDEPENDENT_CORRESPONDENCE_VALIDITY_REQUIRED'));seal(out);print('FG_CONTROLS_READY',qualified,flush=True)
if __name__=='__main__':run()
