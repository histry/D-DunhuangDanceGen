"""Read-only hold attribution, followed by a bounded common-domain mechanism pilot."""
from pathlib import Path
import sys,json
from collections import Counter
import numpy as np
from . import metrics
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.sdmr_warp_rank import connectivity
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.warp_coupling_r53.common import array_hash,save_arrays
ROOT=Path('outputs/hold_timing_01');HERE=Path(__file__).parent;CFG=read(HERE/'protocol.json')
STUDIES={'R69':('retiming_baselines_r69_01/main','population.json'),'R73':('compensation_r73_01/controls','registration.json'),'GTW':('gtw_baseline_01/controls','registration.json')}

def binding():
 deps=['gtw_kinematic_baseline/features.py','compensation_r73/warp.py','sdmr_warp_rank/encoder.py','sdmr_warp_rank/readout.py','sdmr_warp_rank/connectivity.py','complete_r36/core.py']
 return dict(code={n:sha(HERE/n) for n in ['study.py','metrics.py','protocol.json']},dependencies={n:sha(HERE.parent/n) for n in deps})

def seal(out):write(out/'manifest.json',dict(binding=binding(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}))

def load_study(name):
 folder,reg=STUDIES[name];p=Path('outputs')/folder
 assert sha(p/'manifest.json')==CFG[name+'_manifest']
 m=read(p/'manifest.json')
 for n in ['arrays.npz','records.json',reg]:assert sha(p/n)==m['files'][n]
 return p,read(p/reg)['cases'],read(p/'records.json'),np.load(p/'arrays.npz')

def calibrate(ctx):
 sources=ctx['reference']['sources'];natural={};all_signals={};thresholds={}
 for uid,R in ctx['lib'].rotations.items():all_signals[uid]=metrics.signals(R,ctx['J'],ctx['parents'])
 for quantile in CFG['sensitivity_quantiles']:
  d={}
  for region in metrics.REGIONS:
   w=[float(np.quantile(all_signals[s['source_uid']][0][region]['angular'],quantile)) for s in sources]
   v=[float(np.quantile(all_signals[s['source_uid']][0][region]['linear'],quantile)) for s in sources]
   d[region]=dict(angular=min(CFG['angular_cap_radps'],float(np.median(w))),linear=float(np.median(v)),train_source_angular_quantiles=w,train_source_FK_quantiles=v)
  thresholds[str(quantile)]=d
 primary=thresholds['0.2'];caps=[]
 for s in sources:
  uid=s['source_uid'];sig,P=all_signals[uid];R=ctx['lib'].rotations[uid]
  h=metrics.describe(R,P,sig['body'],primary['body'])['low_runs'];values=[v['duration_seconds'] for v in h]
  caps.append(dict(source_uid=uid,runs=len(h),q75_seconds=None if not values else float(np.quantile(values,.75))))
 capvalues=[v['q75_seconds'] for v in caps if v['q75_seconds'] is not None]
 result=dict(thresholds=thresholds,primary_quantile=.2,minimum_run_cells=30,training_sources=sources,training_source_rotation_hashes={s['source_uid']:array_hash(ctx['lib'].rotations[s['source_uid']]) for s in sources},cap_sources=caps,
  fixed_cap_seconds=None if not capvalues else float(np.median(capvalues)),cap_qualification='SOURCE_STATISTIC_NOT_HUMAN_TOLERANCE',world_root_contact='UNKNOWN')
 for uid,(sig,P) in all_signals.items():natural[uid]=dict(rotation_sha256=array_hash(ctx['lib'].rotations[uid]),regions={n:metrics.describe(ctx['lib'].rotations[uid],P,sig[n],primary[n]) for n in metrics.REGIONS})
 return result,natural,all_signals

def summarize_motion(R,ctx,cal):
 sig,P=metrics.signals(R,ctx['J'],ctx['parents']);primary=cal['thresholds']['0.2']
 return dict(regions={n:metrics.describe(R,P,sig[n],primary[n]) for n in metrics.REGIONS},sensitivity={q:{n:dict(low_fraction=float(metrics.low_mask(sig[n],t[n]).mean()),longest_seconds=max([b-a for a,b in metrics.spans(metrics.low_mask(sig[n],t[n]),1)],default=0)/30) for n in metrics.REGIONS} for q,t in cal['thresholds'].items()})

def run_audit(dest=None):
 out=Path(dest) if dest else ROOT/'audit';out.mkdir(parents=True,exist_ok=False);ctx=connectivity.prepare();cal,natural,_=calibrate(ctx);rows=[];cache={};verified={};cases=load_study('R69')[1]
 write(out/'registration.json',dict(protocol=CFG,binding=binding(),cases=cases,scope='HISTORICALLY_EXPOSED_CASES_NOT_NEW_TEST'));write(out/'calibration.json',cal);write(out/'natural_sources.json',natural)
 references=[]
 for c in cases:
  uid=c['source_uid'];a,b=c['source_interval'];R=ctx['lib'].rotations[uid][a:b]
  references.append(dict(case_key=c['key'],source_uid=uid,source_interval=[a,b],natural=summarize_motion(R,ctx,cal),source_holds=metrics.occupancy(np.arange(a,b),natural[uid]['regions']['body']['low_runs'],[a,b])))
 write(out/'references.json',references)
 for label in STUDIES:
  folder,cc,rr,z=load_study(label);bykey={c['key']:c for c in cc}
  for r in rr:
   c=cc[r['case_index']] if 'case_index' in r else bykey[r['case_key']];uid=c['source_uid'];unit=r.get('unit',c['key']);prefix=unit+'__'+r['method']+'__';key=prefix+'R'
   row=dict(study=label,unit=unit,case_key=c['key'],method=r['method'],source_uid=uid,source_interval=c['source_interval'],status='MISSING_EXECUTION',original_status=r.get('status'),quality='NOT_EVALUABLE')
   if key not in z:rows.append(row);continue
   R=z[key];q=z[prefix+'q'];h=array_hash(R);qh=array_hash(q);identity=(uid,qh)
   if identity not in verified:verified[identity]=array_hash(rotation_sample(ctx['lib'].rotations[uid],q))
   assert verified[identity]==h and R.shape==(240,22,3,3) and q[0]==c['source_interval'][0] and q[-1]==c['source_interval'][1]-1 and (np.diff(q)>0).all()
   if h not in cache:cache[h]=summarize_motion(R,ctx,cal)
   fullkey=prefix+'full_R';full=None
   if fullkey in z:
    F=z[fullkey];fh=array_hash(F)
    if fh not in cache:cache[fh]=summarize_motion(F,ctx,cal)
    full=dict(rotation_sha256=fh,metrics=cache[fh])
   holds=metrics.occupancy(q,natural[uid]['regions']['body']['low_runs'],c['source_interval'])
   rows.append(dict(row,status='OBSERVED',rotation_sha256=h,clock_sha256=qh,core=cache[h],mapped_source_holds=holds,source_statistic_excess=metrics.excess(holds,cal['fixed_cap_seconds']),full_pair=full,
    geometry_screen=(r.get('connection') or {}).get('passes_original_screen'),historical_geometry_only=True,semantic_hold='UNKNOWN',can_route=False))
  z.close();print('HOLD_AUDIT_STUDY',label,len(rr),flush=True)
 write(out/'records.json',rows);write(out/'summary.json',dict(positions=len(rows),actual=sum(r['status']=='OBSERVED' for r in rows),missing=sum(r['status']!='OBSERVED' for r in rows),unique_motion_profiles=len(cache),source_intervals=len({(c['source_uid'],tuple(c['source_interval'])) for c in cases}),source_recordings=len({c['source_uid'] for c in cases}),human_records=0,quality='NOT_EVALUABLE',can_route=False,new_training_steps=0));seal(out)

if __name__=='__main__':run_audit()
