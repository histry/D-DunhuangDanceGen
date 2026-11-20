"""Canonical scalar release; media, weights and original rotations stay private."""
from pathlib import Path
from collections import Counter
from .study import ROOT,CFG
from dunhuang.research.local_timing_r68.common import read,write,sha

def compact(m):
 return dict(regions={name:{k:v for k,v in row.items() if k not in ['cell_angular','cell_linear','low_cells']} for name,row in m['regions'].items()},sensitivity=m.get('sensitivity'))

def run():
 independent=read(ROOT/'receipts/independent.json');assert independent['status']=='PASS'
 out=Path('reports/hold_timing_release');out.mkdir(exist_ok=False);audit=read(ROOT/'audit/records.json');controls=read(ROOT/'controls/records.json');cases=read(ROOT/'audit/registration.json')['cases'];candidates=read(ROOT/'controls/candidates.json')
 def scrub(rows):
  result=[]
  for r in rows:
   r=dict(r)
   if 'core' in r:r['core']=compact(r['core'])
   if r.get('full_pair'):r['full_pair']=dict(rotation_sha256=r['full_pair']['rotation_sha256'],metrics=compact(r['full_pair']['metrics']))
   if r.get('connection'):r['connection']={k:r['connection'].get(k) for k in ['constructible','passes_original_screen','gap_frames','total_pair_frames','full_R_sha256','bridge_R_sha256','candidate_R_sha256','previous_bridge_qualification_inherited','fallback_used','actual_geometry_calls_this_request']}
   result.append(r)
  return result
 write(out/'audit_records.json',scrub(audit));write(out/'controls_records.json',scrub(controls));write(out/'candidates.json',candidates)
 references=read(ROOT/'audit/references.json')
 write(out/'references.json',[dict(r,natural=compact(r['natural'])) for r in references])
 write(out/'attribution.json',read(ROOT/'attribution/records.json'))
 for source,target in [('audit/calibration.json','calibration.json'),('audit/summary.json','audit_summary.json'),('controls/summary.json','controls_summary.json'),('controls/registration.json','controls_registration.json'),('controls/readout.json','readout.json'),('receipts/preview_browser_acceptance.json','preview_browser_acceptance.json'),('receipts/independent.json','independent.json'),('audit/manifest.json','audit_manifest.json'),('controls/manifest.json','controls_manifest.json')]:write(out/target,read(ROOT/source))
 write(out/'protocol.json',CFG)
 capacity=[]
 for c in cases:
  a,b=c['source_interval'];own=[v for v in candidates if v['case_key']==c['key']]
  whole=[v for v in own if any(h['source_supported_frame_centers']==[float(a),float(b-1)] for h in v['mapped_source_holds'])]
  if whole:capacity.append(dict(case_key=c['key'],source_uid=c['source_uid'],source_interval=[a,b],pool=len(own),whole_source_low_support=len(whole),target_durations_seconds=[min(h['target_seconds'] for v in whole for h in v['mapped_source_holds'] if h['source_supported_frame_centers']==[float(a),float(b-1)]),max(h['target_seconds'] for v in whole for h in v['mapped_source_holds'] if h['source_supported_frame_centers']==[float(a),float(b-1)])],scope='CURRENT_FROZEN_SOURCE_AND_240_FRAME_BUDGET_NOT_GLOBAL_INFEASIBILITY'))
 write(out/'capacity.json',capacity)
 by={(r['case_key'],r['method']):r for r in controls};comparisons=[]
 for method in sorted({r['method'] for r in controls}):
  own=[r for r in controls if r['method']==method and r['status']=='EXECUTED'];pairs=[(r,by[r['case_key'],'GTWLocal']) for r in own if by[r['case_key'],'GTWLocal']['status']=='EXECUTED'];differences=[a['core']['regions']['body']['longest_low_seconds']-b['core']['regions']['body']['longest_low_seconds'] for a,b in pairs]
  comparisons.append(dict(method=method,actual=len(own),G1_pass=sum(r['connection']['passes_original_screen'] is True for r in own),different_clock_from_GTWLocal=sum(a['clock_sha256']!=b['clock_sha256'] for a,b in pairs),different_motion_from_GTWLocal=sum(a['rotation_sha256']!=b['rotation_sha256'] for a,b in pairs),longest_low_decreased=sum(v<-1e-9 for v in differences),longest_low_increased=sum(v>1e-9 for v in differences),longest_low_unchanged=sum(abs(v)<=1e-9 for v in differences),quality='NOT_EVALUABLE'))
 write(out/'comparisons.json',comparisons)
 opportunity=[]
 for c in cases:
  own=[v for v in candidates if v['case_key']==c['key']]
  opportunity.append(dict(case_key=c['key'],source_uid=c['source_uid'],candidate_count=len(own),hold_cost_distinct=len({v['hold_cost'] for v in own}),music_hold_cost_distinct=len({v['music_hold_cost'] for v in own}),wrong_music_hold_cost_distinct=len({v['wrong_music_hold_cost'] for v in own}),mapped_hold_count_range=[min(len(v['mapped_source_holds']) for v in own),max(len(v['mapped_source_holds']) for v in own)]))
 write(out/'opportunity.json',opportunity)
 write(out/'qualification.json',dict(status='COMPLETED_RESEARCH_DIAGNOSTIC',independent_human_records=0,hold_semantics='UNKNOWN',hold_cap='TRAIN_SOURCE_STATISTIC_NOT_ACCEPTABILITY_TRUTH',music_tolerance_validity='NOT_EVALUABLE',preference_gain='NOT_EVALUABLE',can_route=False,R74='NOT_STARTED',training_steps=0,optimizer_calls=0))
 for p in out.iterdir():
  if p.is_file():p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},raw_motion_copied=False,media_copied=False))
 print('HOLD_PUBLIC_RELEASE_COMPLETE',len(audit),len(controls),flush=True)

if __name__=='__main__':run()
