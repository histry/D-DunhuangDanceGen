"""Attribute detected target runs to actual source support, without semantic labels."""
from pathlib import Path
import numpy as np
from .study import ROOT,load_study
from dunhuang.research.local_timing_r68.common import read,write,sha

def run():
 natural=read(ROOT/'audit/natural_sources.json');rows=read(ROOT/'audit/records.json');packs={k:load_study(k) for k in ['R69','R73','GTW']};result=[]
 for r in rows:
  if r['status']!='OBSERVED':continue
  q=packs[r['study']][3][r['unit']+'__'+r['method']+'__q'];mask=np.asarray(natural[r['source_uid']]['regions']['body']['low_cells']);n=len(mask)
  runs=[]
  for v in r['core']['regions']['body']['low_runs']:
   a,b=v['cells_half_open'];l,rr=float(q[a]),float(q[b]);cells=np.arange(max(0,int(np.floor(l))),min(n,int(np.ceil(rr))))
   widths=np.maximum(0,np.minimum(cells+1,rr)-np.maximum(cells,l));fraction=float(np.sum(widths*mask[cells])/max(rr-l,1e-12))
   runs.append(dict(target_cells=[a,b],target_seconds=(b-a)/30,source_frame_centers=[l,rr],source_natural_seconds=(rr-l)/30,source_low_support_fraction=fraction,
    all_source_support_low=bool(abs(fraction-1)<1e-9),added_seconds=((b-a)-(rr-l))/30,source_to_target_mean_rate=(b-a)/(rr-l),angular_excursion_max_rad=v['angular_excursion_max_rad'],FK_excursion_rms_max_m=v['FK_excursion_rms_max_m'],semantic_quality='UNKNOWN'))
  result.append(dict(study=r['study'],unit=r['unit'],case_key=r['case_key'],method=r['method'],source_uid=r['source_uid'],runs=runs,whole_core_source_low_support=any(h['source_supported_frame_centers']==[float(r['source_interval'][0]),float(r['source_interval'][1]-1)] for h in r['mapped_source_holds'])))
 for p in packs.values():p[3].close()
 out=ROOT/'attribution';out.mkdir(exist_ok=False);write(out/'records.json',result);write(out/'manifest.json',dict(audit_manifest_sha256=sha(ROOT/'audit/manifest.json'),code_sha256=sha(Path(__file__)),records_sha256=sha(out/'records.json'),scope='SOURCE_SUPPORT_AND_TIME_INCREASE_NOT_ARTISTIC_WRONGNESS'))
 print('HOLD_ATTRIBUTION_COMPLETE',len(result),flush=True)

if __name__=='__main__':run()
