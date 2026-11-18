"""Recompute regional statistics from frozen rotations; never rerun selection."""
import numpy as np
from .common import read,write,fresh,digest
from dunhuang.research.validation_r51.common import regions


def run(out):
 out=fresh(out);parent='outputs/validation_r51_01/controls/plans';r=read(parent+'/result.json');z=np.load(parent+'/arrays.npz');rows=[];invalid=0
 for case in r['records']:
  for arm,record in case['selected'].items():
   if record['status']!='NUMERICAL_SHORT_PLAN':continue
   legacy=record['kinematics'];invalid+=sum(x['metrics']['D3_Q95'] is None for x in legacy['regions'])
   current=regions(z[case['key']+'_'+arm+'_R'],z['J'],z['parents'],case['regions'],origin=case['music_interval'][0])
   assert current['whole']==legacy['whole'],'Whole metric must remain unchanged'
   assert all(x['metrics']['D3_Q95'] is not None and 0<x['metrics']['frames']<=len(z[case['key']+'_'+arm+'_R']) for x in current['regions'])
   rows.append(dict(key=case['key'],arm=arm,music_origin=case['music_interval'][0],metrics=current))
 write(out/'result.json',dict(records=rows,legacy_missing_D3_records=invalid,corrected_region_records=sum(len(r['metrics']['regions']) for r in rows),frozen_arrays_sha256=digest(parent+'/arrays.npz'),frozen_choices_sha256=digest(parent+'/result.json'),selection_changed=False,version='R52_EXPLICIT_CLIP_ORIGIN',native24='NOT_EVALUABLE_22_OF24'))

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
