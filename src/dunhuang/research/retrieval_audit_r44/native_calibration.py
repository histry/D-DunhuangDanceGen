"""Apply the frozen Train-only scale to Native tokens, without target labels or routing."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from .path import partial

def run(calibration,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh Native calibration')
    out.mkdir(parents=True);cal=read(calibration);parent=Path('outputs/sdmr_event_rank_01/change_shadow_v3');meta=read(parent/'result.json');z=Arrays(parent/'arrays.npz');cfg=read('src/dunhuang/research/sdmr_event_rank/config.json');arrays={};records=[]
    for r in meta['records']:
        key=r['key'];q={n:z[key+'_c0_'+n] for n in ('mt','dt','mv','dv','positions')};values=[]
        for j in range(len(r['candidates'])):
            c={n:z[key+f'_c{j}_'+n] for n in ('mt','dt','mv','dv','positions')};values.append(partial(q,c,cfg,cal['drop_cost']))
        values=np.array(values);ids=np.flatnonzero(np.isclose(values[:,0],values[:,0].max(),atol=1e-7,rtol=0));arrays[key+'_calibrated']=values;records.append(dict(key=key,mean_candidate_coverage=float(values[:,1].mean()),mean_tied_top1_coverage=float(values[ids,1].mean()),fractional_top1_candidates=ids.tolist(),semantic_accuracy=None,calibration_informative_on_train_probe=cal['informative']))
    np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,calibration_sha256=sha(calibration),drop_cost=cal['drop_cost'],train_probe_informative=cal['informative'],native_semantic_truth='UNKNOWN',scope='FIXED_TRAIN_SCALE_TRANSFER_DIAGNOSTIC_NOT_TARGET_DOMAIN_NULL_QUALIFICATION',optimizer_steps=0,can_route=False,operational_qualified=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--calibration',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.calibration,a.out)
