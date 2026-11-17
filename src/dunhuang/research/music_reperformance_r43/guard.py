"""Observable whole-window Geometry fallback. Never relabel a failed Core candidate as passed."""
import argparse,copy,shutil
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha


def decision(candidate,geometry):
    activity=candidate['core_activity']/max(geometry['core_activity'],1e-8)
    if geometry['core_activity']<1e-8 and candidate['core_activity']<1e-8:activity=1.
    passed=.85<=activity<=1.15 and candidate['core_D3']<=1.05*geometry['core_D3'] and candidate['gap_D3']<=1.05*geometry['gap_D3'] and candidate['protected_exact']
    return dict(use='Core' if passed else 'Geometry',activity_ratio=activity,core_D3_ratio=candidate['core_D3']/max(geometry['core_D3'],1e-8),gap_D3_ratio=candidate['gap_D3']/max(geometry['gap_D3'],1e-8),candidate_passed_observable_risk_gate=bool(passed),quality_qualified=False)


def run(main,qualification,out):
    from dunhuang.research.source_context_r42.diagnose import measure
    main=Path(main);out=Path(out)
    if out.exists():raise ValueError('Fresh guarded cached output')
    out.mkdir(parents=True);meta=copy.deepcopy(read(main/'result.json'));q={r['key']:r for r in read(qualification)['records']};z=np.load(main/'arrays.npz');arrays={k:z[k] for k in z.files};records=[]
    for row in meta['records']:
        key=row['key'];choices=[decision(a,b) for a,b in zip(q[key]['methods']['Core'],q[key]['methods']['Geometry'])]
        arrays[key+'_Guarded_139']=np.stack([z[key+'_'+d['use']+'_139'][j] for j,d in enumerate(choices)]);arrays[key+'_Guarded_22']=np.stack([z[key+'_'+d['use']+'_22'][j] for j,d in enumerate(choices)])
        row['methods'].append('Guarded');values,curve=measure(arrays[key+'_Guarded_139'],z[key+'_reference_R'],z['skeleton_J22'],z['parents'],z[key+'_Geometry_139']);row['metrics']['values']['Guarded']=values;row['metrics']['curves']['Guarded']=curve;row['guard_decisions']=choices;row['scope']+=' Guarded为按可观测风险整窗回退Geometry；不是新采样，不改变Core失败结论。'
        if row['audio'] and not (out/row['audio']).exists():shutil.copyfile(main/row['audio'],out/row['audio'])
        records.extend(dict(key=key,seed=row['seeds'][j],**d) for j,d in enumerate(choices))
    np.savez_compressed(out/'arrays.npz',**arrays);meta.update(neural_sampling_calls=0,source_main_arrays_sha256=sha(main/'arrays.npz'),guard_scope='POST_PILOT_ENGINEERING_REJECTION_USING_REGISTERED_ACTIVITY_D3_THRESHOLDS_NOT_A_PRE_REGISTERED_MODEL_COMPARISON',operational_qualified=False);write(out/'result.json',meta);write(out/'guard.json',dict(records=records,accepted=sum(r['use']=='Core' for r in records),fallback=sum(r['use']=='Geometry' for r in records),thresholds=dict(activity=[.85,1.15],core_D3_ratio_max=1.05,gap_D3_ratio_max=1.05),selection_uses_paired_target_errors=False,post_selection_display_errors_use_reference=True,whole_window_not_framewise_selection=True,neural_sampling_calls=0,operational_qualified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--main',required=True);p.add_argument('--qualification',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.main,a.qualification,a.out)
