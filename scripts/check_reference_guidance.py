
import json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.io import sha,checked,write
from dunhuang.research.source_tasks.common import read
root=Path(sys.argv[1]);control=Path(sys.argv[2]) if len(sys.argv)>2 else root;c=read(control/'control/result.json');e=read(root/'events/result.json')
assert len(c['records'])==240 and len(c['derivatives'])==96
assert len({(r['scope'],r['case'],r['timestep'],r['source'],r['arm']) for r in c['records']})==240
assert c['native_evaluations']==44 and c['full_dances_generated']==0
arrays=dict(np.load(control/'control/arrays.npz'));norm_error=[];random_invariance=0
for scope in ('train','val'):
    for case in range(4):
        for tick in (999,499,19):
            rs=[r for r in c['records'] if r['scope']==scope and r['case']==case and r['timestep']==tick]
            ns=[r['input_norm'] for r in rs];norm_error.append((max(ns)-min(ns))/max(ns))
            label=scope+'__'+str(case)+'__'+str(tick)
            assert np.array_equal(arrays[label+'__0__B3__predicted_x0'],arrays[label+'__1__B3__predicted_x0'])
            random_invariance+=1
assert max(norm_error)<1e-6
cf=read('outputs/source_tasks_02/cf/protocol.json');qual=read(checked(cf['qualified']));pre=read(checked(qual['previous_preflight']));manifest=read(checked(pre['manifest']));index={r['window_uid']:r for r in manifest['rows']}
known=np.ones((150,8),bool);known[25:40]=False;known[90:115]=False
def err(a,b):
    ar=matrices(a[:,79:127].reshape(150,8,6)).astype(np.float64);br=matrices(b[:,79:127].reshape(150,8,6)).astype(np.float64)
    return float(Rotation.from_matrix((ar.swapaxes(-1,-2)@br)[known]).magnitude().mean())
errors=[]
for case,pair in enumerate(manifest['validation_pairs']):
    refs=[np.load(checked(index[pair[k]]['motion'])) for k in ('a','b')]
    base=np.load(control/'control'/('val__%s__19__base'%case)/'raw_native.npy')
    for r in c['records']:
        if r['scope']!='val' or r['case']!=case or r['timestep']!=19:continue
        folder=control/'control'/('val__%s__19__%s__%s'%(case,r['source'],r['arm']))
        out=np.load(folder/'raw_native.npy');s=r['source'];own=err(base,refs[s])-err(out,refs[s]);wrong=err(base,refs[1-s])-err(out,refs[1-s])
        errors.extend([abs(own-r['own_gain_rad']),abs((own-wrong)-r['directional_gain_rad'])])
assert max(errors)<1e-6
# Independently recompute macro recording -> music AP/F1 from saved labels/scores.
data=read('outputs/source_tasks_02/content_data/protocol.json');rows=read(checked(data['manifest']))['rows']
scores=dict(np.load(root/'events/scores.npz'));event_diffs=[]
def metric(y,p):
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order]
    if not y.any() or y.all():return None
    end=np.r_[np.where(p[1:]!=p[:-1])[0],len(y)-1];tp=y.cumsum()[end];ap=float(((np.diff(np.r_[0,tp])/y.sum())*(tp/(end+1))).sum())
    pred=p>=.5;f1=float(2*(y&pred).sum()/max(int(y.sum()+pred.sum()),1))
    return np.array([ap,f1])
for branch,kinds in (('music',['onset','beat']),('motion',['upper_motion_onset','upper_speed_peak','upper_direction_turn','upper_pose_hold'])):
    indices=scores[branch+'__indices'];role=np.array([rows[int(i)]['role'] for i,_,_ in indices]);val=role=='val';group=np.array([rows[int(i)]['source_group'] for i,_,_ in indices])[val];record=np.array([rows[int(i)]['recording_uid'] for i,_,_ in indices])[val]
    for column,kind in enumerate(kinds):
        tasks=e['probes'][branch][kind]
        if tasks.get('status')=='NOT_EVALUABLE':continue
        y=scores[branch+'__labels'][val,column]
        calculated={}
        for name in tasks:
            p=scores[branch+'__'+kind+'__'+name][val];groups={}
            for g in sorted(set(group)):
                if branch=='music':
                    v=metric(y[group==g],p[group==g])
                else:
                    items=[metric(y[record==r],p[record==r]) for r in sorted(set(record[group==g]))];items=[v for v in items if v is not None];v=np.mean(items,0) if items else None
                if v is not None:groups[g]=v
            calculated[name]=groups
        for name,g in calculated.items():
            for key,baseline in (('vs_simple','simple_time_signal'),('vs_raw','raw_observations')):
                old=tasks[name][key]
                if old['status']=='NOT_EVALUABLE':continue
                ids=sorted(set(g)&set(calculated[baseline]));a=np.mean([g[i] for i in ids],0);b=np.mean([calculated[baseline][i] for i in ids],0)
                event_diffs.extend(abs(a-np.array([old['AP'],old['F1']])).tolist());event_diffs.extend(abs((a-b)-np.array([old['delta_AP'],old['delta_F1']])).tolist())
assert max(event_diffs,default=0)<1e-12
pub=read('reports/source_tasks_summary.json');weights={}
for k,v in pub['model_sha256'].items():
    file=Path('outputs/source_tasks_02')/k/('adapter.pt' if k.startswith('cf/') else 'content.pt')
    assert sha(file)==v;weights[k]=v
attempts={}
for n in ('01','02','03','04'):
    receipt=Path('outputs/reference_guidance_'+n+'_receipt')
    attempts[n]=dict(execution_commit=(receipt/'execution_commit.txt').read_text().strip(),exit_status=int((receipt/'exit_status.txt').read_text()),native_arrays_present=len(list(Path('outputs/reference_guidance_'+n).rglob('native.npy'))),receipt_sha256={f.name:sha(f) for f in receipt.iterdir() if f.is_file()},log_sha256=sha('outputs/reference_guidance_'+n+'.log'))
write(root/'public_checks.json',dict(schema='independent_reference_guidance_checks',verifier_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),verifier_sha256=sha(__file__),source_direction_readouts_verified=len(errors)//2,equal_norm_relative_spread=max(norm_error),source_blind_random_invariance_cases=random_invariance,max_direction_readout_error_rad=max(errors),frame_metric_values_verified=len(event_diffs),max_frame_metric_error=max(event_diffs,default=0),frozen_model_files_rechecked=weights,derivative_records=len(c['derivatives']),trusted_derivatives=sum(r['numeric_trusted'] for r in c['derivatives']),records=240,attempts=attempts,replay_optimizer_calls=0,replay_fit_calls=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False))
print('INDEPENDENT_CHECKS_WRITTEN',len(errors)//2,len(event_diffs))
