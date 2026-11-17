"""Actual-frame boundary, content and world-geometry diagnostics; no contact truth."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.transition_r37.evaluate import decode,metrics,curves,world_fk

def root_profile(root,boundaries):
    root=np.asarray(root,float);out={};templates=[]
    for order,label in ((1,'velocity'),(2,'acceleration'),(3,'jerk')):
        value=np.linalg.norm(np.diff(root,n=order,axis=0)*30**order,axis=-1);i=int(np.argmax(value));out[label+'_peak']=float(value[i]);out[label+'_peak_template']=[i,i+order];out[label+'_peak_frame']=i+order
        out[label+'_curve']=[0.]*order+value.tolist()
        for boundary in boundaries:
            for start in range(max(0,boundary-order),min(len(value),boundary)):
                templates.append(dict(boundary=boundary,order=order,interval=[start,start+order],value=float(value[start])))
    out['cross_boundary_templates']=templates;return out

def frequency_error(points,reference,gap):
    a,b=gap;e=(points-points[:,:1])-(reference-reference[:,:1]);e=e[a:b];e-=e.mean(0,keepdims=True);f=np.fft.rfftfreq(b-a,1/30);p=(abs(np.fft.rfft(e*np.hanning(b-a)[:,None,None],axis=0))**2).sum((1,2))
    high=p[f>6].sum();total=p.sum()
    return dict(error_position_spectral_energy=float(total),above6hz_error_energy=float(high),above6hz_error_fraction=float(high/max(total,1e-30)),scope='HANN_ROOT_RELATIVE_ERROR_GAP_ONLY_NOT_SEMANTIC_EVENT_ERROR')

def peaks(points,target,gap):
    a,b=gap;v=np.linalg.norm(np.diff(points,axis=0)*30,axis=-1).mean(-1);t=np.linalg.norm(np.diff(target,axis=0)*30,axis=-1).mean(-1)
    def find(x):return find_peaks(x[a:b-1],distance=3,prominence=max(.01,.1*np.ptp(x[a:b-1])))[0]+a+1
    actual=find(v);expected=find(t);pairs=[]
    if len(actual) and len(expected):
        costs=abs(expected[:,None]-actual[None]);rows,cols=linear_sum_assignment(costs)
        pairs=[dict(reference_frame=int(expected[i]),generated_frame=int(actual[j]),absolute_difference_frames=int(costs[i,j])) for i,j in zip(rows,cols) if costs[i,j]<=12]
    return dict(reference_speed_peaks=expected.tolist(),generated_speed_peaks=actual.tolist(),matched_within12frames=pairs,proxy_not_annotated_phase=True)

def run(root,out):
    root=Path(root);a=np.load(root/'generation/arrays.npz');meta=read(root/'generation/result.json');J=a['skeleton_J22'];parents=a['parents'];records=[]
    for row in meta['records']:
        key=row['key'];ref=a[key+'_reference_R'];gap=row['gap'];boundaries=sorted(set(row['cuts']+gap+[gap[0]+1,gap[1]-1]));reference_points=relative_fk(ref,J,parents);methods={};diag=[]
        for mode in ('Geometry','Raw','RootContinuous','ResidualRTS'):
            reports=[]
            for i,raw in enumerate(a[key+'_'+mode+'_139']):
                R,Root=decode(raw);P=relative_fk(R,J,parents);world=world_fk(R,Root,J,parents);saved=a[key+'_'+mode+'_22'][i]
                if abs(world-saved).max()>2e-6:raise ValueError('Saved render positions differ from independent CPU FK')
                result=metrics(R,ref,Root,J,parents,boundaries,gap,128);lo,hi=gap;profile=root_profile(Root,boundaries);e=P-reference_points
                result.update(gap_fk_rmse_m=float(np.sqrt(np.mean(e[lo:hi]**2))),root=profile,frequency_error=frequency_error(P,reference_points,gap),speed_peaks=peaks(P,reference_points,gap),max_source_error_frame=int(np.sqrt(np.mean(e**2,axis=(1,2))).argmax()),reference_is_continuous_truth=row['reference_is_target'],foot_world_height_range_m=[float(world[:,[7,8,10,11],1].min()),float(world[:,[7,8,10,11],1].max())],ground_height='UNKNOWN',body_gap_amplitude_rms_m=float(np.sqrt(np.mean(np.ptp(P[lo:hi],axis=0)**2))))
                if row['reference_is_target']:
                    truth=a[key+'_target_139'][i];root_error=Root-truth[:,4:7]
                    result['root_truth_errors']={f'D{k}':float(np.sqrt(np.mean((np.diff(root_error[lo-3:hi+3],n=k,axis=0)*30**k)**2))) for k in (0,1,2,3)}
                reports.append(result)
            methods[mode]=reports
        for i in range(2):
            d={}
            for mode in methods:
                raw=a[key+'_'+mode+'_139'][i];R,Root=decode(raw);v=curves(R,ref,Root,J,parents);v['root_acceleration']=methods[mode][i]['root']['acceleration_curve'];v['root_jerk']=methods[mode][i]['root']['jerk_curve'];d[mode]=v
            diag.append(d)
        projection={}
        for mode in ('Raw','RootContinuous'):
            trace=a[key+'_'+mode+'_root_projection_trace'];projection[mode]=dict(last_step=[dict(before=root_profile(trace[-1,i,0],boundaries),after=root_profile(trace[-1,i,1],boundaries)) for i in range(2)],same_actual_xt_per_pre_post_pair=True)
        if not np.array_equal(a[key+'_Raw_root_projection_trace'][0,:,0],a[key+'_RootContinuous_root_projection_trace'][0,:,0]):raise ValueError('First x0 proposal differs before intervention')
        records.append(dict(key=key,kind=row['kind'],methods=methods,diagnostics_variants=diag,projection=projection,all_boundaries=boundaries,reference_is_target=row['reference_is_target'],raw_r38_bitwise_replay=row['raw_r38_bitwise_replay'],smoothing=row['smoothing']))
    result=dict(records=records,arrays_sha256=sha(root/'generation/arrays.npz'),optimizer_steps=0,whole_song_generations=0,operational_qualified=False,full_native='NOT_EVALUABLE_22_OF24',foot_contact_truth='UNKNOWN',parameter_selection='TRAIN_ONLY_NO_VAL_GRID')
    write(out,result);print('R39_METRICS',len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
