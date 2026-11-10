"""Build a compact publishable readout from verified local or server artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def collect(first,whole,phrase,physical):
    durations={}
    for label,folder in [('5s',first),('15s',first),('whole',whole)]:
        comparison=read(folder/(label+'_comparison.json'))
        rows={}
        for name,metrics in comparison['rows'].items():
            run=folder/(label+'_'+name)
            request=read(run/'request.json')
            generation=read(run/'generation.json')
            limits=read(run/'physical.json')['absolute']['limits']
            if generation['weights_before']!=generation['weights_after']:
                raise ValueError('Model changed')
            rows[name]={k:metrics[k] for k in ('frames','windows','motion_sha256','noise_sha256','mean_center_angle_rad',
                'mean_trajectory_angle_rad','anchor_step_m','jerk_mps3','skate_p95','reasons','native_pass',
                'native_metrics','jerk_peak_frame','jerk_peak_joint','normalized_overlap_max','raw_overlap_max',
                'final_prediction_residual')}
            rows[name]['commit']=request['code_commit']
        a,b,c=rows.values()
        if len({r['noise_sha256'] for r in rows.values()})!=1:raise ValueError('Noise replay mismatch')
        durations[label]=dict(rows=rows,directional_checks=comparison['checks'],limits=limits,
            native_numeric_increases_vs_hard={k:[v,c['native_metrics'][k]] for k,v in b['native_metrics'].items()
                if c['native_metrics'][k]>v},hard_replay_identical=b['motion_sha256']==read(first/'saved_diagnostics.json')[label+'_hard']['motion_sha256'])
    census=read(first/'upper10_census.json')
    diagnostics=read(first/'saved_diagnostics.json')
    for data in diagnostics.values():
        populations={}
        for interval in data.pop('static_intervals'):
            foot=populations.setdefault(interval['joint'],dict(intervals=0,frames=0,peak_mps=0.))
            foot['intervals']+=1
            foot['frames']+=interval['end_exclusive']-interval['start']
            foot['peak_mps']=max(foot['peak_mps'],interval['speed']['max'])
        data['static_support_per_foot']=populations
    g=phrase/'generation'
    return dict(durations=durations,old_diagnostics=diagnostics,census=census,
        phrase=dict(selection=read(phrase/'selection.json'),preflight=read(g/'normalization_preflight.json'),
            mean_angle_rad=float(np.mean([a['mean_radians'] for a in read(g/'evaluation.json')['anchor_errors']])),
            native=read(g/'status.json'),reasons=read(g/'physical.json')['absolute']['reasons']),
        independent_physical=read(physical/'comparison.json'),n_songs=1,n_seeds=1,training_steps=0)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ['first','whole','phrase','physical','destination']:ap.add_argument(key,type=Path)
    a=ap.parse_args()
    result=collect(a.first,a.whole,a.phrase,a.physical)
    a.destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
