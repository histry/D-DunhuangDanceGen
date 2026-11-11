"""Grouped, Train-only descriptive calibration. Unknown labels are not good GT."""
import hashlib
from collections import defaultdict
import numpy as np


def weighted_quantile(values,weights,q):
    v=np.asarray(values,float);w=np.asarray(weights,float)
    if v.ndim!=1 or w.shape!=v.shape or not len(v) or not np.isfinite(v).all() or np.any(w<=0):
        raise ValueError('Finite observations with positive weights required')
    order=np.argsort(v,kind='stable');v=v[order];w=w[order]
    return float(v[min(np.searchsorted(np.cumsum(w)/w.sum(),q,side='left'),len(v)-1)])


def grouped_quantile(records,metric,q=.95,group_key='recording_uid',iterations=300,seed=20261116):
    if not 0<q<1 or type(iterations) is not int or iterations<1:
        raise ValueError('Predeclared quantile and bootstrap count required')
    groups=defaultdict(list)
    for row in records:
        value=row['metrics'].get(metric)
        if value is not None:
            if not np.isfinite(value):raise ValueError('Nonfinite calibration metric')
            groups[row[group_key]].append(float(value))
    names=sorted(groups)
    if not names:return dict(status='NOT_EVALUABLE',value=None,interval95=None,groups=0,samples=0)
    def estimate(keys):
        values=[];weights=[]
        for key in keys:
            v=groups[key];values.extend(v);weights.extend([1/len(v)]*len(v))
        return weighted_quantile(values,weights,q)
    value=estimate(names);rng=np.random.default_rng(seed)
    samples=[estimate([names[i] for i in rng.integers(0,len(names),len(names))]) for _ in range(iterations)]
    interval=np.percentile(samples,[2.5,97.5]).tolist() if len(names)>=2 else None
    return dict(status='DESCRIPTIVE_ONLY',value=value,interval95=interval,groups=len(names),
        samples=sum(map(len,groups.values())),quantile=q,iterations=iterations,seed=seed,
        estimand='equal cluster weight, equal window weight within cluster',
        interval_method='cluster resampling percentile interval; independence/coverage not established',
        one_group_interval_undefined=len(names)<2)


def partition(records):
    if any(r['role']!='train' for r in records):raise ValueError('Native calibration uses historical Train only')
    components=sorted({r['source_group'] for r in records},
                      key=lambda k:hashlib.sha256(k.encode()).hexdigest())
    if len(components)<3:raise ValueError('At least three disjoint source components required')
    cut=max(1,min(len(components)-1,int(np.ceil(2*len(components)/3))))
    cal=set(components[:cut]);hold=set(components[cut:])
    fitting=[r for r in records if r['source_group'] in cal]
    validation=[r for r in records if r['source_group'] in hold]
    for key in ('recording_uid','choreography_uid','music_uid'):
        if {r[key] for r in fitting}&{r[key] for r in validation}:
            raise ValueError('Associated identity crosses calibration split: '+key)
    return fitting,validation,dict(rule='hash-sorted existing connected source components; first ceil(2/3) calibration',
        calibration_components=len(cal),holdout_components=len(hold),
        shared_recording_choreography_music=False,fresh_blind_test=False)


def reference_report(records,registry,iterations=300,seed=20261116):
    fitting,held,split=partition(records)
    entries=[]
    for spec in registry['metrics']:
        metric=spec['metric'];q=.95 if spec['direction']=='upper' else .05
        row=dict(metric=metric,units=spec['units'],category=spec['category'],
                 legacy_threshold=spec['legacy_threshold'],legacy_gate_unchanged=True)
        row['recording_grouped']=grouped_quantile(fitting,metric,q=q,group_key='recording_uid',iterations=iterations,seed=seed)
        row['choreography_grouped']=grouped_quantile(fitting,metric,q=q,group_key='choreography_uid',iterations=iterations,seed=seed)
        row['connected_component_grouped']=grouped_quantile(fitting,metric,q=q,group_key='source_group',iterations=iterations,seed=seed)
        valid=[r for r in held if r['metrics'].get(metric) is not None]
        exceeds=[(r['metrics'][metric]>spec['legacy_threshold'] if spec['direction']=='upper'
                  else r['metrics'][metric]<spec['legacy_threshold']) for r in valid]
        row.update(held_reference_windows=len(valid),
            held_legacy_exceedance_fraction=float(np.mean(exceeds)) if exceeds else None,
            false_reject_rate=None,false_negative_rate=None,
            candidate_threshold=None,qualification='BLOCKED_NO_INDEPENDENT_REAL_QUALITY_AND_CONTACT_LABELS',
            descriptive_quantile_is_not_a_threshold=True)
        entries.append(row)
    return dict(schema='native_calibration_shadow_v2',status='SHADOW_UNQUALIFIED',
        historical_train_windows=len(records),calibration_windows=len(fitting),holdout_windows=len(held),
        split=split,metrics=entries,operational_qualified=False,can_route=False,
        independent_quality_labels=0,independent_contact_labels=0,optimizer_calls=0,
        human_false_rejection_rate=None,expert_agreement=None,
        reason='Reference exceedance is not false rejection; fitted motions were not independently certified high quality')


def jerk_report(joints,limit,fps=30,filter_sigma_seconds=.03333333333333333):
    from scipy.ndimage import gaussian_filter1d
    x=np.asarray(joints)
    if x.ndim!=3 or x.shape[1:]!=(24,3) or len(x)<4 or fps!=30 or not np.isfinite(x).all():
        raise ValueError('Finite SMPL24 clock required for jerk')
    def profile(v):return np.linalg.norm(np.diff(v,n=3,axis=0)*fps**3,axis=-1)
    raw=profile(x);frame=raw.max(1);bad=frame>limit
    edges=np.diff(np.r_[False,bad,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    smooth=gaussian_filter1d(x.astype(float),filter_sigma_seconds*fps,axis=0,mode='nearest')
    filtered=profile(smooth)
    return dict(status='COMPUTED',fps=fps,units='m/s^3',stencils=len(frame),
        raw=dict(p50=float(np.percentile(raw,50)),p95=float(np.percentile(raw,95)),
                 p99=float(np.percentile(raw,99)),max=float(raw.max())),
        violating_stencils=int(bad.sum()),violation_fraction=float(bad.mean()),
        violations_per_second=float(bad.sum()/((len(x)-1)/fps)),
        longest_consecutive_stencil_seconds=float(max(ends-starts,default=0)/fps),
        intervals=[[int(a),int(b+3)] for a,b in zip(starts,ends)],
        filtered_diagnostic=dict(sigma_seconds=filter_sigma_seconds,operator='Gaussian nearest boundary',
            p95=float(np.percentile(filtered,95)),max=float(filtered.max()),
            used_for_acceptance=False,motion_saved_or_modified=False),
        legacy_threshold_reference=float(limit),legacy_result_reclassified=False)
