"""Optional externally reviewed quality annotations and Shadow-only candidates."""
import json
import numpy as np
from dunhuang.io import checked
from .calibration import partition,grouped_quantile


def load_quality(reference,by_uid):
    if reference is None:return {}
    d=json.loads(checked(reference).read_text())
    if (d.get('schema')!='independent_native_quality_v1' or d.get('validated') is not True or
        d.get('source')!='manual_motion_review' or d.get('independent_of_native_metrics') is not True or
        not d.get('reviewer') or len(d.get('review_protocol_sha256',''))!=64):
        raise ValueError('Independent reviewed quality protocol required')
    result={}
    for row in d['records']:
        uid=row['window_uid'];target=by_uid[uid]
        if uid in result or row['motion_sha256']!=target['motion']['sha256'] or target['role']!='train':
            raise ValueError('Quality annotation identity/role mismatch')
        if row['quality'] not in ('high_quality','known_error','unknown'):
            raise ValueError('Unknown independent quality label')
        result[uid]=row
    return result


def candidates(records,registry,quality,iterations=300,seed=20261116):
    fitting,held,split=partition(records);entries=[]
    for spec in registry['metrics']:
        key=spec['metric']
        def included(rows,label):
            return [r for r in rows if quality.get(r['window_uid'],{}).get('quality')==label
                    and r.get('candidate_metrics',{}).get(key) is not None]
        good=included(fitting,'high_quality')
        test_good=included(held,'high_quality');test_bad=included(held,'known_error')
        value=None;stat=None
        if good and spec['category']=='B_plausibility_proxy':
            renamed=[dict(r,metrics=r['candidate_metrics']) for r in good]
            stat=grouped_quantile(renamed,key,q=.95 if spec['direction']=='upper' else .05,
                                  group_key='source_group',iterations=iterations,seed=seed)
            value=stat['value']
        def errors(rows,reject):
            if value is None or not rows:return None
            decisions=[r['candidate_metrics'][key]>value if spec['direction']=='upper'
                       else r['candidate_metrics'][key]<value for r in rows]
            return float(np.mean(decisions if reject else np.logical_not(decisions)))
        blockers=[]
        if value is None:blockers.append('NO_INDEPENDENT_GOOD_CALIBRATION_POPULATION')
        if not test_good:blockers.append('NO_INDEPENDENT_GOOD_HOLDOUT')
        if not test_bad:blockers.append('NO_INDEPENDENT_ERROR_HOLDOUT')
        if len({r['source_group'] for r in good})<5:blockers.append('FEWER_THAN_FIVE_CALIBRATION_COMPONENTS')
        entries.append(dict(metric=key,candidate_threshold=value,statistics=stat,
            independent_good_calibration=len(good),independent_good_holdout=len(test_good),
            independent_error_holdout=len(test_bad),false_reject_rate=errors(test_good,True),
            missed_error_rate=errors(test_bad,False),blockers=blockers,
            mode='SHADOW',operational_qualified=False,can_route=False,
            scope='annotation-conditioned proposal, not universal physical law or automatic release'))
    return dict(schema='externally_labelled_native_candidates_v1',records=entries,split=split,
                independent_quality_annotations=len(quality),mode='SHADOW',operational_qualified=False,can_route=False)
