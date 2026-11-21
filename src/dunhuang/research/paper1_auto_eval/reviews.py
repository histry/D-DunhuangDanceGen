"""Numeric identity controls never become extra preference-effect observations."""
from collections import Counter
from copy import deepcopy
import numpy as np
from .common import CFG
from dunhuang.research.gtw_kinematic_baseline import reviews as legacy

def classification(left,right):
    if left is None or right is None:return 'UNAVAILABLE'
    a,b=np.asarray(left),np.asarray(right)
    return 'VALID_IDENTICAL' if a.shape==b.shape and np.allclose(a,b,atol=CFG['classification_rotation_absolute_tolerance'],rtol=0) else 'VALID_DISTINCT'

def summarize(documents,admin,roster=None):
    for d in documents:legacy.validate(d,admin)
    keep={r['id'] for r in admin['items'] if r.get('flow_control') or r.get('classification')=='VALID_DISTINCT'}
    main=deepcopy(admin);main['items']=[r for r in main['items'] if r['id'] in keep]
    docs=deepcopy(documents)
    for d in docs:d['items']=[r for r in d['items'] if r['id'] in keep]
    result=legacy.summarize(docs,main,roster)
    controls={r['id'] for r in admin['items'] if r.get('classification')=='VALID_IDENTICAL'}
    result['identical_control_choices']=dict(Counter(r['choice'] for d in documents for r in d['items'] if r['id'] in controls))
    result['registered_classes']=dict(Counter(r['classification'] for r in admin['items'] if not r.get('flow_control')))
    result['method_effect_population']='VALID_DISTINCT_ONLY; IDENTICAL_EXCLUDED; TWO_OR_MORE_ADMIN_VERIFIED_PEOPLE_PER_ITEM'
    return result

if __name__=='__main__':
    import argparse
    from .common import read,write
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--admin',required=True);p.add_argument('--roster');p.add_argument('--reviews',nargs='*',default=[]);p.add_argument('--out',required=True);a=p.parse_args()
    if Path(a.out).exists():raise ValueError('Fresh import output required')
    write(a.out,summarize([read(p) for p in a.reviews],read(a.admin),read(a.roster) if a.roster else None))
