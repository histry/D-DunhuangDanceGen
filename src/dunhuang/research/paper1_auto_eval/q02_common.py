"""Q02 binding and immutable-output helpers; leave all Q01 producers untouched."""
from pathlib import Path
import numpy as np
from .common import HERE,ROOT as Q01,read,write,sha,prepare,checked_parent
ROOT=Path('outputs/paper1_auto_eval_02');CFG=read(HERE/'q02_protocol.json')

def binding():
    names=['common.py','music.py','experiment.py','controls.py','observability.py','view.py']
    code={p.name:sha(p) for p in HERE.glob('q02*') if p.is_file()}
    code.update({n:sha(HERE/n) for n in names})
    return code

def parents():
    result={}
    for phase,key in [('controls','parent_controls_manifest'),('preview','parent_preview_manifest')]:
        path=Q01/phase/'manifest.json';assert sha(path)==CFG[key];m=read(path)
        for name,h in m['files'].items():assert sha(path.parent/name)==h,name
        result[phase]=sha(path)
    return result

def fresh(p):
    p=Path(p);p.mkdir(parents=True,exist_ok=False);return p

def seal(out):
    write(out/'manifest.json',dict(code=binding(),parents=parents(),files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}))

def verify_files(out):
    m=read(Path(out)/'manifest.json')
    for n,h in m['files'].items():assert sha(Path(out)/n)==h,n
    assert m['code']==binding(),'Q02 source changed after freeze'
    return m

def rank(values,correct=2):
    if not values or any(v is None or not np.isfinite(v) for v in values):return dict(status='NOT_EVALUABLE',strict_top1=None,tied_top1_credit=None,paired_win=None)
    a=np.asarray(values);tol=CFG['tie_tolerance'];ties=np.flatnonzero(abs(a-a.max())<=tol);v=a[correct]
    wins=[1. if v>x+tol else .5 if abs(v-x)<=tol else 0. for i,x in enumerate(a) if i!=correct]
    return dict(status='EVALUABLE',strict_top1=bool(len(ties)==1 and ties[0]==correct),tied_top1_credit=float(1/len(ties) if correct in ties else 0),tied_maximum_indices=ties.tolist(),paired_win=float(np.mean(wins)),correct_rank_lower=1+int((a>v+tol).sum()))
