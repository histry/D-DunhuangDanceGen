from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
ROOT=Path('outputs/validation_r51_01')
GAP=(104,152)
SEEDS=(51051,51052)

def fresh(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh R51 output required: '+str(out))
    out.mkdir(parents=True)
    return out

def regions(R,J,parents,spans):
    P=relative_fk(R,J,parents)
    def stat(a,b):
        result={'frames':b-a}
        for order in (1,2,3):
            values=np.sqrt(np.mean((np.diff(P,n=order,axis=0)*30**order)**2,axis=(1,2)))
            # Interior stencils and crossing stencils are deliberately separate.
            v=values[a:max(a,b-order)]
            result['D%d_Q95'%order]=float(np.quantile(v,.95)) if len(v) else None
            result['D%d_max'%order]=float(v.max()) if len(v) else None
        return result
    rows=[]
    for r in spans:
        a,b=r['interval'];rows.append(dict(r,metrics=stat(a,b)))
        if a:
            lo=max(0,a-4);hi=min(len(P),a+4)
            rows.append(dict(kind='left_boundary',interval=[lo,hi],metrics=stat(lo,hi)))
    return dict(regions=rows,whole=stat(0,len(P)),root_contact='UNKNOWN',native24='NOT_EVALUABLE_22_OF24')
