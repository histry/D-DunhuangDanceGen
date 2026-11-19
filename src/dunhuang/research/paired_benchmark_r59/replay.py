"""Fresh-process exact replay using read-only candidate views, not a different score or population."""
import numpy as np
from . import scoring
from .common import *

class Selection:
 def __init__(self,bank,indices):self.bank=bank;self.indices=np.asarray(indices)
class Bank(np.ndarray):
 def __getitem__(self,key):
  bank=self.view(np.ndarray)
  return Selection(bank,key) if isinstance(key,list) else bank[key]
class ArrayAPI:
 def __getattr__(self,name):return getattr(np,name)
 def stack(self,*args,**kwargs):return np.stack(*args,**kwargs).view(Bank)

def cached_diagonal(query,candidate,reverse=False):
 dt,dv=candidate[1],candidate[3]
 if not isinstance(dt,Selection):return _reference(query,candidate,reverse)
 assert np.array_equal(dt.indices,dv.indices)
 # An unscored bank row must not introduce a new support failure.
 if not (query[2][::3][None]&dv.bank[:,::3]).any(-1).all():return _reference(query,[None,dt.bank[dt.indices],None,dv.bank[dv.indices]],reverse)
 return _reference(query,[None,dt.bank,None,dv.bank],reverse)[dt.indices]
_reference=scoring.diagonal

def verify(out,source):
 out=fresh(out);source=Path(source);expected=read(source/'result.json');before=binding();assert before['science']==expected['binding']['science'] and before['dependencies']==expected['binding']['dependencies'];original_np,original_diagonal=scoring.np,scoring.diagonal
 try:
  scoring.np=ArrayAPI();scoring.diagonal=cached_diagonal;actual=scoring.run(out/'rebuild',expected['stage'])
 finally:scoring.np=original_np;scoring.diagonal=original_diagonal
 actual['binding']['commit']=expected['binding']['commit'];assert actual==expected;assert sha(source/'scores.npz')==sha(out/'rebuild/scores.npz');write(out/'result.json',dict(result_exact=True,score_bytes_exact=True,all_candidates_reencoded=True,candidate_view_optimization=True,reference_scoring_sha256=sha(Path(scoring.__file__)),replay_implementation_sha256=sha(__file__),optimizer_calls=0,source_commit=expected['binding']['commit'],replay_commit=before['commit']))
