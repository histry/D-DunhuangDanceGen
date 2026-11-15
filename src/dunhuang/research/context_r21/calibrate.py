"""Train-only fixed gradient budget for source-geometry support proxies."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.objective_r19.common import parameters,gradient,group_norm
from .engine import load,cache_data,state,model_meta
from .objective import losses

def run(cfg,root,out,replay=None):
    out=fresh(out);forbid_optimizers();w,d,data,p,h=load(cfg,root,"F");cache=cache_data(cfg,root,d,data)
    before=model_meta(w,h);records=[];arrays={};split=sum(x.numel() for x in w.selected())
    names=("support_velocity","support_anchor","swing_clearance")
    for case in cache["population"]["follow_cases"]:
        for t in (999,519,119,19):
            s,_,proxy=state(cfg,cache,case,t);y,m,o,v,k,hard,meta,ts,xt=s
            pred,_=p(xt,m,v,k,meta,ts);terms,counts=losses(pred,y,v,k,hard,cache,o,True,proxy)
            norms={name:group_norm(gradient(terms[name],parameters(w,h)),split) for name in ("unknown",)+names}
            arrays[str(case)+"_"+str(t)]=np.array([[norms[n][g] for g in ("motion","head")] for n in ("unknown",)+names])
            records.append(dict(case=case,t=t,norms=norms,counts={n:counts[n] for n in names}))
    reference={g:float(np.median([r["norms"]["unknown"][g] for r in records])) for g in ("motion","head")}
    weights={}
    for name in names:
        ratios=[float(np.median([r["norms"][name][g] for r in records]))/max(reference[g],1e-12) for g in reference]
        weights[name]=min(cfg["maximum_support_weight"],cfg["support_gradient_budget"]/(len(names)*max(1.,*ratios)))
    assert model_meta(w,h)==before
    np.savez_compressed(out/"norms.npz",**arrays)
    result=dict(weights=weights,reference=reference,records=records,optimizer_steps=0,dev_used=False,support_truth="SOURCE_GEOMETRY_PROXY",operational_qualified=False)
    if replay:
        assert result==read(Path(replay)/"result.json");same_arrays(out/"norms.npz",Path(replay)/"norms.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),source=ref(Path(root)/"models/B3/result.json")))
    p.close();print("R21_SUPPORT_WEIGHTS",weights,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
