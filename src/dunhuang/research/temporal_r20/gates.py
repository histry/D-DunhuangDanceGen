"""Preregistered Train-only structure and cache admission; DEV never selects."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import write
from dunhuang.research.source_capacity.common import read,commit
def qmean(rows,case,control="correct",times=(999,519,119,19)):
    return float(np.mean([r["unknown"] for r in rows if r["case"]==case and r["control"]==control and r["t"] in times]))

def run(cfg,root,which):
    root=Path(root)
    if which=="structure":
        a=read(root/"models/S0/result.json");b=read(root/"models/S1/result.json")
        physical=read(root/"structure_native/result.json")["records"];g=cfg["structure_gate"];cases=[]
        for case in cfg["train_cases"]:
            good=qmean(b["scores"],case);base=qmean(a["scores"],case)
            sides={c:qmean(b["scores"],case,c) for c in ("left_wrong","right_wrong","left_absent","right_absent")}
            ra=next(x for x in physical if x["case"]==case and x["arm"]=="S0" and x["control"]=="correct")
            rb=next(x for x in physical if x["case"]==case and x["arm"]=="S1" and x["control"]=="correct")
            passed=good<=base*g["q_tolerance"] and all(good<x for x in sides.values())
            passed=passed and rb["jerk_p95_mps3"]<=ra["jerk_p95_mps3"]*g["jerk_p95_tolerance"] and rb["activity_mps"]>=ra["activity_mps"]*g["minimum_activity_ratio"]
            passed=passed and rb["pose_error_rad"]<=ra["pose_error_rad"]*g["pose_tolerance"] and rb["root_rmse_m"]<=ra["root_rmse_m"]*g["root_tolerance"]
            edges=lambda rows:float(np.mean([r["raw_edge_root_rmse_m"] for r in rows if r["case"]==case and r["control"]=="correct" and r["t"] in (999,519,119,19)]))
            passed=passed and edges(b["scores"])<=edges(a["scores"])*g["raw_edge_tolerance"]
            cases.append(dict(case=case,joint_accepted=bool(passed),joint_unknown=good,split_unknown=base,side_errors=sides))
        selected="S1" if all(c["joint_accepted"] for c in cases) else "S0"
        r=read(root/"models"/selected/"result.json")
        result=dict(cases=cases,selected=selected,layout=r["layout"],checkpoint=r["checkpoint"],dev_used=False,
            scope="Exploratory Train-only selection, not operational or independent validation",operational_qualified=False)
        path=root/"structure_gate.json"
    elif which=="cache":
        a=read(root/"models/A2/result.json");samplers=read(root/"gap_rollout/result.json")["bindings"];cases=[]
        for case in cfg["train_cases"]:
            initial=qmean(a["initial_scores"],case);final=qmean(a["scores"],case)
            r=next(r for r in samplers if r["case"]==case);receipt=read(r["sampler"]["path"])
            actual=receipt["records"][-1]["unknown"];q19=qmean(a["scores"],case,times=(19,))
            cases.append(dict(case=case,q_retained=final<=initial*cfg["gate"]["q_retention"],actual_t19_unknown=actual,q_t19_unknown=q19,gap=actual/max(q19,1e-30)))
        result=dict(cases=cases,cache_admitted=all(c["q_retained"] for c in cases) and any(c["gap"]>=cfg["gate"]["minimum_gap"] for c in cases),
            dev_used=False,operational_qualified=False);path=root/"cache_gate.json"
    else:raise ValueError("Gate")
    assert not path.exists();write(path,dict(result,code_commit=commit()));print("R20_GATE",which,result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--which",choices=["structure","cache"],required=True)
    a=p.parse_args();run(read(a.config),a.root,a.which)
