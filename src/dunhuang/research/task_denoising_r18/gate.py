"""Train-only learnability admission; does not grant physical or semantic qualification."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import write
from dunhuang.research.source_capacity.common import read,commit

def assess(result,cfg):
    cases=cfg["tiny_cases"];answers=[]
    def mean(rows,case,task,control):
        return float(np.mean([r["unknown_mse"] for r in rows if r["case"]==case and r["task"]==task and r["control"]==control]))
    for case in cases:
        for task in cfg["tasks"]:
            a=mean(result["initial_scores"],case,task,"correct");b=mean(result["scores"],case,task,"correct")
            wrong=mean(result["scores"],case,task,"content");obs=mean(result["scores"],case,task,"observation")
            answers.append(dict(case=case,task=task,initial_unknown_mse=a,final_unknown_mse=b,
                ratio=b/max(a,1e-12),learned=b<=a*cfg["tiny_gate"]["unknown_ratio"] and b<obs,
                content_win=b<wrong,observation_win=b<obs))
    enough=sum(r["learned"] for r in answers)>=cfg["tiny_gate"]["minimum_tasks"]
    content=sum(r["content_win"] for r in answers)>=cfg["tiny_gate"]["minimum_content_wins"]
    grads=all(r["updates"]>0 and r["ranking_updates"]>0 and r["motion_nonzero_gradients"]==r["updates"] for r in result["task_counts"].values())
    return dict(cases=answers,learnability_passed=enough and grads,content_passed=content,
        main_admitted=enough and grads and content)

def run(cfg,root):
    root=Path(root);m={mode:read(root/"tiny"/mode/"result.json") for mode in ("self","unique")}
    assert m["self"]["ledger"]==m["unique"]["ledger"]
    assert m["self"]["initial_motion"]==m["unique"]["initial_motion"] and m["self"]["initial_head"]==m["unique"]["initial_head"]
    assert m["self"]["motion_parameters"]==m["unique"]["motion_parameters"]
    result=dict(code_commit=commit(),models={k:assess(v,cfg) for k,v in m.items()},
        main_admitted=False,D3_admitted=False,operational_qualified=False)
    result["main_admitted"]=result["models"]["self"]["learnability_passed"] and result["models"]["unique"]["main_admitted"]
    path=root/"tiny_gate.json";assert not path.exists();write(path,result)
    print("R18_GATE",result["main_admitted"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);a=p.parse_args();run(read(a.config),a.root)
