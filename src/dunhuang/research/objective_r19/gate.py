"""Train-only admission for optional matched unique-information comparison."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import write
from dunhuang.research.source_capacity.common import read,commit

def means(rows,case,control,term):
    return float(np.mean([r["losses"][term] for r in rows if r["case"]==case and r["control"]==control]))

def assess(cfg,models):
    records=[]
    for case in cfg["cases"]:
        a,b=models["N1"],models["N2"]
        initial=means(a["initial_scores"],case,"correct","unknown")
        final=means(a["scores"],case,"correct","unknown")
        obs=means(a["scores"],case,"observation","unknown")
        observed_initial=means(a["initial_scores"],case,"correct","observed")
        observed_final=means(a["scores"],case,"correct","observed")
        learnable=final<=initial*cfg["qualification"]["unknown_ratio"] and final<obs and observed_final<=observed_initial*.9
        n2=means(b["scores"],case,"correct","unknown")
        records.append(dict(case=case,n1_unknown_ratio=final/max(initial,1e-30),
            n1_correct_unknown=final,n1_observation_mismatch_unknown=obs,
            n1_observed_ratio=observed_final/max(observed_initial,1e-30),n1_learnable=learnable,
            n2_unknown=n2,n2_preserves_basic=n2<=final*cfg["qualification"]["calibrated_unknown_tolerance"]))
    passed=all(r["n1_learnable"] for r in records)
    return dict(records=records,basic_learnability_passed=passed,
        unique_admitted=cfg["optional_unique"] and passed and all(r["n2_preserves_basic"] for r in records),
        gradnorm_executed=False,operational_qualified=False)

def run(cfg,root):
    root=Path(root);models={arm:read(root/"models"/arm/"result.json") for arm in cfg["arms"]}
    assert len({m["initial_motion"] for m in models.values()})==1
    assert len({m["initial_head"] for m in models.values()})==1
    assert models["N0"]["ledger"]==models["N1"]["ledger"]==models["N2"]["ledger"]
    result=dict(assess(cfg,models),code_commit=commit());path=root/"gate.json";assert not path.exists();write(path,result)
    print("R19_GATE",result,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True)
    a=p.parse_args();run(read(a.config),a.root)
