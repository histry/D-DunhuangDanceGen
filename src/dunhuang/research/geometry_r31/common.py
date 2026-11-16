"""R31 reads frozen source geometry and keeps historical contracts untouched."""
from pathlib import Path
import numpy as np
from dunhuang.research.state_r27.common import read,sha,checked
from dunhuang.research.continuity_r30.common import runtime as old_runtime,cohort,values

def parent_config():
    return read("src/dunhuang/research/continuity_r30/config.json")

def sources(cfg):
    p=Path(cfg["r30"]);assert sha(p/"artifacts_manifest.json")==cfg["r30_manifest_sha256"]
    rows,ledger,original=cohort(parent_config());pop=read("outputs/time_r29_01/coverage_data/result.json")["records"]
    result=[]
    for row in pop:
        result.append(dict(uid="aist_train_"+str(row["case"]),domain="aist",role="train",case=row["case"],data=rows[row["case"]],Root_observed=True))
    for domain in ("aist","fine"):
        for role in (("holdout",) if domain=="aist" else ("train","holdout")):
            data,items=values(parent_config(),domain,role)
            for d,item in zip(data,items):result.append(dict(uid=domain+"_"+role+"_"+str(item).replace(":","_"),domain=domain,role=role,case=item,data=d,Root_observed=True))
    old=np.load(checked(read("outputs/lodge_reference_r24_01/main/protocol.json")["arrays"]))
    for i in range(2):
        result.append(dict(uid="change_"+str(i),domain="change",role="exposed_transfer",case=i,
            data=dict(model_raw=old["source139"][i,64:160],model_points=old["source22"][i,64:160]),Root_observed=False))
    return result,ledger,original

def runtime(cfg):
    return old_runtime(parent_config())
