from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,pinned
from dunhuang.research.source_capacity.common import read

def cases(cfg):
    parent=Path(cfg["parent"]);seal=read("reports/r10_summary.json")
    assert sha(parent/"generation/protocol.json")==seal["evidence"]["balanced/generation/protocol.json"]
    bindings=read(parent/"generation/protocol.json")["bindings"];assert len(bindings)==8
    result=[]
    for b in bindings:
        assert b["arm"]=="C2"
        folder=Path(checked(b["native"])).parent;q=read(folder/"request.json")
        checked(q["r10_adapter"]);checked(q["context"]);checked(q["features"]);checked(b["fk"])
        target=b["target"]
        if b["case"].startswith("val_"):
            full=np.load(checked(target["source_native"]));old=full[target["start_frame"]:target["start_frame"]+150].copy()
        else:old=np.load(checked(target["native"])).copy()
        assert len(old)>=150
        result.append(dict(name=b["case"],binding=b,request=q,original=old,parent_folder=folder))
    for i,item in enumerate(result):
        j=(i+1)%4 if i<4 else 4+(i-3)%4
        item["wrong_context"]=result[j]["request"]["context"];item["wrong_case"]=result[j]["name"]
    return result

def native_environment(q):
    import sys
    sys.path.insert(0,str(pinned(q["legacy_root"],q["legacy_commit"])))
    from data_pipeline.prepare_domains import _baseline_environment
    _baseline_environment()
