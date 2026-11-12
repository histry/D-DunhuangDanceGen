import json,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import sha,checked,write

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def commit():return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
def fresh(p):
    p=Path(p).resolve();p.mkdir(parents=True,exist_ok=False);return p
def same_arrays(a,b):
    x=dict(np.load(a));y=dict(np.load(b))
    if set(x)!=set(y) or any(not np.array_equal(x[k],y[k],equal_nan=True) for k in x):raise ValueError("Frozen array replay differs")
def forbid_optimizers():
    import torch
    def forbid(*a,**k):raise RuntimeError("Independent replay cannot construct optimizers")
    torch.optim.AdamW=forbid;torch.optim.Adam=forbid;torch.optim.SGD=forbid
def source_protocol(config):
    root=Path(config["source"]);pub=read("reports/source_necessity_summary.json")
    for name in ("preflight/protocol.json","preflight/manifest.json","preflight/supervision.npz","preflight/schedule.npz","g2/protocol.json","g2/result.json"):
        if sha(root/name)!=pub["evidence"][name]:raise ValueError("Published source artifact differs: "+name)
    p=read(root/"preflight/protocol.json");m=read(checked(p["manifest"]))
    return root,p,m
