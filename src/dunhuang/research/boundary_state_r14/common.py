from pathlib import Path
import hashlib
from dunhuang.io import checked,sha
from dunhuang.research.source_capacity.common import read
from dunhuang.research.masked_denoiser.common import assets_and_population

def population(cfg):
    root=Path(cfg["r13_root"]);seal=read("reports/r13_summary.json")
    for name in ("main/result.json","main_rollout/result.json","main_native/result.json","inspection/result.json"):
        if sha(root/name)!=seal["evidence"][name]:raise ValueError("Frozen R13 artifact differs: "+name)
    oldcfg=read(cfg["r13_config"]);assets,rows,tiny,monitor,fits=assets_and_population(oldcfg)
    groups=sorted(set(r["source_group"] for r in rows),key=lambda g:hashlib.sha256((str(cfg["seed"])+g).encode()).hexdigest())
    held=set(groups[:cfg["head_holdout_groups"]])
    fit=[r for r in rows if r["source_group"] not in held];holdout=[r for r in rows if r["source_group"] in held]
    assert len(groups)==36 and not set(r["source_group"] for r in fit)&held
    return assets,fit,holdout,groups,monitor
