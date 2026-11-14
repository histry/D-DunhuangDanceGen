from pathlib import Path
from dunhuang.io import sha,checked
from dunhuang.research.source_capacity.common import read
from dunhuang.research.boundary_state_r14.common import population

def source(cfg,*,manifest=False):
    root=Path(cfg["r14_root"]);seal=read("reports/r14_summary.json")
    assert sha("reports/r14_summary.json")==read("reports/r14_receipt.json")["summary_sha256"]
    for name in ("audit/result.json","audit/protocol.json","capacity/result.json","capacity/protocol.json","capacity_native/result.json"):
        if sha(root/name)!=seal["evidence"][name]:raise ValueError("Frozen R14 artifact differs: "+name)
    listing=read(root/"artifact_manifest.json");hashes={f["path"]:f["sha256"] for f in listing["files"]}
    if manifest:
        for path,digest in hashes.items():checked(dict(path=path,sha256=digest))
    assets,_,hold,_,monitors=population(read(cfg["r14_config"]))
    rows=read(root/"capacity/result.json")["bindings"];groups=[r["row"]["source_group"] for r in rows]
    assert len(rows)==8 and len(set(groups))==8 and all(b["row"]["role"]=="train" for b in rows)
    assert set(groups)<={r["source_group"] for r in hold}
    assert not set(groups)&{r["row"]["source_group"] for r in monitors}
    for b in rows:
        path=b["trace"];checked(dict(path=path,sha256=hashes[path]))
    return assets,rows,monitors,hashes
