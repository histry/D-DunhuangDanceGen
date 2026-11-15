"""Use sealed R22 results without reconstructing or selecting a new population."""
from pathlib import Path
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers,same_arrays

def inputs(cfg):
    root=Path(cfg["parent"]);receipt=read("reports/r22_release/r22_receipt.json")
    assert sha("reports/r22_release/r22_summary.json")==receipt["summary_sha256"]
    for name,digest in receipt["evidence"].items():assert sha(root/name)==digest,("R22 seal",name)
    pc=read(cfg["parent_config"]);assert pc==read(root/"data/protocol.json")["config"]
    return root,pc,read(root/"data/result.json")

def finish(out,result,cfg,replay=None):
    if replay:assert result==read(Path(replay)/"result.json"),"Fresh-process logical result differs"
    write(Path(out)/"result.json",result)
    write(Path(out)/"protocol.json",dict(config=cfg,code_commit=commit(),parent_receipt_sha256=sha("reports/r22_release/r22_receipt.json"),
        optimizer_steps=0,whole_song_generations=0,operational_qualified=False))

def arrays(out,name,values,replay=None):
    p=Path(out)/name;np.savez_compressed(p,**values)
    if replay:same_arrays(p,Path(replay)/name)
    return name
