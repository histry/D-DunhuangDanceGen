"""Fit and independently replay feature RMS on the complete frozen Train population only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked,write
from dunhuang.research.source_capacity.common import read,commit,fresh
from dunhuang.research.boundary_context.common import Data,masks
from .adapter import ContentContextAdapter

def run(cfg,out,replay=None):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=fresh(out);pf=Path(cfg["preflight"]);assets=read(checked(read(pf/"protocol.json")["assets"]))
    scaler=np.load(checked(assets["scaler"]));data=Data(scaler);rows=[r for r in read(pf/"rows.json") if r["role"]=="train"]
    assert len(rows)==1833
    adapter=ContentContextAdapter(data.scale,data.bias).cuda().eval()
    total=np.zeros(606,np.float64);count=0
    with torch.no_grad():
        for start in range(0,len(rows),16):
            target,_=data.batch(rows[start:start+16]);known,_=masks(target,60,90);t=torch.zeros(len(target),device="cuda",dtype=torch.long)
            f,_=adapter.features(target,known,t)
            total+=(f.double().square().sum((0,1))).cpu().numpy();count+=len(target)*150
    rms=np.sqrt(total/count);inv=np.ones(606,np.float64)
    content=np.r_[4:151,306:453,457:604]
    reliable=content[rms[content]>=1e-4];inv[reliable]=1/rms[reliable]
    inv=inv.astype(np.float32)
    # Mask, clock, padding/contact columns retain their original units.
    np.save(out/"inv_rms.npy",inv)
    result=dict(rms=rms.tolist(),inv_rms=inv.tolist(),training_windows=len(rows),role="train",validation_tensor_reads=0,test_tensor_reads=0,
        method="per-column RMS over registered complete Train windows and fixed 8+8 context; no centering, no observed/target clipping",floor=1e-4,zero_rare_columns_keep_identity=True,
        optimizer_steps=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),preflight=read(pf/"protocol.json"),rows=rows,config=cfg))
    if replay:
        assert result==read(Path(replay)/"result.json") and np.array_equal(inv,np.load(Path(replay)/"inv_rms.npy"))
    print("R10_TRAIN_UNITS",len(rows),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
