"""Exact frozen B0 trajectory parity before any new generator training."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime,packet,rows
from dunhuang.research.state_r27.common import read,write,sha,commit,fresh,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.trace import sample
def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();parent=Path(cfg["r28"])
    assert sha(parent/"artifacts_manifest.json")==cfg["r28_manifest_sha256"]
    manifest=read(parent/"artifacts_manifest.json")["files"]
    for name in ("models/T1/adapter.pt","evaluation/arrays.npz"):assert sha(parent/name)==manifest[name]["sha256"]
    old=np.load(parent/"evaluation/arrays.npz")
    d,n,s,f,e,info,a=runtime(cfg,0,parent/"models/T1/adapter.pt")
    arrays={};receipts=[];calls=0
    for role,key in (("train","train"),("holdout","holdout_exposed_R26")):
        values,_=rows(cfg,root,"aist",role);music=torch.tensor(np.stack([x["music"] for x in values]),device="cuda")
        for mode,oldmode in (("correct","correct"),("swapped","wrong"),("none","none")):
            a.record_attention=True
            pred,receipt,_=sample(d,a,None if mode=="none" else packet(values,mode),music,cfg["real_eval_seeds"][0]);calls+=1
            expected=old[f"{key}_T1_{oldmode}_0_normalized"]
            assert np.array_equal(pred.cpu().numpy(),expected),(role,mode)
            arrays[role+"_"+mode]=pred.cpu().numpy();receipts.append(dict(role=role,mode=mode,noise=receipt,original_R28_exact=True))
    result=dict(records=receipts,sampler_calls=calls,generated_trajectories=4*calls,optimizer_steps=0,instrumentation_changes_predictions=False,
        B0_original_R28_exact=True,parameters=sum(p.numel() for p in a.parameters()),operational_qualified=False)
    path=out/"arrays.npz";np.savez_compressed(path,**arrays);write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(path),checkpoint=info))
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(path,Path(replay)/path.name)
    a.close();print("R29_B0_PARITY_EXACT",calls,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
