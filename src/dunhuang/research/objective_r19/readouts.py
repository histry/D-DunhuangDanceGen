"""Post-run aligned sampler metrics and source Native controls; no model selection."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write
from dunhuang.data.sources import ref
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from dunhuang.research.state_bridge.common import native_environment
from dunhuang.research.bridge_inpainting.study import audit_native
from dunhuang.research.continuous_r16.evaluate import angular
from .common import population

def balanced(pred,target,known):
    losses={};counts={}
    for prefix,mask in (("unknown",~known),("observed",known)):
        for kind,sl in (("root",slice(4,7)),("rotation",slice(7,151))):
            use=mask[:,sl];count=int(use.sum())
            losses[prefix+"_"+kind]=float(np.mean((pred[:,sl]-target[:,sl])[use]**2)) if count else 0.
            counts[prefix+"_"+kind]=count
        losses[prefix]=.5*(losses[prefix+"_root"]+losses[prefix+"_rotation"])
    return losses,counts

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pre,rows=population(cfg);native_environment(pre["assets"])
    by={r["case"]:r for r in rows};sources=[];records=[];mapping={}
    for row in rows:
        z=np.load(checked(row["arrays"]));x=z["native"].copy();fk,physical=audit_native(x)
        folder=out/("source_"+str(row["case"]));folder.mkdir()
        np.save(folder/"native.npy",x);np.save(folder/"fk.npy",fk);write(folder/"physical.json",physical)
        sources.append(dict(case=row["case"],source=row["left"],native_ok=physical["absolute"]["ok"],
            physical=ref(folder/"physical.json"),scope="Raw paired target is not guaranteed physical acceptance"))
        if replay:
            other=Path(replay)/folder.name
            assert np.array_equal(x,np.load(other/"native.npy")) and np.array_equal(fk,np.load(other/"fk.npy"))
            assert physical==read(other/"physical.json")
    for folder,b in read(root/"native/protocol.json")["saved"].items():
        z=np.load(checked(by[b["case"]]["arrays"]));tr=np.load(checked(b["trace"]))
        losses,counts=balanced(tr["raw_final"],z["encoded"],tr["known"])
        records.append(dict(case=b["case"],arm=b["arm"],control=b["control"],raw_losses=losses,counts=counts))
        mapping[b["case"],b["arm"],b["control"]]=root/"native"/folder/"native.npy"
    responses=[]
    if read(root/"gate.json")["unique_admitted"]:
        for case in cfg["cases"]:
            a=np.load(mapping[case,"U","correct"]);b=np.load(mapping[case,"U","typed_content"])
            responses.append(dict(case=case,root_response_rmse_m=float(np.sqrt(np.mean((a[:,4:7]-b[:,4:7])**2))),
                local_pose_response_rad=angular(a,b),scope="Full matched-noise trajectories, not same-state q probes"))
    result=dict(records=records,source_native=sources,unique_full_responses=responses,optimizer_steps=0,
        whole_song_generations=0,operational_qualified=False)
    if replay:
        old=read(Path(replay)/"result.json")
        # Physical source references differ only in path; their content hashes are exact.
        for a,b in zip(sources,old["source_native"]):assert a["physical"]["sha256"]==b["physical"]["sha256"]
        def comparable(r):return dict(r,source_native=[{k:v for k,v in s.items() if k!="physical"} for s in r["source_native"]])
        assert comparable(result)==comparable(old)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,
        scope="Post-run reporting only; never consumed by gate, calibration or training",
        losses="Same balanced unknown/observed Root/rotation definition as training and q gate"))
    print("R19_ALIGNED_READOUT",len(records),"source_Native",sum(s["native_ok"] for s in sources),"/",len(sources),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
