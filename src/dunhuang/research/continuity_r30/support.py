"""Pre-normalization support, kept distinct from already-clipped training targets."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import runtime,values
from dunhuang.research.state_r27.common import read,write,fresh,commit,forbid_optimizers

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();d,n,s,f,e,info,a=runtime(cfg);a.close()
    z=np.load(root/"audit/arrays.npz");records=[]
    for domain in ("aist","fine"):
        for role in ("train","holdout"):
            data,items=values(cfg,domain,role)
            for i,row in enumerate(data):
                raw=torch.tensor(row["model_raw"],device="cuda")
                with torch.no_grad():
                    affine=raw*n.scaler.scale_.to(raw)+n.scaler.min_.to(raw)
                    normalized=n.normalize(raw.clone())
                expected=affine.clamp(*n.scaler.feature_range) if n.scaler.clip else affine
                assert torch.equal(expected,normalized)
                assert np.array_equal(normalized.cpu().numpy(),row["model_target"])
                source=z[f"{domain}_{role}_{i}_source22"];roundtrip=z[f"{domain}_{role}_{i}_roundtrip22"]
                delta=roundtrip[64:160]-source[64:160]
                records.append(dict(domain=domain,role=role,item=items[i],
                    affine_preclip_Core_outside_fraction={name:float((affine[64:160,sl].abs()>1).float().mean()) for name,sl in (("C4",slice(0,4)),("Root",slice(4,7)),("body",slice(7,139)))},
                    post_normalize_Core_outside_fraction=float((normalized[64:160].abs()>1).float().mean()),
                    roundtrip_position_rmse_m=float(np.sqrt(np.mean(delta**2))),
                    roundtrip_joint_displacement_max_m=float(np.linalg.norm(delta,axis=-1).max())))
    result=dict(records=records,normalizer_clip=bool(n.scaler.clip),normalizer_unchanged=True,
        actual_training_target_matches=True,preclip_and_postclip_statistics_separated=True,
        scope="SOURCE_MODEL0_OFFICIAL22; SAME_SEALED_ROUNDTRIP_ARRAYS; NOT_NATIVE24",
        optimizer_steps=0,sampler_calls=0,generated_trajectories=0,operational_qualified=False)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),checkpoint=info))
    print("R30_PRECLIP_SUPPORT",len(records),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
