"""Read-only pre-transform domain audit: normalized outputs can already have been clipped."""
import argparse
from pathlib import Path
import torch
from dunhuang.research.state_r27.common import read,write,commit,forbid_optimizers,runtime
from .train import dataset

def run(cfg,root,out,replay=None):
    root=Path(root);out=Path(out);assert not out.exists();forbid_optimizers()
    d,n,s,f,e,info,a=runtime(read("src/dunhuang/research/state_r27/config.json"));a.close()
    rows=dataset(root,cfg["train_cases"]+cfg["holdout_cases"]);records=[]
    for case,row in zip(cfg["train_cases"]+cfg["holdout_cases"],rows):
        for mode in ("legacy","model"):
            raw=torch.tensor(row[mode+"_raw"],device="cuda")
            affine=raw*n.scaler.scale_.to(raw.device)+n.scaler.min_.to(raw.device)
            expected=affine.clamp(*n.scaler.feature_range) if n.scaler.clip else affine
            normalized=n.normalize(raw.clone());assert torch.equal(expected,normalized)
            records.append(dict(case=case,contract=mode,raw_root_Y_range=[float(raw[:,5].min()),float(raw[:,5].max())],
                affine_preclip_outside={name:float((affine[:,lo:hi].abs()>1).float().mean()) for name,lo,hi in (("C4",0,4),("Root",4,7),("body",7,139))},
                normalized_boundary_fraction=float((normalized.abs()>=1).float().mean()),normalized_outside_fraction=float((normalized.abs()>1).float().mean()),
                root_roundtrip_rmse_m=float((n.unnormalize(normalized.clone())[:,4:7]-raw[:,4:7]).square().mean().sqrt())))
    result=dict(records=records,scaler_transform_clips=bool(n.scaler.clip),scaler_feature_range=list(n.scaler.feature_range),
        archived_data_field_statistics_clipping_scope="POST_NORMALIZE_OUTSIDE_FRACTION_NOT_PRECLIP_INPUT_RATE",
        source_preclip_support_available_here=True,normalizer_unchanged=True,optimizer_steps=0,code_commit=commit())
    # Commit belongs to provenance, not numeric equality across later documentation publication.
    if replay:
        old=read(replay);assert {k:v for k,v in result.items() if k!="code_commit"}=={k:v for k,v in old.items() if k!="code_commit"}
    write(out,result);print("R28_PRECLIP_AUDIT_READY",bool(n.scaler.clip),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.replay_from)
