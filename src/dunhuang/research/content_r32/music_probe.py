"""Secondary matched-noise music intervention; sensitivity is not musical coordination."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import *
from dunhuang.research.state_r27.common import write,fresh,commit,forbid_optimizers,same_arrays
from dunhuang.research.state_r27.trace import sample

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();pop,ledger,data,z=population(cfg)
    arrays={};records=[];noise=[];null=None
    music=torch.tensor(z["music"][[0,0,0,1,1,1]],device="cuda");events=[0,0,1,0,0,1];drops=[True,False,False,True,False,False]
    seeds=[320501,320511,320521,320501,320511,320521]
    for arm in ("CanonicalIO","AIST","ChangE"):
        d,n,s,f,e,info,b=runtime(cfg,None if arm=="CanonicalIO" else root/f"models/{arm}/adapter.pt");frozen=fingerprint(d,b)
        ref=reference([z["source139"][i,64:160,7:] for i in events],drops=drops)
        p,nr,_=sample(d,b,ref,music,seeds);raw=n.unnormalize(p.clone())
        with torch.no_grad():points=f(raw,s)[:,:,:22]
        assert nr["row_sha256"][:3]==nr["row_sha256"][3:]
        if null is None:null=p[[0,3]].cpu().numpy()
        else:assert np.array_equal(null,p[[0,3]].cpu().numpy())
        for key,x in (("normalized",p),("139",raw),("22",points)):arrays[arm+"_"+key]=x.cpu().numpy()
        for i,event in enumerate((-1,0,1)):
            rel=points[:,64:160]-points[:,64:160,:1]
            records.append(dict(arm=arm,event=event,normalized_music_response_rmse=float((p[i,64:160]-p[i+3,64:160]).square().mean().sqrt()),
                body_Root_relative_music_response_rmse_m=float((rel[i]-rel[i+3]).square().mean().sqrt()),same_latent_noise=True,same_reference=True,
                scope="MUSIC_INTERVENTION_SENSITIVITY_NOT_COORDINATION_OR_PAIRING_TRUTH"))
        noise.append(dict(arm=arm,noise=nr));assert_frozen(d,b,frozen);b.close()
    np.savez_compressed(out/"arrays.npz",**arrays)
    result=dict(records=records,noise=noise,sampler_calls=3,generated_trajectories=18,optimizer_steps=0,fixed_NoRef_exact_across_arms=True,
        seeds=seeds,not_used_for_training_or_selection=True,operational_qualified=False,whole_song_generations=0)
    if replay:assert result==read(Path(replay)/"result.json");same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
    write(out/"result.json",result);write(out/"protocol.json",dict(config=cfg,code_commit=commit(),arrays_sha256=sha(out/"arrays.npz")))
    print("R32_MUSIC_PROBE",3,18,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.root,a.out,a.replay_from)
