"""Read-only source response, distinct from directional following."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import sha,checked,write
from dunhuang.data.sources import ref
from .common import read,commit,fresh,source_protocol
from .math import response

def compute(root,manifest,masks,result):
    records=[]
    for pair_id,pair in enumerate(manifest["validation_pairs"]):
        a,b=pair["a"],pair["b"];index={r["window_uid"]:r for r in manifest["rows"]}
        ra=np.load(checked(index[a]["motion"]));rb=np.load(checked(index[b]["motion"]))
        oa=np.load(checked(index[a]["observed"]))[:,79:127].reshape(150,8,6).all(-1)
        ob=np.load(checked(index[b]["observed"]))[:,79:127].reshape(150,8,6).all(-1)
        regions=dict(observed=oa&ob,provided=masks[a+"__known"]&masks[b+"__known"],hidden=oa&ob&~(masks[a+"__known"]|masks[b+"__known"]))
        for pref in [r for r in result["preferences"] if r["case"]==pair_id]:
            folder=Path(root)/("case_"+str(pair_id)+"_seed_"+str(pref["seed"]))
            xa=np.load(folder/(pref["arm"]+"_0")/"native.npy");xb=np.load(folder/(pref["arm"]+"_1")/"native.npy")
            records.append(dict(case=pair_id,arm=pref["arm"],seed=pref["seed"],
                regions={k:response(xa,xb,ra,rb,v) for k,v in regions.items()},
                source_a_preference=pref["source_a_preference"],source_b_preference=pref["source_b_preference"],
                source_following_pass=pref["source_following_pass"],activity_retention=pref["activity_retention"]))
    arms=sorted({r["arm"] for r in records})
    return dict(records=records,mean_provided_response={a:float(np.mean([r["regions"]["provided"]["normalized_response"] for r in records if r["arm"]==a])) for a in arms},
        label_scope="measured source response, not semantic ground truth",optimizer_calls=0,operational_qualified=False)

def run(config,out,replay_from=None):
    out=fresh(out);root,p,m=source_protocol(config);masks=dict(np.load(checked(p["supervision"])))
    g=read(root/"g2/result.json");n=0
    for file in sorted((root/"g2").glob("case_*/*/native.npy")):
        other=root/"g2_replay"/file.relative_to(root/"g2")
        if not np.array_equal(np.load(file),np.load(other)):raise ValueError("Prior exact replay no longer agrees")
        n+=1
    result=compute(root/"g2",m,masks,g);result["frozen_generations_checked"]=n
    write(out/"result.json",result)
    write(out/"protocol.json",dict(code_commit=commit(),config=config,source_result=ref(root/"g2/result.json"),source_manifest=p["manifest"],source_supervision=p["supervision"],optimizer_calls=0))
    if replay_from:
        if read(Path(replay_from)/"result.json")!=result:raise ValueError("Source effect replay differs")
        write(out/"replay.json",dict(status="FRESH_SOURCE_RESPONSE_REPLAY",optimizer_calls=0,regenerated=0))
    print("SOURCE_RESPONSE",result["mean_provided_response"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
