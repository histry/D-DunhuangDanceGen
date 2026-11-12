"""Read-only aggregate context: candidate exclusions, gradient blocks and occupancy null."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import write,sha,checked
from dunhuang.alignment.clock_data import read

def occupancy(counts):
    a=np.sort(counts);n=len(a);total=max(a.sum(),1)
    return np.array([a[-max(1,int(np.ceil(n*.01))):].sum()/total,
        (a==0).mean(),2*(np.arange(1,n+1)*a).sum()/(n*total)-(n+1)/n])

def run(out,dest):
    out=Path(out);p=read(out/"protocol.json");result=read(out/"scientific/result.json")
    replay=read(checked(read(out/"replay_verified.json")))
    if replay["status"]!="FRESH_LOCAL_IDENTIFIABILITY_REPLAY_VERIFIED":raise ValueError("Replay incomplete")
    rows=read(Path(p["source"])/"data/manifest.json")["rows"];candidate=read(out/"candidate_evidence.json")
    populations={}
    for role in ("train","val"):
        pairs=[r for r in candidate["evidence"] if rows[r["query"]]["role"]==role]
        populations[role]=dict(comparisons=len(pairs),queries_with_nonoverlap_candidates=len({r["query"] for r in pairs}),
            individual_rule_pass_counts={name:sum(r["rules"][name] for r in pairs) for name in candidate["failed_rule_counts"]},
            joint_pass_count=sum(r["accepted"] for r in pairs),
            window_start_frame_counts={str(v):sum(r["role"]==role and r["start_frame"]==v for r in rows) for v in sorted({r["start_frame"] for r in rows if r["role"]==role})})
    array=np.load(out/"scientific/gradients/gradients.npz");gradients={}
    old_config=read(Path(p["source"])/"protocol.json")["config"]
    weights=dict(retrieval=1.,shift=old_config["shift_weight"],partial=old_config["partial_weight"],null=old_config["null_weight"],weak=old_config["weak_weight"])
    for name in result["gradients"]["objectives"]:
        vectors=[array[str(batch)+"_"+name].reshape(2,144,144) for batch in p["config"]["gradient_batches"]]
        gradients[name]=dict(original_loss_weight=weights[name],mean_raw_total_norm=float(np.mean([np.linalg.norm(x) for x in vectors])),
            mean_weighted_total_norm=float(np.mean([np.linalg.norm(x)*weights[name] for x in vectors])),
            mean_input_block_norms={side:dict(token128=float(np.mean([np.linalg.norm(x[k,:,:128]) for x in vectors])),
                descriptor16=float(np.mean([np.linalg.norm(x[k,:,128:]) for x in vectors]))) for k,side in enumerate(("music","motion"))})
    null={}
    rng=np.random.default_rng(p["config"]["seed"]+700)
    for role,levels in result["geometry"].items():
        if role not in ("train","val"):continue
        null[role]={}
        for name,values in levels.items():
            h=values["hubness"];draws=np.stack([occupancy(np.bincount(rng.integers(h["candidate_tokens"],size=h["query_tokens"]),minlength=h["candidate_tokens"])) for _ in range(200)])
            null[role][name]=dict(query_tokens=h["query_tokens"],candidate_tokens=h["candidate_tokens"],uniform_occupancy_mean=draws.mean(0).tolist(),
                uniform_occupancy_95=np.percentile(draws,[2.5,97.5],axis=0).tolist(),metrics=["top_one_percent_share","unused_fraction","gini"],
                scope="finite-count uniform occupancy context, not a learned baseline or semantic truth")
    record=dict(schema="local_identifiability_supplement_v1",execution_commit=p["code_commit"],candidate_populations=populations,
        failed_rule_counts=candidate["failed_rule_counts"],gradient_blocks=gradients,hubness_occupancy_null=null,
        gradient_scope="same final F checkpoint across four Train batches; raw Euclidean gradients; virtual SGD does not reproduce historical AdamW",
        source_gradients_sha256=sha(out/"scientific/gradients/gradients.npz"),optimizer_calls=0,operational_qualified=False)
    encoded=json.dumps(record)
    if any(s in encoded for s in ("/home/","known_component:","aist_capture:","window_uid")):raise ValueError("Private identity in aggregate supplement")
    write(dest,record)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--dest",required=True);a=p.parse_args();run(a.out,a.dest)
