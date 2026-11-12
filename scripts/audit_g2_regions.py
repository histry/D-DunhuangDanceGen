"""Read-only region errors on already frozen, exactly replayed generations."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import checked,sha,write
from dunhuang.research.source_identifiability.g2 import make_condition,source_error

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def run(source,out,replay_from=None):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    summary=read(source/"public_summary.json");saved=read(source/"g2/protocol.json")
    if sha(source/"g2/result.json")!=summary["evidence"]["g2/result.json"]:raise ValueError("Frozen G2 readouts differ")
    old=read(checked(saved["source_protocol"]));manifest=read(checked(old["paired_manifest"]));index={r["window_uid"]:r for r in manifest["rows"]}
    old_result=read(Path(saved["source_protocol"]["path"]).parent/"result.json")
    records=[];seals={};legacy_fields=0
    for rec in summary["g2"]["generation_records"]:
        case,seed,name=rec["case"],rec["seed"],rec["control"]
        uids=[old["evaluation_uids"][case],old["contrast_uids"][case]]
        s=int(name.rsplit("_",1)[-1]) if name[-1:] in ("0","1") else 0
        uid=uids[s];r=index[uid];reference=np.load(checked(r["motion"]));condition,keys,_,_=make_condition(r,saved["config"]["old_event_source"])
        known=condition.known[0,:,79:127].reshape(150,8,6).all(-1).numpy();key=keys[0].numpy()
        p=source/"g2"/("case_"+str(case)+"_seed_"+str(seed))/name/"native.npy";motion=np.load(p)
        metric=dict(known_error_radians=source_error(motion,reference,known),
            unprovided_error_radians=source_error(motion,reference,~known),key_error_radians=source_error(motion,reference,key))
        if name!="null":
            previous=next(x for x in old_result["records"] if x["case"]==case and x["seed"]==seed and x["control"]==name)
            if any(metric[k]!=previous[k] for k in metric):raise ValueError("Historical region errors differ")
            legacy_fields+=len(metric)
        records.append(dict(case=case,seed=seed,control=name,**metric,provided_pose_frames=int(known.any(-1).sum()),
            hidden_frames=int((~known).any(-1).sum()),key_frames=int(key.any(-1).sum()),
            key_joint_observations=int(key.sum()),upper8_joint_observations=int(known.sum())))
        seals[str(p.relative_to(source))]=sha(p)
    result=dict(schema="posthoc_g2_region_diagnosis",analysis_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        g2_execution_commit=summary["execution_commit"],source_summary_sha256=sha(source/"public_summary.json"),
        source_protocol_sha256=sha(source/"g2/protocol.json"),motion_sha256=seals,records=records,
        legacy_region_fields_exactly_replayed=legacy_fields,optimizer_calls=0,new_generations=0,
        label_scope="same frozen historical source pair, not unseen-source generalization")
    write(out/"public_regions.json",result)
    if replay_from:
        if read(Path(replay_from)/"public_regions.json")!=result:raise ValueError("Region diagnostic replay differs")
        write(out/"replay.json",dict(status="FROZEN_GENERATION_REGION_REPLAY_VERIFIED",optimizer_calls=0,new_generations=0,records=len(records)))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(a.source,a.out,a.replay_from)
