from pathlib import Path
from dunhuang.io import sha,checked
from dunhuang.research.source_capacity.common import read,commit,fresh,same_arrays,forbid_optimizers,frozen,weight

def prior(config):
    root=Path(config["protected_source"]);public=read("reports/protected_source_summary.json");receipt=read("reports/protected_source_receipt.json")
    if sha("reports/protected_source_summary.json")!=receipt["summary_sha256"] or receipt["exit_status"]!=0:raise ValueError("Frozen R5 receipt differs")
    for stage in ("protected","information","planner"):
        for name in ("protocol.json","result.json"):
            if sha(root/stage/name)!=public["evidence"][stage+"/"+name]:raise ValueError("Frozen R5 stage differs")
    if sha(root/"protected/arrays.npz")!=public["private_arrays"]["protected/arrays.npz"]:raise ValueError("Frozen R5 arrays differ")
    if config["protected"]!=read(root/"protected/protocol.json")["config"]["protected"] or config["protection"]!=read(root/"protected/protocol.json")["config"]["protection"]:raise ValueError("Frozen protection/numeric criteria differ")
    return root,public

def selected_pairs(config,manifest):
    from dunhuang.data.split_registry import stable
    root,_=prior(config);p=read(root/"protected/protocol.json");cp=read(checked(read(checked(p["frozen_control"]))["capacity_protocol"]))
    old=cp["chosen_train_pairs"];groups={r["source_group"] for r in old};index={r["window_uid"]:r for r in manifest["rows"]}
    all_pairs=read(checked(read(Path(config["source"])/"pairs/protocol.json")["pairs"]))["train"]
    used={index[p[k]]["raw_motion"]["sha256"] for p in old for k in ("a","b")};new=[];seen=set()
    for pair in sorted(all_pairs,key=lambda p:stable([config["selection_seed"],p["a"],p["b"]])):
        if pair["source_group"] in groups|seen or any(index[pair[k]]["raw_motion"]["sha256"] in used for k in ("a","b")):continue
        new.append(pair);seen.add(pair["source_group"])
        if len(new)==config["probe"]["new_train_music_pairs"]:break
    if len(new)!=config["probe"]["new_train_music_pairs"]:raise ValueError("Disjoint known-Train probe population insufficient")
    for p in old+new:
        if any(index[p[k]]["role"]!="train" for k in ("a","b")):raise ValueError("Train-only objective probe required")
    return old,new
