from pathlib import Path
from dunhuang.io import checked,sha
from dunhuang.research.source_tasks.common import read,commit,fresh,forbid_optimizers,same_arrays

def frozen(config):
    root=Path(config["source"]);pub=read("reports/source_tasks_summary.json")
    for name in ("cf/protocol.json","cf/result.json","content/protocol.json","content/result.json","pairs/protocol.json","content_data/protocol.json"):
        if sha(root/name)!=pub["evidence"][name]:raise ValueError("Frozen experiment seal differs: "+name)
    cp=read(root/"cf/protocol.json");q=read(checked(cp["qualified"]))
    previous=read(checked(q["previous_preflight"]));manifest=read(checked(previous["manifest"]))
    return root,pub,previous,manifest

def weight(root,pub,part,arm):
    path=root/part/arm/("adapter.pt" if part=="cf" else "content.pt")
    if sha(path)!=pub["model_sha256"][part+"/"+arm]:raise ValueError("Frozen model seal differs")
    return path
