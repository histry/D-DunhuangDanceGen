"""Reuse only the completed, independently rebuilt A2 stage after a G2 failure."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import checked,write,sha

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def reuse(source,out):
    source=Path(source).resolve();out=Path(out)
    p=read(source/"a2/protocol.json");r=read(source/"a2_replay/replay.json")
    if r["status"]!="FRESH_V3_RAW_AUDIO_MOTION_CANDIDATE_REPLAY" or r["optimizer_calls"]:raise ValueError("Completed fresh A2 replay absent")
    files=["src/dunhuang/research/transfer_audit/a2.py","src/dunhuang/research/transfer_audit/contracts.py",
        "src/dunhuang/research/source_identifiability/a2.py","configs/transfer_audit_v1.json",
        "src/dunhuang/alignment","src/dunhuang/data","src/dunhuang/planning/time_map.py"]
    if subprocess.check_output(["git","diff","--name-only",p["code_commit"],"HEAD","--",*files],text=True).strip():raise ValueError("A2 implementation or data policy changed")
    for item in (p["assets"],p["registry"],p["paired_manifest"]):checked(item)
    for name in ("result.json","train_statistics.json","candidate_evidence.json"):
        if read(source/"a2"/name)!=read(source/"a2_replay"/name):raise ValueError("Completed A2 readouts differ")
    for name in ("nodes.npz","nuisance.npz","candidates.npz","sparse_candidates.npz"):
        x=dict(np.load(source/"a2"/name));y=dict(np.load(source/"a2_replay"/name))
        if set(x)!=set(y) or any(not np.array_equal(x[k],y[k]) for k in x):raise ValueError("Completed A2 arrays differ")
    for seal in read(source/"a2/source_seals.json"):
        for k in ("raw_motion","audio","source_clock","native"):checked(seal[k])
    for sub in ("a2","a2_replay"):(out/sub).symlink_to(source/sub,target_is_directory=True)
    write(out/"a2_reuse.json",dict(source_execution_commit=p["code_commit"],reuse_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        unchanged_source_policy=True,independent_raw_replay_verified=r,
        evidence={n:sha(source/n) for n in ("a2/protocol.json","a2/result.json","a2/manifest.json","a2/train_statistics.json","a2/candidates.npz","a2_replay/replay.json")}))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--out",required=True);a=p.parse_args();reuse(a.source,a.out)
