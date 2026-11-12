"""Post-hoc diagnosis of fixed candidate support; never changes labels or thresholds."""
import argparse,collections,json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write
from dunhuang.research.source_identifiability.a2 import candidates
from dunhuang.research.transfer_audit.contracts import validate_song_task

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def run(source,out,replay_from=None):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    rows=read(source/"a2/manifest.json")["rows"];result=read(source/"a2/result.json")
    bank=dict(np.load(source/"a2/nodes.npz"));stats=dict(np.load(source/"a2/nuisance.npz"))
    cfg=read(source/"a2/protocol.json")["config"]
    ids=np.array([i for i,r in enumerate(rows) if r["role"]=="val"],int)
    _,v=candidates([rows[i] for i in ids],{k:x[ids] for k,x in bank.items()},{k:x[ids] for k,x in stats.items()},cfg["strict"],role_validator=validate_song_task)
    all_codes=collections.Counter({int(k):n for k,n in result["dense_rule_failure_intersections"].items()})
    val=collections.Counter({int(k):n for k,n in v["rule_failure_intersections"].items()})
    train=all_codes.copy();train.subtract(val)
    if any(n<0 for n in train.values()) or v["populations"]["val"]!=result["dense_population"]["val"]:raise ValueError("Subset support population differs")
    reports={}
    for role,counter in (("train",train),("val",val)):
        reports[role]=dict(comparisons=sum(counter.values()),all_rules_pass=counter[0],
            single_rule_failure_counts={name:counter[1<<i] for i,name in enumerate(("count","time_w1","histogram","energy","onset"))},
            individual_rule_pass_counts={name:sum(n for k,n in counter.items() if not k&(1<<i)) for i,name in enumerate(("count","time_w1","histogram","energy","onset"))},
            failure_intersections={str(k):n for k,n in sorted(counter.items()) if n})
    lineage=collections.Counter();unique={}
    for r in rows:unique.setdefault(r["raw_motion"]["sha256"],r)
    for r in unique.values():lineage[r["historical_role"]+"_to_"+r["role"]]+=1
    overlaps={}
    for k in ("music_uid","choreography_uid","recording_uid","source_group"):
        overlaps[k]=len({r[k] for r in rows if r["role"]=="train"}&{r[k] for r in rows if r["role"]=="val"})
    public=dict(schema="posthoc_fixed_support_diagnosis",analysis_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_execution_commit=read(source/"a2/protocol.json")["code_commit"],
        input_sha256={name:sha(source/name) for name in ("a2/result.json","a2/manifest.json","a2/nodes.npz","a2/nuisance.npz","a2/train_statistics.json")},
        reports=reports,historical_source_role_transitions=dict(lineage),known_identity_overlap_counts=overlaps,
        thresholds_changed=False,new_candidates_selected=False,optimizer_calls=0,test_tensor_reads=0,
        interpretation="Comparison counts are dependent; single-failure responsibility is diagnostic, not relaxed qualification")
    write(out/"public_support.json",public)
    if replay_from:
        if read(Path(replay_from)/"public_support.json")!=public:raise ValueError("Independent support count replay differs")
        write(out/"replay.json",dict(status="FIXED_SUPPORT_DIAGNOSIS_REPLAY_VERIFIED",optimizer_calls=0))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();run(a.source,a.out,a.replay_from)
