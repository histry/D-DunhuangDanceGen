"""Original baselines stay frozen; source and physical proxies keep explicit scopes."""
import argparse
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,sha
from dunhuang.research.source_capacity.common import read,commit,fresh,forbid_optimizers
from .metrics import source_metrics,motion_metrics,timing_proxy

def run(cfg,root,out,replay=None):
    out=fresh(out);root=Path(root);forbid_optimizers()
    z=dict(np.load(root/"samples/arrays.npz"));receipt=read(root/"samples/result.json")
    parent=Path(cfg["parent"]);old=dict(np.load(checked(read(parent/"main/protocol.json")["arrays"])));r24=read(parent/"main/result.json")
    pop=read(root/"data/result.json");paired={r["case"]:dict(np.load(root/"data"/r["arrays"])) for r in pop["records"]};roles={r["case"]:r["role"] for r in pop["records"]}
    cue=old["cue_mask"][:,7:].all(-1).astype(bool);core=np.zeros(256,bool);core[64:160]=True;unknown=core&~cue;records=[]
    def measure(motion,points,references,reference_points,music,meta):
        row=dict(meta,quality=motion_metrics(motion,points,cue),source_retention_scope="Retimed reference proxy, not paired truth for new music",
            errors=[{name:source_metrics(motion,ref,points,rp,mask) for name,mask in (("unprompted",unknown),("cue",cue),("core",core))} for ref,rp in zip(references,reference_points)],
            timing=[timing_proxy(motion,ref,music) for ref in references])
        row["diagnostic_partition"]="Frozen R24 cue neighborhoods; new adapters have NO actual state-write cues"
        records.append(row)
    for row in receipt["receipts"]:
        motion=z[row["stem"]+"_139"];points=z[row["stem"]+"_22"]
        if row["dataset"]=="change":
            for i,(music,event) in enumerate(row["conditions"]):
                measure(motion[i],points[i],old["source139"],old["source22"],old["music"][music],dict(dataset="change",arm=row["arm"],mode=row["mode"],seed=row["seed"],music=music,event=event))
        else:
            for i,case in enumerate(row["cases"]):
                d=paired[case];measure(motion[i],points[i],[d["raw139"]],[d["source22"]],d["music"],dict(dataset="paired",arm=row["arm"],mode=row["mode"],seed=row["seed"],case=case,role=roles[case]))
    for seed in cfg["eval_seeds"]:
        for i,(music,event) in enumerate([(m,e) for m in range(2) for e in (-1,0,1)]):
            measure(old[str(seed)+"_139"][i],old[str(seed)+"_22"][i],old["source139"],old["source22"],old["music"][music],
                dict(dataset="change",arm="R24",mode="frozen_sparse",seed=seed,music=music,event=event))
    get=lambda arm,mode,seed,m,e:next(r for r in records if r["dataset"]=="change" and (r["arm"],r["mode"],r["seed"],r["music"],r["event"])==(arm,mode,seed,m,e))
    comparisons=[]
    for arm in cfg["arms"]:
        for seed in cfg["eval_seeds"]:
            for m in range(2):
                tests=[]
                for e in range(2):
                    new=get(arm,"correct",seed,m,e);sparse=get("R24","frozen_sparse",seed,m,e);base=get("R24","frozen_sparse",seed,m,-1)
                    er=new["errors"][e]["unprompted"];other=new["errors"][1-e]["unprompted"];old_error=sparse["errors"][e]["unprompted"]
                    q=new["quality"];qb=base["quality"];region="core_interior_unprompted"
                    values=dict(source_body_gain=er["body22_rad"]<=(1-cfg["min_gain"])*old_error["body22_rad"],
                        wrists_preserved=er["wrist_rotation_rad"]<=cfg["max_quality_ratio"]*old_error["wrist_rotation_rad"],
                        assigned_body_preferred=er["body22_rad"]<other["body22_rad"],assigned_upper10_preferred=er["upper10_rad"]<other["upper10_rad"],
                        global_jerk=q["jerk_p95_mps3"]<=cfg["max_quality_ratio"]*qb["jerk_p95_mps3"],
                        core_jerk=q["regions"][region]["p95"]<=cfg["max_quality_ratio"]*qb["regions"][region]["p95"],
                        outside_jerk=q["regions"]["outside_core"]["p95"]<=cfg["max_quality_ratio"]*qb["regions"]["outside_core"]["p95"])
                    tests.append(dict(event=e,tests=values,passed=all(values.values())))
                comparisons.append(dict(arm=arm,seed=seed,music=m,events=tests,passed=all(x["passed"] for x in tests)))
    phase_contrasts=[]
    for seed in cfg["eval_seeds"]:
        for m in range(2):
            for e in range(2):
                correct=get("phase","correct",seed,m,e);base=get("event","correct",seed,m,e);removed=get("phase","phase_removed",seed,m,e);wrong=get("phase","phase_wrong",seed,m,e)
                phase_contrasts.append(dict(seed=seed,music=m,event=e,correct_source=correct["errors"][e]["unprompted"],event_only_source=base["errors"][e]["unprompted"],
                    phase_removed_source=removed["errors"][e]["unprompted"],phase_wrong_source=wrong["errors"][e]["unprompted"],
                    correct_timing=correct["timing"][e],event_only_timing=base["timing"][e],
                    phase_correct_better_than_removed=correct["errors"][e]["unprompted"]["body22_rad"]<removed["errors"][e]["unprompted"]["body22_rad"],
                    phase_correct_better_than_wrong=correct["errors"][e]["unprompted"]["body22_rad"]<wrong["errors"][e]["unprompted"]["body22_rad"]))
    result=dict(records=records,comparisons=comparisons,phase_contrasts=phase_contrasts,
        passed_groups={arm:sum(r["passed"] for r in comparisons if r["arm"]==arm) for arm in cfg["arms"]},
        original_R24_gate_unchanged=read("reports/r24_release/r24_summary.json")["qualification"],full_native_status="NOT_EVALUABLE",
        human_reviewed=False,contact_truth="UNKNOWN",whole_song_generations=0,operational_qualified=False,optimizer_steps=0)
    if replay:assert result==read(Path(replay)/"result.json")
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,arrays_sha256=sha(root/"samples/arrays.npz")))
    print("R25_EVALUATION",result["passed_groups"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read(a.config),a.root,a.out,a.replay_from)
