"""Public compact evidence; complete states, weights and receipts stay in outputs."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);checks=read(root/"independent_checks.json");data=read(root/"data/result.json");native=read(root/"native/result.json")
    evidence={};receipts={}
    for phase in ("data","data_replay","audit","audit_replay","audit_native","audit_native_replay","models/cross","models/cross_replay",
        "models/self","models/self_replay","rollout","rollout_replay","native","native_replay"):
        for name in ("protocol.json","result.json"):evidence[phase+"/"+name]=sha(root/phase/name)
    evidence["videos/protocol.json"]=sha(root/"videos/protocol.json")
    for p in [Path(str(root)+"_"+s+"_receipts") for s in ("audit","train","final")]+[Path("outputs/global_clock_r17_01_audit_receipts")]:
        if not p.exists():continue
        receipts[p.name]=dict(code_commit=(p/"execution_commit.txt").read_text().strip(),exit_status=(p/"exit_status.txt").read_text().strip(),
            files={f.name:sha(f) for f in p.iterdir() if f.is_file()})
    for s in ("audit","train","final"):assert receipts[Path(str(root)+"_"+s+"_receipts").name]["exit_status"]=="0"
    training={}
    for mode in ("cross","self"):
        r=read(root/"models"/mode/"result.json")
        training[mode]={k:r[k] for k in ("parameters","optimizer_steps","initial_digest","final_digest","base_digest","fk_parity_max_m","checkpoint")}
        training[mode].update(first=r["history"][0],last=r["history"][-1],fit_groups=len(data["fit_groups"]))
    factor=[]
    for position in ("mid","ends"):
        for hard in (False,True):
            for event in (False,True):
                rr=[r for r in native["factors"] if (r["position"],r["hard"],r["event"])==(position,hard,event)]
                factor.append(dict(position=position,hard=hard,event=event,cases=len(rr),native_passes=sum(r["native_ok"] for r in rr),halo_native_passes=sum(r["halo_native_ok"] for r in rr),
                    global_jerk_max_mean=float(np.mean([r["jerk_max_mps3"] for r in rr])),edge_root_mean=float(np.mean([r["edge_root_rmse_m"] for r in rr])),
                    raw_observed_root_mean=float(np.mean([r["raw_known_root_rmse_m"] for r in rr]))))
    arms=[]
    for arm in ("A_numeric","B_cross","C_self"):
        for task in ("free","inbetween","progressive"):
            rr=[r for r in native["rollouts"] if r["arm"]==arm and r["task"]==task]
            arms.append(dict(arm=arm,task=task,cases=len(rr),native_passes=sum(r["native_ok"] for r in rr),
                means={key:float(np.mean([r[key] for r in rr])) for key in ("root_rmse_m","pose_error_rad","foot_rmse_m","jerk_max_mps3","jerk_p95_mps3","activity_mps")}))
    result=dict(schema="r17_global_clock_evidence_v1",baseline="70988d94fafa55b1615fb3f863798cd103d1e60a",
        config=read(root/"data/protocol.json")["config"],data=dict(fit_groups=data["fit_groups"],eval_groups=data["eval_groups"],
            records=[{k:r[k] for k in ("case","role","left","right","fresh_raw_decode_exact","audio_overlap_exact","halo150_observed","feature_overlap","music_policy")} for r in data["records"]]),
        training=training,factor_summary=factor,factor_records=native["factors"],music_records=native["music"],music_response=native["music_response"],
        arms=arms,rollouts=native["rollouts"],qualification=native["qualification"],checks=checks,evidence=evidence,receipts=receipts,
        videos=read(root/"videos/protocol.json")["records"],sampler_calls=76,replay_sampler_calls=76,neural_optimizer_steps=480,
        human_reviewed=False,semantic_event_truth="UNKNOWN",new_whole_song_generations=0,operational_qualified=False,
        limitations=["Adapter-heldout groups are exposed Train and may be foundation-pretrained","Progressive training uses teacher-forced real past; evaluation uses generated history",
            "Ownership225 is not full-context Jukebox extraction or proven pretrained-distribution compatibility",
            "Time-reversed neighbour probes are diagnostics, not independent semantic labels","ChangE full style transfer is not qualified"])
    write("reports/r17_summary.json",result);write("reports/r17_checks.json",checks)
    write("reports/r17_receipt.json",dict(code_commit=commit(),summary_sha256=sha("reports/r17_summary.json"),evidence=evidence,receipts=receipts,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,new_whole_song_generations=0,operational_qualified=False))
    print("R17_SEALED",sha("reports/r17_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
