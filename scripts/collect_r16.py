"""Compact public R16 evidence; full traces stay in the sealed outputs directory."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import sha,write
from dunhuang.research.source_capacity.common import read,commit

def run(root):
    root=Path(root);check=read(root/"independent_checks.json");pre=read(root/"preflight/result.json");n=read(root/"native/result.json")
    evidence={phase+"/"+f:sha(root/phase/f) for phase in ("preflight","preflight_replay","samples","samples_replay","native","native_replay") for f in ("protocol.json","result.json")}
    evidence["videos/protocol.json"]=sha(root/"videos/protocol.json");receipts={}
    dirs=[Path(str(root)+"_receipts"),Path(str(root)+"_final_receipts")]
    for number in (1,2):
        old=Path("outputs/continuous_r16_0"+str(number)+"_receipts")
        if old.exists():dirs.append(old)
    for p in dirs:
        code=(p/"exit_status.txt").read_text().strip()
        receipts[p.name]=dict(code_commit=(p/"execution_commit.txt").read_text().strip(),exit_status=code,
            files={f.name:sha(f) for f in p.iterdir() if f.is_file()})
    assert receipts[Path(str(root)+"_receipts").name]["exit_status"]=="0" and receipts[Path(str(root)+"_final_receipts").name]["exit_status"]=="0"
    arms={}
    for arm in read(root/"preflight/protocol.json")["config"]["arms"]:
        rr=[r for r in n["records"] if r["arm"]==arm]
        arms[arm]=dict(cases=len(rr),native_passes=sum(r["native_ok"] for r in rr),halo_native_passes=sum(r["halo_native_ok"] for r in rr),
            center_jerk_mean_mps3=float(np.mean([r["center_jerk_max_mps3"] for r in rr])),
            edge_root_rmse_mean_m=float(np.mean([r["edge_root_rmse_m"] for r in rr])),
            event_pose_mean_rad=float(np.mean([r["event_pose_error_rad"] for r in rr])),
            common_span_pose_mean_rad=float(np.mean([r["common_generated_span_pose_rad"] for r in rr])))
    samples=read(root/"samples/result.json")
    result=dict(schema="continuous_generation_r16_evidence_v1",baseline="da412faebec3d05332d3abf8ec439c493a9c38cc",
        config=read(root/"preflight/protocol.json")["config"],data=pre,arms=arms,records=n["records"],qualification=n["qualification"],
        checks=check,evidence=evidence,receipts=receipts,videos=read(root/"videos/protocol.json")["records"],
        first_sampler_calls=samples["full_window_sampler_calls"],replay_sampler_calls=samples["full_window_sampler_calls"],
        optimizer_steps=0,c2_model_implemented=False,c2_training_started=False,new_whole_song_generations=0,
        human_reviewed=False,semantic_event_truth="UNKNOWN",operational_qualified=False,
        limitations=["Exposed Train mechanism study, not independent generalization","AIST source segments are proxies, not semantic event truth",
            "C0 copies parent outside30; C1 generates all150, common-span pose errors reported separately",
            "Canonical 225-frame Jukebox uses explicit crop ownership, not full-context extraction",
            "Frozen ChangE plan contract verification is not ChangE style-generation qualification",
            "No FlowMDM or DanceFusion architecture/weight reproduction"])
    write("reports/r16_summary.json",result);write("reports/r16_checks.json",check)
    write("reports/r16_receipt.json",dict(code_commit=commit(),summary_sha256=sha("reports/r16_summary.json"),evidence=evidence,receipts=receipts,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,new_whole_song_generations=0,operational_qualified=False))
    print("R16_SEALED",sha("reports/r16_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
