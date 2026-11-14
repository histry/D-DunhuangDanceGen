"""Seal R10 completed fixed audits and final-checkpoint contrast, including failures."""
import argparse,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha
from dunhuang.research.source_capacity.common import read

def mean(rows,key):
    values=[r[key] for r in rows if key in r and r[key] is not None]
    return float(np.mean(values)) if values else None

def run(audit,training,post_attempt,balanced=None):
    audit=Path(audit);training=Path(training);receipts={}
    for label,root in (("audit",audit),("training",training)):
        folder=Path(str(root)+"_receipts");assert (folder/"exit_status.txt").read_text().strip()=="0"
        receipts[label]=dict(code_commit=(folder/"execution_commit.txt").read_text().strip(),started=(folder/"started_utc.txt").read_text().strip(),finished=(folder/"finished_utc.txt").read_text().strip(),
            files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
    post=Path(str(training)+"_post_"+post_attempt+"_receipts");assert (post/"exit_status.txt").read_text().strip()=="0"
    receipts["post"]=dict(code_commit=(post/"execution_commit.txt").read_text().strip(),files={p.name:sha(p) for p in post.iterdir() if p.is_file()})
    failed=Path(str(training)+"_post_receipts")
    assert (failed/"exit_status.txt").read_text().strip()=="1"
    receipts["post_01_failed"]=dict(code_commit=(failed/"execution_commit.txt").read_text().strip(),files={p.name:sha(p) for p in failed.iterdir() if p.is_file()})
    evidence={};stages={}
    for label,root,phases in (("audit",audit,["p0","p0_replay","root","root_replay","hermite","hermite_replay"]),("training",training,["C0","C0_replay","C1","C1_replay","generation","generation_replay","media_"+post_attempt,"feature_probe_"+post_attempt])):
        for phase in phases:
            stages[label+"/"+phase]=read(root/phase/"protocol.json")["code_commit"]
            for name in ("protocol.json","result.json","interface_gate.json","media_checks.json"):
                file=root/phase/name
                if file.is_file():evidence[label+"/"+phase+"/"+name]=sha(file)
    identity_arrays=0
    for i in range(4):
        for name in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy","edited_native.npy","edited_fk.npy"):
            a=np.load(training/"generation"/("val_"+str(i)+"_zero")/name)
            b=np.load(training/"generation"/("val_"+str(i)+"_C1_removed")/name)
            assert np.array_equal(a,b);identity_arrays+=1
    p0=read(audit/"p0/result.json");root=read(audit/"root/result.json");h=read(audit/"hermite/result.json")
    neural=read(training/"generation/result.json")
    bindings=read(training/"generation/protocol.json")["bindings"]
    for row in neural["records"]:
        binding=next(b for b in bindings if b["case"]==row["case"] and b["arm"]==row["arm"])
        folder=Path(binding["native"]["path"]).parent
        roi=read(folder/"roi_physical.json");global_report=read(folder/"edited_physical.json")
        row["roi_audit"]=roi["audit"];row["roi_failure_reasons"]=roi["absolute"]["reasons"];row["global_failure_reasons"]=global_report["absolute"]["reasons"]
    checks=read(training/"independent_checks.json");arms={}
    for arm in ("C0","C1"):
        data=read(training/arm/"result.json");assert data["optimizer_steps"]==200 and read(training/(arm+"_replay")/"result.json")["optimizer_steps"]==0
        scores={}
        for control in ("zero","observed","removed","wrong","shifted"):
            rows=[r for r in data["validation"] if r["control"]==control]
            scores[control]={k:mean(rows,k) for k in ("masked_cfg_mse","response_root_error_m","response_root_rms_m")}
        arms[arm]=dict(checkpoint=data["checkpoint"],parameters=data["parameters"],steps=data["optimizer_steps"],cf_support_skips=data["cf_support_skips"],scores=scores,backbone_updated=data["backbone_updated"])
    groups={}
    for population in ("val","r8"):
        for arm in ("zero","C0","C1","C1_removed","C1_wrong"):
            rows=[r for r in neural["records"] if r["arm"]==arm and r["case"].startswith("val_")== (population=="val")]
            if not rows:continue
            groups[population+"/"+arm]=dict(cases=len(rows),native_roi=sum(r["roi_native_ok"] for r in rows),native_global=sum(r["global_native_ok"] for r in rows),
                target_native=sum(r["target_native_ok"] for r in rows),root_rmse_m=mean(rows,"root_rmse_m"),rotation_error_rad=mean(rows,"rotation_error_rad"),source_error_rad=mean(rows,"source_error_rad"),
                joint_jerk_mps3_p95=mean([r["audit"] for r in rows],"joint_jerk_mps3_p95"),foot_skate_mps_p95=mean([r["audit"] for r in rows],"foot_skate_mps_p95"),
                roi_joint_jerk_mps3_p95=mean([r["roi_audit"] for r in rows],"joint_jerk_mps3_p95"),roi_foot_skate_mps_p95=mean([r["roi_audit"] for r in rows],"foot_skate_mps_p95"))
    u=np.arange(1,31)/31;zero_cf_error=.15*np.sqrt(np.mean((u**3*(10+u*(-15+6*u)))**2)/3)
    summary=dict(schema="r10_condition_dependency_root_foot_summary",baseline="44235357b9763bc8b047e4cda1579035faa3d7b9",p0=p0,
        root_capacity=dict(population=root["population"],research_passes=root["research_passes"],native_passes=root["native_passes"],records=root["records"]),
        hermite_capacity=h,training=arms,generation_groups=groups,generation_records=neural["records"],synthetic_zero_response_error_m=float(zero_cf_error),
        feature_probe=read(training/("feature_probe_"+post_attempt)/"result.json"),removed_equals_zero_arrays=identity_arrays,checks=checks,evidence=evidence,stage_commits=stages,receipts=receipts,adapter_optimizer_steps=400,backbone_optimizer_steps=0,
        new_neural_generations=neural["new_neural_generations"]*2,new_whole_song_generations=0,operational_qualified=False,human_reviewed=False,cultural_authenticity="UNKNOWN",
        limits=["C0/C1 isolate only synthetic Root response loss; shared geometry/interface changes not individually isolated","Synthetic state response is not physical contact or artistic truth","Context-only bridge is not full ChangE source replay","Finite Root/quintic proposals are not global impossibility proofs"])
    if balanced:
        br=Path(balanced)
        for label,folder in (("c2",Path(str(br)+"_receipts")),("c2_post",Path(str(br)+"_post_01_receipts"))):
            assert (folder/"exit_status.txt").read_text().strip()=="0"
            receipts[label]=dict(code_commit=(folder/"execution_commit.txt").read_text().strip(),files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})
        for phase in ("units","units_replay","C2","C2_replay","generation","generation_replay","feature_probe_01","media_01"):
            stages["balanced/"+phase]=read(br/phase/"protocol.json")["code_commit"]
            for name in ("protocol.json","result.json","interface_gate.json","media_checks.json","inv_rms.npy"):
                file=br/phase/name
                if file.is_file():evidence["balanced/"+phase+"/"+name]=sha(file)
        bt=read(br/"C2/result.json");bg=read(br/"generation/result.json");bc=read(br/"independent_checks.json")
        assert bt["optimizer_steps"]==200 and bc["neural_arrays"]==64 and bc["units_val_test_tensor_reads"]==0
        bscore={}
        for control in ("zero","observed","removed","wrong","shifted"):
            rows=[r for r in bt["validation"] if r["control"]==control]
            bscore[control]={k:mean(rows,k) for k in ("masked_cfg_mse","response_root_error_m","response_root_rms_m")}
        bgroup={}
        for population in ("val","r8"):
            rows=[r for r in bg["records"] if r["case"].startswith("val_")==(population=="val")]
            rois=[read(br/"generation"/(r["case"]+"_C2")/"roi_physical.json") for r in rows]
            bgroup[population]=dict(cases=len(rows),native_roi=sum(r["roi_native_ok"] for r in rows),native_global=sum(r["global_native_ok"] for r in rows),
                root_rmse_m=mean(rows,"root_rmse_m"),rotation_error_rad=mean(rows,"rotation_error_rad"),source_error_rad=mean(rows,"source_error_rad"),
                roi_joint_jerk_mps3_p95=mean([r["audit"] for r in rois],"joint_jerk_mps3_p95"),roi_foot_skate_mps_p95=mean([r["audit"] for r in rois],"foot_skate_mps_p95"),
                both_joins_nonregression=sum(all(v["maximum"]<=u["maximum"]+1e-6 for v,u in zip(r["joins"],r["original_joins"])) for r in rows))
        summary["balanced_correction"]=dict(training=bt,validation_scores=bscore,generation_groups=bgroup,generation=bg,units=read(br/"units/result.json"),checks=bc,feature_probe=read(br/"feature_probe_01/result.json"))
        summary["adapter_optimizer_steps"]+=200;summary["new_neural_generations"]+=2*len(bg["records"])
        checks["balanced_correction"]=bc;checks["total_neural_arrays"]=checks["neural_arrays"]+bc["neural_arrays"]
    write("reports/r10_summary.json",summary);write("reports/r10_checks.json",checks)
    write("reports/r10_receipt.json",dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),summary_sha256=sha("reports/r10_summary.json"),
        stage_commits=stages,receipts=receipts,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,formal_changed_paths=checks["formal_changed_paths"],
        new_whole_song_generations=0,operational_qualified=False))
    print("R10_SEALED",sha("reports/r10_summary.json"),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--audit",required=True);p.add_argument("--training",required=True);p.add_argument("--post",required=True);p.add_argument("--balanced");a=p.parse_args();run(a.audit,a.training,a.post,a.balanced)
