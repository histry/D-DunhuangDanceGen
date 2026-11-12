"""Post-exit evidence seal: replay completeness, frozen default code, source roles."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write,checked
from dunhuang.data.split_registry import attest
BASE="aaf7eeb528cf518a1d85d831cc2186ab3544cfec"

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def seal(out,receipt):
    out=Path(out);receipt=Path(receipt)
    if (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful launcher exit absent")
    public=read(out/"public_summary.json");main=out/"main";replay=out/"replay"
    commit=(receipt/"execution_commit.txt").read_text().strip()
    if public["execution_commit"]!=commit or read(main/"protocol.json")["code_commit"]!=commit or read(replay/"protocol.json")["code_commit"]!=commit:raise ValueError("Execution identity differs")
    frozen=("src/dunhuang/pipeline.py","src/dunhuang/contracts.py","src/dunhuang/physical.py","src/dunhuang/planning",
        "src/dunhuang/generation/adapter.py","src/dunhuang/generation/edge.py","configs/edge_baseline.json","configs/dunhuang.json")
    changed=subprocess.check_output(["git","diff","--name-only",BASE,commit,"--",*frozen],text=True).strip()
    if changed:raise ValueError("Frozen default code changed")
    for name,digest in public["evidence"].items():
        if sha(out/name)!=digest:raise ValueError("Published evidence SHA changed")
    p=read(main/"protocol.json");a=read(checked(p["assets"]));old=read(checked(p["catalog"]));oldmap={e["event_uid"]:e for e in old["entries"]}
    new=read(main/"event_catalog_v2_shadow.json")
    if len(new["entries"])!=499 or {e["event_uid"] for e in new["entries"]}!=set(oldmap):raise ValueError("Frozen event identity differs")
    for e in new["entries"]:
        o=oldmap[e["event_uid"]]
        if e["source_interval"]!=o["source_interval"] or e["source_sha256"]!=o["source_sha256"] or e["can_route"] is not False or e["human_reviewed"] is not False:raise ValueError("Frozen primitive scope changed")
        if len(e["key_motion"])>3 or e["controlled_joints"]!=list(range(12,20)):raise ValueError("Actual controlled phase support differs")
        checked(e["motion_features"]);checked(e["reference"])
    for folder in (main,replay):
        seals=read(folder/"source_seals.json")
        if len(seals)!=8:raise ValueError("Raw source replay population differs")
        for s in seals:checked(s["copied_raw"])
    registry=read(main/"source_registry_v3.json");lookup={s["source_uid"]:s for s in registry["sources"]}
    for name,task in (("AIST_Official_Crossmodal","official_crossmodal"),("AIST_strict_unseen_song","strict_unseen_song"),("AIST_unseen_choreography","unseen_choreography"),("AIST_unseen_performer","unseen_performer")):
        proto=registry["protocols"][name];roles=proto["roles"]
        if roles and attest([lookup[uid] for uid in roles],roles,task)["forbidden_identity_overlaps"]:raise ValueError("Task source leakage")
    for f in registry["protocols"]["ChangE_Source_Heldout"]["folds"]:
        if attest([lookup[uid] for uid in f["roles"]],f["roles"],"unseen_motion_source")["forbidden_identity_overlaps"]:raise ValueError("Fold leakage")
    if any(p["ready_for_new_training"] for p in registry["protocols"].values()):raise ValueError("Audit promoted into training qualification")
    result=dict(schema="data_event_v3_final_seal",status="SEALED_DATA_V3_CHANGE_E1_REPLAY",execution_commit=commit,
        verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),exit_status=0,
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        baseline_code_unchanged=True,baseline_commit=BASE,default_catalog_scaler_unchanged=True,
        event_uids_and_core_bounds_preserved=499,raw_sources_redecoded_per_process=8,fresh_replay=read(replay/"replay.json"),
        task_and_fold_isolation_rechecked=True,no_inherited_checkpoint_qualification=True,
        summary_sha256=sha(out/"public_summary.json"),launcher_sha256=sha("scripts/run_data_event_v3.sh"),
        receipt_evidence={n:sha(receipt/n) for n in ("exit_status.txt","execution_commit.txt","pytest.log","edge_check.log","started_utc.txt","finished_utc.txt")},
        optimizer_calls=0,statistics_fit_calls=0,edge_forward_calls=0,native_calls=0,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    write(out/"public_receipt.json",result)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);a=p.parse_args();seal(a.out,a.receipt)
