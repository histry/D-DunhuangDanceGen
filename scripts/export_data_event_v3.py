"""Publish aggregate evidence only; private source ledgers stay on the server."""
import argparse,collections,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def export(out,receipt):
    out=Path(out);receipt=Path(receipt)
    r=read(out/"main/result.json");registry=read(out/"main/source_registry_v3.json");replay=read(out/"replay/replay.json")
    protocol=read(out/"main/protocol.json")
    if replay["source_raw_decodes"]!=8 or replay["all_event_ids_replayed"]!=499 or replay["optimizer_calls"]:raise ValueError("Full independent replay missing")
    if any(r[k] for k in ("optimizer_calls","statistics_fit_calls","edge_forward_calls","native_calls")):raise ValueError("Offline budget violated")
    counts=lambda name:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1]
    populations={}
    for dataset in ("aistpp","finedance","change"):
        rows=[s for s in registry["sources"] if s["dataset_id"]==dataset]
        populations[dataset]=dict(sources=len(rows),solo_admitted=sum(s["admission"]["solo_admitted"] for s in rows),
            unknown_performer=sum(not s.get("performer_uid") for s in rows),unknown_music_work=sum(not s.get("music_work_uid") for s in rows),
            full151_windows=sum(s.get("windows_with_full151_and_jukebox",0) for s in rows),
            raw_audio_integrity=dict(collections.Counter(s.get("asset_integrity",{}).get("audio","UNPAIRED_NO_AUDIO") for s in rows)),
            admission_status=dict(collections.Counter(s["admission"]["solo_status"] for s in rows)))
    protocols={}
    lookup={s["source_uid"]:s for s in registry["sources"]}
    for name,p in registry["protocols"].items():
        roles=p["roles"]
        entry=dict(identity_scope=p["identity_scope"],ready_for_new_training=p["ready_for_new_training"],
            role_sources=dict(collections.Counter(roles.values())),attestation=p.get("attestation"),
            role_windows={role:sum(lookup[uid].get("windows_with_full151_and_jukebox",0) for uid,r in roles.items() if r==role) for role in ("train","val","test")})
        if "folds" in p:
            entry["folds"]=[dict(fold=f["fold"],role_sources=dict(collections.Counter(f["roles"].values())),
                forbidden_identity_overlaps=f["attestation"]["forbidden_identity_overlaps"],index_rebuild_required=f["index_rebuild_required"],
                train_only_statistics_required=f["train_only_statistics_required"],existing_checkpoint_qualified=f["existing_full_catalog_checkpoint_qualified"]) for f in p["folds"]]
        protocols[name]=entry
    report=read(out/"main/change_event_e1_report.json")
    per_source=[{k:v for k,v in s.items() if k!="source_uid"} for s in report["per_source"]]
    public=dict(schema="public_data_event_v3_e1",execution_commit=(receipt/"execution_commit.txt").read_text().strip(),
        config=protocol["config"],populations=populations,protocols=protocols,result=r,per_recording=per_source,replay=replay,
        tests=dict(main=counts("pytest.log"),edge=counts("edge_check.log")),
        evidence={name:sha(out/name) for name in ("main/protocol.json","main/source_registry_v3.json","main/split_attestation_v3.json",
            "main/data_exposure_audit.json","main/source_seals.json","main/event_catalog_v2_shadow.json","main/change_event_e1_report.json",
            "main/event_diagnostics.json","main/planner_eligibility.npz","main/music_slots.json","main/result.json","replay/replay.json")},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    text=json.dumps(public)
    if any(s in text for s in ("/home/","aist_capture:","known_component:","source_uid","window_uid","smpl_poses")):raise ValueError("Private source evidence must remain on server")
    if not all(public["tests"].values()):raise ValueError("Regression evidence missing")
    write(out/"public_summary.json",public)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True)
    a=p.parse_args();export(a.out,a.receipt)
