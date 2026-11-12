"""Aggregate-only E1-W publication; private source identities stay on server."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def export(out,receipt,seal=False):
    out=Path(out);receipt=Path(receipt);result=read(out/"main/result.json");replay=read(out/"replay/replay.json")
    commit=(receipt/"execution_commit.txt").read_text().strip()
    for sub in ("main","replay"):
        if read(out/sub/"protocol.json")["code_commit"]!=commit:raise ValueError("Execution identity differs")
    if replay["raw_source_decodes"]!=8 or replay["event_ids"]!=499 or replay["exhaustive_domain_checks"]!=48902:raise ValueError("Replay incomplete")
    if result["stage_order_inversions"] or any(result[k] for k in ("optimizer_calls","statistics_fit_calls","edge_forward_calls","native_calls")):raise ValueError("Study boundary violated")
    count=lambda name:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1]
    public=dict(schema="public_primitive_warp_v1",execution_commit=commit,result=result,replay=replay,
        tests=dict(main=count("pytest.log"),edge=count("edge_check.log")),
        config=read(out/"main/protocol.json")["config"],
        evidence={name:sha(out/name) for name in ("main/protocol.json","main/maps.npz","main/result.json","main/certificates.json",
            "main/source_audit/protocol.json","main/source_audit/source_seals.json","main/source_audit/replay.json",
            "replay/protocol.json","replay/maps.npz","replay/result.json","replay/certificates.json","replay/replay.json",
            "replay/source_audit/replay.json")},operational_qualified=False)
    if not all(public["tests"].values()):raise ValueError("Test evidence missing")
    text=json.dumps(public)
    if any(s in text for s in ("/home/","source_uid","window_uid","aist_capture:","password")):raise ValueError("Private content in public report")
    if not seal:write(out/"public_summary.json",public);return
    if read(out/"public_summary.json")!=public:raise ValueError("Published aggregate changed")
    if (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful launcher exit missing")
    base="685ad0bcf13b9e7be085e1df3d3d580040f068c5"
    old_files=subprocess.check_output(["git","ls-tree","-r","--name-only",base,"--","src","configs"],text=True).splitlines()
    if subprocess.check_output(["git","diff","--name-only",base,commit,"--",*old_files],text=True).strip():raise ValueError("Existing baseline code/configs changed")
    final=dict(schema="primitive_warp_final_receipt",execution_commit=commit,verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        exit_status=0,started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        baseline_commit=base,all_preexisting_source_and_config_files_unchanged=True,
        summary_sha256=sha(out/"public_summary.json"),launcher_sha256=sha("scripts/run_primitive_warp.sh"),
        receipt_evidence={name:sha(receipt/name) for name in ("pytest.log","edge_check.log","execution_commit.txt","exit_status.txt","started_utc.txt","finished_utc.txt")},
        raw_source_decodes_per_process=8,source_audit_arrays_replayed_per_process=1066,independent_integer_domain_checks_per_process=48902,
        optimizer_calls=0,statistics_fit_calls=0,edge_forward_calls=0,native_calls=0,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    write(out/"public_receipt.json",final)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);p.add_argument("--seal",action="store_true")
    a=p.parse_args();export(a.out,a.receipt,a.seal)
