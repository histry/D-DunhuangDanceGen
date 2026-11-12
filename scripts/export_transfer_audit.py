"""Public aggregate evidence and post-exit seal for the separated A2/G2 audits."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def export(out,receipt,seal=False):
    out=Path(out);receipt=Path(receipt);a=read(out/"a2/result.json");g=read(out/"g2/result.json")
    ar=read(out/"a2_replay/replay.json");gr=read(out/"g2_replay/replay.json");commit=(receipt/"execution_commit.txt").read_text().strip()
    if g["generations"]!=32 or gr["regenerated"]!=32 or g["legacy_predictions_and_generations_exactly_replayed"]!=76:raise ValueError("Frozen G2 reproduction incomplete")
    if any(r["optimizer_calls"] for r in (a,g,ar,gr)) or a["test_tensor_reads"]:raise ValueError("Optimizer/Test boundary violated")
    for sub in ("a2","a2_replay","g2","g2_replay"):
        if read(out/sub/"protocol.json")["code_commit"]!=commit:raise ValueError("Execution identity differs")
    counts=lambda name:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1]
    names=("a2/protocol.json","a2/result.json","a2/manifest.json","a2/source_seals.json","a2/train_statistics.json","a2/nodes.npz","a2/nuisance.npz","a2/candidates.npz","a2/sparse_candidates.npz","a2_replay/replay.json",
        "g2/protocol.json","g2/result.json","g2/traces.npz","g2/latents.npz","g2_replay/replay.json")
    public=dict(schema="public_transfer_audit_v1",execution_commit=commit,a2=a,g2=g,replay=dict(a2=ar,g2=gr),
        tests=dict(main=counts("pytest.log"),edge=counts("edge_check.log")),evidence={name:sha(out/name) for name in names},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    text=json.dumps(public)
    if any(s in text for s in ("/home/","source_uid","window_uid","aist_capture:","password")):raise ValueError("Private identity in public artifact")
    if not all(public["tests"].values()):raise ValueError("Test evidence missing")
    if not seal:write(out/"public_summary.json",public);return
    if read(out/"public_summary.json")!=public or (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful published receipt missing")
    base="4adee590d810c5dedc928be7b2a21258166ff86e"
    old_files=subprocess.check_output(["git","ls-tree","-r","--name-only",base,"--","src","configs"],text=True).splitlines()
    changes=subprocess.check_output(["git","diff","--name-only",base,commit,"--",*old_files],text=True).splitlines()
    if changes!=["src/dunhuang/research/source_identifiability/a2.py"]:raise ValueError("Unexpected old source/config mutation")
    # This compatibility extension keeps the old default validator; it is the sole old worker change.
    write(out/"public_receipt.json",dict(schema="transfer_audit_final_seal",execution_commit=commit,verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        exit_status=0,started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        baseline_commit=base,existing_runtime_and_config_unchanged=True,legacy_a2_default_validator_preserved=True,
        summary_sha256=sha(out/"public_summary.json"),launcher_sha256=sha("scripts/run_transfer_audit.sh"),
        receipt_evidence={n:sha(receipt/n) for n in ("pytest.log","edge_check.log","execution_commit.txt","exit_status.txt","started_utc.txt","finished_utc.txt")},
        optimizer_calls=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);p.add_argument("--seal",action="store_true")
    a=p.parse_args();export(a.out,a.receipt,a.seal)
