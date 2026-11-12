"""Publish aggregated support/factorial results and preserve baseline evidence."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))

def export(out,receipt,seal=False):
    out=Path(out);receipt=Path(receipt);commit=(receipt/"execution_commit.txt").read_text().strip()
    a=read(out/"a2/result.json");q=read(out/"preflight/result.json");g=read(out/"g2/result.json")
    replays={k:read(out/(k+"_replay")/"replay.json") for k in ("a2","preflight","g2")}
    if any(p["optimizer_calls"] for p in replays.values()):raise ValueError("Replay optimizer forbidden")
    qualified=q["qualification"]["qualified"];expected=52 if qualified else 0
    if g["generations"]!=expected or replays["g2"]["regenerated"]!=expected:raise ValueError("Registered generation budget incomplete")
    if g["optimizer_updates"]!=(2000 if qualified else 0):raise ValueError("Registered optimizer budget differs")
    for sub in ("a2","preflight","g2"):
        if read(out/sub/"protocol.json")["code_commit"]!=commit:raise ValueError("Study execution identity differs")
    counts=lambda name:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1]
    names=["a2/protocol.json","a2/result.json","a2/support_model.npz","a2/time.npz","a2/association.npz",
        "preflight/protocol.json","preflight/result.json","preflight/source_seals.json","preflight/manifest.json","preflight/supervision.npz","preflight/schedule.npz","g2/protocol.json","g2/result.json"]
    public=dict(schema="public_source_necessity_v1",execution_commit=commit,a2=a,qualification=q,g2=g,replays=replays,
        tests=dict(main=counts("pytest.log"),edge=counts("edge_check.log")),evidence={name:sha(out/name) for name in names},
        model_sha256={arm:sha(out/"g2"/arm/"adapter.pt") for arm in ("M0","M1","M2","M3","M3_shuffle")} if qualified else {},
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    if not all(public["tests"].values()):raise ValueError("Test receipt missing")
    text=json.dumps(public)
    if any(s in text for s in ("/home/","source_uid","window_uid","aist_capture:","password")):raise ValueError("Private source content in public artifact")
    if not seal:write(out/"public_summary.json",public);return
    if read(out/"public_summary.json")!=public or (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful sealed run absent")
    base="c7257d26a1881438eb6ee38e399f9c3d26c789ae"
    old_files=subprocess.check_output(["git","ls-tree","-r","--name-only",base,"--","src","configs"],text=True).splitlines()
    if subprocess.check_output(["git","diff","--name-only",base,commit,"--",*old_files],text=True).strip():raise ValueError("Existing baseline source/config changed")
    write(out/"public_receipt.json",dict(schema="source_necessity_final_receipt",execution_commit=commit,
        verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),baseline_commit=base,
        baseline_existing_source_and_configs_unchanged=True,exit_status=0,
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        summary_sha256=sha(out/"public_summary.json"),launcher_sha256=sha("scripts/run_source_necessity.sh"),
        receipt_sha256={n:sha(receipt/n) for n in ("pytest.log","edge_check.log","execution_commit.txt","exit_status.txt","started_utc.txt","finished_utc.txt")},
        main_optimizer_updates=g["optimizer_updates"],replay_optimizer_calls=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);p.add_argument("--seal",action="store_true")
    a=p.parse_args();export(a.out,a.receipt,a.seal)
