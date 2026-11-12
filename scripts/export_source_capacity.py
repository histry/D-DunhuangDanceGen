"""Release only aggregated frozen audits, with no new training claim."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text())
def export(out,receipt,seal=False):
    out=Path(out);receipt=Path(receipt);execution=(receipt/"execution_commit.txt").read_text().strip()
    public=dict(schema="public_source_capacity_v1",execution_commit=execution,
        capacity=read(out/"capacity/result.json"),local=read(out/"local/result.json"),
        replays={k:read(out/(k+"_replay")/"replay.json") for k in ("capacity","local")},
        evidence={k+"/"+name:sha(out/k/name) for k in ("capacity","local") for name in ("protocol.json","result.json")},
        tests={k:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1] for k,name in (("main","pytest.log"),("edge","edge_check.log"))},
        optimizer_updates=0,statistics_fit_calls=0,human_reviewed=False,operational_qualified=False,g3_started=False,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0)
    if public["capacity"]["generations"]!=32 or public["replays"]["capacity"]["regenerated"]!=32 or public["replays"]["capacity"]["arrays_verified"]!=96:raise ValueError("Fixed frozen trajectory budget differs")
    for k in ("capacity","local"):
        if read(out/k/"protocol.json")["code_commit"]!=execution or public["replays"][k]["optimizer_calls"]:raise ValueError("Execution identity or optimizer invariant differs")
    if not all(public["tests"].values()):raise ValueError("Regression receipt missing")
    if any(v in json.dumps(public) for v in ("/home/","window_uid","source_uid","aist_capture:","password")):raise ValueError("Private identities in public artifact")
    if not seal:write(out/"public_summary.json",public);return
    if read(out/"public_summary.json")!=public or (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful sealed launcher required")
    baseline="6fb5bdbc2e721b054e249dd4469ec20add6f7e85"
    frozen=subprocess.check_output(["git","ls-tree","-r","--name-only",baseline,"--","src","configs"],text=True).splitlines()
    if subprocess.check_output(["git","diff","--name-only",baseline,execution,"--",*frozen],text=True).strip():raise ValueError("Existing baseline files changed")
    write(out/"public_receipt.json",dict(schema="source_capacity_sealed_receipt",baseline=baseline,execution_commit=execution,exit_status=0,
        verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),baseline_existing_source_and_configs_unchanged=True,
        summary_sha256=sha(out/"public_summary.json"),launcher_sha256=sha("scripts/run_source_capacity.sh"),
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        receipt_sha256={p.name:sha(p) for p in receipt.iterdir() if p.is_file()},optimizer_updates=0,replay_optimizer_calls=0,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);p.add_argument("--seal",action="store_true");a=p.parse_args();export(a.out,a.receipt,a.seal)
