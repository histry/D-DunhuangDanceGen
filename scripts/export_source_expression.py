"""Seal separately executed R6 diagnosis, fixed Train probes and review preparation."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write
def read(p):return json.loads(Path(p).read_text())
def export(out,receipt,seal=False):
    out=Path(out);receipt=Path(receipt);execution=(receipt/"execution_commit.txt").read_text().strip();parts=("prepared","diagnosis","probe","music","review")
    public=dict(schema="source_expression_r6_public_v1",execution_commit=execution,**{k:read(out/k/"result.json") for k in parts},
        replays={k:read(out/(k+"_replay")/"replay.json") for k in parts},
        evidence={k+"/"+n:sha(out/k/n) for k in parts for n in ("protocol.json","result.json")},
        private_arrays={k+"/arrays.npz":sha(out/k/"arrays.npz") for k in ("diagnosis","probe")},
        tests={k:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/n).read_text()) or [None])[-1] for k,n in (("main","pytest.log"),("edge","edge_check.log"))},
        optimizer_updates=0,full_chain_started=False,new_mapping_training_started=False,human_reviewed=False,semantic_event_truth="UNKNOWN",operational_qualified=False,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0)
    for k in parts:
        if read(out/k/"protocol.json")["code_commit"]!=execution or public["replays"][k]["optimizer_calls"]:raise ValueError("Stage execution/replay differs")
    if len(public["probe"]["solves"])!=240 or len(public["probe"]["records"])!=480:raise ValueError("Registered probe budget differs")
    if public["review"]["actual_records"]!=0 or public["probe"]["optimizer_calls"] or not all(public["tests"].values()):raise ValueError("Execution boundary differs")
    if any(v in json.dumps(public) for v in ("/home/","window_uid","source_uid","aist_capture:","password")):raise ValueError("Private identity in public result")
    if not seal:write(out/"public_summary.json",public);return
    if read(out/"public_summary.json")!=public or (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful immutable receipt required")
    baseline="0265485e20442684fac2edaf9758bc982507f79e"
    frozen=subprocess.check_output(["git","ls-tree","-r","--name-only",baseline,"--","src","configs"],text=True).splitlines()
    if subprocess.check_output(["git","diff","--name-only",baseline,execution,"--",*frozen],text=True).strip():raise ValueError("Existing source/config baseline changed")
    write(out/"public_receipt.json",dict(schema="source_expression_r6_sealed_receipt",baseline=baseline,execution_commit=execution,
        verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),summary_sha256=sha(out/"public_summary.json"),exit_status=0,
        existing_baseline_source_and_configs_unchanged=True,launcher_sha256=sha("scripts/run_source_expression.sh"),
        receipt_sha256={f.name:sha(f) for f in receipt.iterdir() if f.is_file()},started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        optimizer_updates=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);p.add_argument("--seal",action="store_true")
    a=p.parse_args();export(a.out,a.receipt,a.seal)
