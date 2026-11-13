"""Publish aggregates and a post-exit seal; private arrays remain on the server."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write
def read(p):return json.loads(Path(p).read_text())
def export(out,receipt,seal=False,control_root=None):
    out=Path(out);receipt=Path(receipt);execution=(receipt/"execution_commit.txt").read_text().strip()
    control=Path(control_root) if control_root else out
    if control_root:
        cfg=read("configs/reference_event_readout_v1.json")["frozen_control"]
        if control.resolve()!=Path(cfg["root"]).resolve():raise ValueError("Unregistered frozen control root")
        for name,digest in cfg["files"].items():
            if sha(control/name)!=digest:raise ValueError("Frozen single-step evidence changed")
    stage_roots={"control":control,"events":out}
    public=dict(schema="reference_guidance_public_v2",execution_commit=execution,
        control=read(control/"control/result.json"),events=read(out/"events/result.json"),
        replays={k:read(stage_roots[k]/(k+"_replay")/"replay.json") for k in ("control","events")},
        evidence={k+"/"+n:sha(stage_roots[k]/k/n) for k in ("control","events") for n in ("protocol.json","result.json")},
        private_array_sha256={k+"/"+n:sha(stage_roots[k]/k/n) for k,n in (("control","arrays.npz"),("events","scores.npz"),("events","fits.npz"))},
        tests={k:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/n).read_text()) or [None])[-1] for k,n in (("main","pytest.log"),("edge","edge_check.log"))},
        optimizer_updates=0,human_reviewed=False,operational_qualified=False,full_chain_started=False,mapping_training_started=False,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0)
    public["stage_execution_commits"]={k:read(stage_roots[k]/k/"protocol.json")["code_commit"] for k in stage_roots}
    if not all(public["tests"].values()):raise ValueError("Regression receipts missing")
    if public["control"]["full_dances_generated"] or public["control"]["native_evaluations"]!=44:raise ValueError("Single-step budget differs")
    if public["replays"]["control"]["native_arrays_verified"]!=132:raise ValueError("Independent Native array replay incomplete")
    for k in ("control","events"):
        if read(stage_roots[k]/k/"protocol.json")["code_commit"]!=(cfg["execution_commit"] if control_root and k=="control" else execution) or public["replays"][k]["optimizer_calls"]:raise ValueError("Execution identity or replay optimizer differs")
    if any(s in json.dumps(public) for s in ("/home/","window_uid","source_uid","aist_capture:","password")):raise ValueError("Private identities in aggregate")
    if not seal:write(out/"public_summary.json",public);return
    if read(out/"public_summary.json")!=public or (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful launcher required")
    baseline="e531ea7ca8f32299c9a222ef8b6ae681ace79d1e"
    files=subprocess.check_output(["git","ls-tree","-r","--name-only",baseline,"--","src","configs"],text=True).splitlines()
    if subprocess.check_output(["git","diff","--name-only",baseline,execution,"--",*files],text=True).strip():raise ValueError("Existing baseline source/config changed")
    write(out/"public_receipt.json",dict(schema="reference_guidance_sealed_receipt",baseline=baseline,execution_commit=execution,verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        exit_status=0,baseline_existing_source_and_configs_unchanged=True,summary_sha256=sha(out/"public_summary.json"),stage_execution_commits=public["stage_execution_commits"],launcher_sha256=sha("scripts/run_reference_events.sh" if control_root else "scripts/run_reference_guidance.sh"),
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),receipt_sha256={p.name:sha(p) for p in receipt.iterdir() if p.is_file()},
        replay_optimizer_calls=0,replay_fit_calls=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);p.add_argument("--seal",action="store_true");p.add_argument("--control-root");a=p.parse_args();export(a.out,a.receipt,a.seal,a.control_root)
