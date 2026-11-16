"""Preserve the failed run and reuse independently replayed, schedule-independent qualifications."""
import argparse,json,shutil,datetime,subprocess
from pathlib import Path
from dunhuang.io import sha,write
from dunhuang.data.sources import ref

def run(source,dest):
    base=Path.cwd().resolve();source=Path(source);dest=Path(dest)
    assert source.resolve().is_relative_to(base) if hasattr(Path,'is_relative_to') else base in source.resolve().parents
    assert base in dest.resolve().parents and not dest.exists()
    failed=Path(str(source)+"_finish_receipts");assert (failed/"exit_status.txt").read_text().strip()=="1"
    assert not (source/"artifacts_manifest.json").exists()
    files={str(p.relative_to(source)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in source.rglob("*") if p.is_file()}
    write(source/"artifacts_manifest.json",dict(files=files,file_count=len(files),total_bytes=sum(x["bytes"] for x in files.values()),status="SUPERSEDED_UNBALANCED_PAIRED_LEDGER; ALL_OUTPUTS_PRESERVED"))
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();dest.mkdir()
    folders=("data","data_replay","readout","readout_replay","models/pilot","models/pilot_replay");copied={}
    for name in folders:
        assert (source/name/"result.json").exists()
        shutil.copytree(source/name,dest/name)
        for p in (dest/name).rglob("*"):
            if p.is_file():
                rel=str(p.relative_to(dest));assert sha(p)==files[rel]["sha256"];copied[rel]=sha(p)
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    provenance=dict(code_commit=commit,source=str(source),superseded_manifest=ref(source/"artifacts_manifest.json"),
        reused_qualification_files=copied,decoder_steps_new=0,pilot_steps_new=0,not_a_new_execution_of_reused_qualifications=True,
        actual_source_presentations_first_main={str(c):n for c,n in __import__('collections').Counter(c for x in json.loads((source/"models/AIST/result.json").read_text())["history"] if x["task"]=="paired" for c in x["cases"]).items()},
        reason="First300 historical ledger rows are not balanced; preserve failed verifier; corrected main uses75 seeded source permutations")
    write(dest/"reuse_receipt.json",provenance)
    for stage in ("initial","pilot"):
        original=Path(str(source)+"_"+stage+"_receipts");assert (original/"exit_status.txt").read_text().strip()=="0"
        receipt=Path(str(dest)+"_"+stage+"_receipts");assert not receipt.exists();receipt.mkdir()
        values=dict(execution_commit=commit,source_execution_commit=(original/"execution_commit.txt").read_text().strip(),
            mode="REUSED_HASH_VERIFIED_QUALIFICATION_FROM_"+str(source),started_utc=start,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase="completed",exit_status="0")
        for name,value in values.items():(receipt/(name+".txt")).write_text(value+"\n")
        write(receipt/"provenance.json",dict(source_receipt=str(original),source_receipt_files={p.name:sha(p) for p in original.iterdir() if p.is_file()},reuse_receipt=ref(dest/"reuse_receipt.json")))
    print("R32_REUSED_QUALIFICATIONS",len(copied),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--dest",required=True);a=p.parse_args();run(a.source,a.dest)
