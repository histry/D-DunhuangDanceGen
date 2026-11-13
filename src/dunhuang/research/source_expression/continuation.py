"""Reuse hash-sealed successful phases from an interrupted media preparation run."""
import argparse,subprocess
from pathlib import Path
from dunhuang.io import sha,write
from dunhuang.data.sources import ref
from .common import read,fresh,commit
PARTS=("prepared","diagnosis","probe","music")
def validate(lineage):
    if lineage["schema"]!="r6_verified_stage_continuation_v1" or lineage["reuse_stages"]!=list(PARTS):raise ValueError("Registered reuse stages differ")
    source=Path(lineage["source_root"]);receipt=Path(lineage["source_receipt"])
    if (receipt/"exit_status.txt").read_text().strip()!="1" or lineage["required_original_exit_status"]!=1:raise ValueError("Preserved failed launcher receipt required")
    original=lineage["original_execution_commit"]
    if (receipt/"execution_commit.txt").read_text().strip()!=original:raise ValueError("Original scientific execution differs")
    required={k+"/"+n for k in PARTS for n in ("protocol.json","result.json")}|{k+"_replay/replay.json" for k in PARTS}|{k+"/arrays.npz" for k in ("diagnosis","probe")}
    if set(lineage["artifacts"])!=required:raise ValueError("Incomplete reusable phase seals")
    for name,digest in lineage["artifacts"].items():
        if sha(source/name)!=digest:raise ValueError("Saved successful phase changed")
    for k in PARTS:
        if read(source/k/"protocol.json")["code_commit"]!=original or read(source/(k+"_replay")/"replay.json")["optimizer_calls"]:raise ValueError("Reusable phase was not freshly replayed")
    kernel=["src/dunhuang/research/source_expression/"+n+".py" for n in ("common","prepare","diagnose","probe","losses","music")]+["configs/source_expression_r6_v1.json"]
    if subprocess.check_output(["git","diff","--name-only",original,"HEAD","--",*kernel],text=True).strip():raise ValueError("Scientific kernel/config changed during media continuation")
    return source,receipt
def stage(lineage_file,out):
    lineage=read(lineage_file);source,receipt=validate(lineage);out=fresh(out)
    for k in PARTS:
        for suffix in ("","_replay"):(out/(k+suffix)).symlink_to((source/(k+suffix)).resolve(),target_is_directory=True)
    write(out/"lineage.json",dict(config=ref(lineage_file),execution_commit=commit(),source_receipt_sha256={f.name:sha(f) for f in receipt.iterdir() if f.is_file()},
        previous_status=1,reused_scientific_stages=list(PARTS),rerun="review media only; no new selection or optimization"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--lineage",required=True);p.add_argument("--out",required=True);a=p.parse_args();stage(a.lineage,a.out)
