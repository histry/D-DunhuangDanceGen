"""Publish statistics and evidence hashes only; no raw weights, audio or trajectories."""
import argparse
from pathlib import Path
from dunhuang.io import sha,write
from dunhuang.research.local_r23.common import read,commit
def run(root,receipt,out,media_receipt,media_dir,failed_media_receipts):
    root=Path(root);receipt=Path(receipt);out=Path(out);media_receipt=Path(media_receipt)
    assert not out.exists(),"Preserve existing publication";assert (receipt/"exit_status.txt").read_text().strip()=="0"
    assert (receipt/"phase.txt").read_text().strip()=="completed"
    assert (media_receipt/"exit_status.txt").read_text().strip()=="0" and (media_receipt/"phase.txt").read_text().strip()=="completed"
    stages=["samples","gradients","represent","external","physical_tracking","native"]
    summary={name:read(root/name/"result.json") for name in stages}
    summary.update(independent_checks=read(root/"independent_checks.json"),media=read(Path(media_dir)/"protocol.json"),
        code_commit=commit(),human_reviewed=False,operational_qualified=False,whole_song_generations=0)
    out.mkdir(parents=True);write(out/"r23_summary.json",summary);write(out/"r23_checks.json",summary["independent_checks"])
    evidence={str(p.relative_to(root)):sha(p) for p in root.rglob("*.json") if p.name in ("result.json","protocol.json","independent_checks.json")}
    write(out/"r23_receipt.json",dict(collection_commit=commit(),execution_commit=(receipt/"execution_commit.txt").read_text().strip(),
        evidence=evidence,summary_sha256=sha(out/"r23_summary.json"),receipts={str(folder):{p.name:sha(p) for p in folder.iterdir() if p.is_file()} for folder in [receipt,media_receipt]+[Path(x) for x in failed_media_receipts]},
        exit_status=0,whole_song_generations=0,optimizer_steps=0,temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        tracking_rollouts=0,operational_qualified=False))
    print("R23_COLLECTED",sha(out/"r23_summary.json"),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--receipts",required=True);p.add_argument("--out",required=True);p.add_argument("--media-receipts",required=True);p.add_argument("--media-dir",required=True);p.add_argument("--failed-media-receipts",nargs="*",default=[]);a=p.parse_args();run(a.root,a.receipts,a.out,a.media_receipts,a.media_dir,a.failed_media_receipts)
