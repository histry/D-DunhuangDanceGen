"""Seal Train-only transfer arms; never select using exposed source generation."""
import argparse
from pathlib import Path
from .common import read,write,sha
def run(root):
    root=Path(root);r=read(root/"branch_summary.json");path=root/"release_admission.json";assert not path.exists()
    assert not r["Q2_eligible"],"Eligible Q2 requires the registered parity experiment before transfer"
    models=r["models_for_transfer"]
    write(path,dict(branch_summary_sha256=sha(root/"branch_summary.json"),models=models,Q2_steps=0,
        reason="Registered Q2 admission failed or unnecessary",transfer_selection_allowed=False,whole_song_generations=0,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
