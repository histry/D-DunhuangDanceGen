"""One retrained ablation, scheduled only after the full framework completes."""
import sys
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dunhuang.alignment.dataset import load
from dunhuang.alignment.train import fit
from dunhuang.io import write

full=Path(sys.argv[1]).resolve()
if not (full/'filename_labels.json').is_file() or not (full/'generation_5s/status.json').is_file():
    raise ValueError('Complete full framework first')
out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False)
rows,arrays,norm=load(ROOT/'inputs/sdmr_features/manifest.json')
arrays['music'][:,:,18:]=0
arrays['mm'][:,:,18:]=False
write(out/'protocol.json',dict(removed='acoustic logmel32, keep rhythm18 and observation masks',
    main_run=str(full),seed=20261111,max_retrieval_steps=5000,max_sync_steps=2000,
    selection='same validation early stopping rules as primary',primary_reselected=False))
_,result=fit(rows,arrays,norm,out,5000,2000)
write(out/'comparison.json',dict(full=json.loads((full/'result.json').read_text()),
    no_acoustic32=result,scope='one retrained feature ablation; identical grouped development regression roles'))
