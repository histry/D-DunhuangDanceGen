from pathlib import Path
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk,identity
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash

ROOT=Path('outputs/edit_support_r61_01')
HERE=Path(__file__).parent
CFG=read(HERE/'protocol.json')
PARENT=Path('outputs/phase_integrity_r60_02')
def fresh(path):
    p=Path(path)
    if p.exists():raise ValueError('Fresh R61 path required: '+str(p))
    p.mkdir(parents=True);return p
def binding():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.html')})
def verify_parent(pop):
    rows=pop['parent_seal']
    for row in rows:assert sha(row['path'])==row['sha256'],row['path']
    return len(rows)
