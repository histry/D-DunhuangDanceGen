from pathlib import Path
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash

ROOT=Path('outputs/duration_r62_01')
HERE=Path(__file__).parent
CFG=read(HERE/'protocol.json')

def fresh(path):
    p=Path(path)
    if p.exists():raise ValueError('Fresh R62 output required: '+str(p))
    p.mkdir(parents=True);return p

def binding():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.html')})

def parent_seal():
    old=Path('outputs/edit_support_r61_01')
    files=read('reports/r61_release/scientific_files.json')['files']
    rows=[]
    for row in files:
        p=old/row['path'];assert sha(p)==row['sha256'],str(p)
        rows.append(dict(path=str(p),sha256=row['sha256']))
    for name in ('reports/r61_release/public_manifest.json','reports/r61_release/scientific_files.json',
                 'outputs/phase_integrity_r60_02/population/result.json','outputs/phase_integrity_r60_02/controls/arrays.npz'):
        rows.append(dict(path=name,sha256=sha(name)))
    return rows

def check_seal(rows):
    for r in rows:assert sha(r['path'])==r['sha256'],r['path']
    return len(rows)
