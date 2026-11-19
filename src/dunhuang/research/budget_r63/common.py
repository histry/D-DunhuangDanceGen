"""R63 run binding and deterministic artifacts; historical baselines stay frozen."""
from pathlib import Path
import subprocess
from dunhuang.research.complete_r36.core import read, write, sha, relative_fk
from dunhuang.research.warp_coupling_r53.common import save_arrays, array_hash

ROOT = Path('outputs/budget_r63_01')
HERE = Path(__file__).parent
CFG = read(HERE/'protocol.json')

def fresh(path):
    p = Path(path)
    if p.exists():
        raise ValueError('Fresh R63 output required: '+str(p))
    p.mkdir(parents=True)
    return p

def binding():
    files = list(HERE.glob('*.py'))+list(HERE.glob('*.json'))+list(HERE.glob('*.html'))
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                code={p.name:sha(p) for p in sorted(files)})

def check_seal(rows):
    for row in rows:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Changed sealed input: '+row['path'])
    return len(rows)
