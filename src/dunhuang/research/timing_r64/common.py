from pathlib import Path
import subprocess
from dunhuang.research.budget_r63.common import read,write,sha,array_hash,save_arrays,relative_fk,check_seal
HERE=Path(__file__).parent
ROOT=Path('outputs/timing_r64_01')
CFG=read(HERE/'protocol.json')

def fresh(path):
    p=Path(path)
    if p.exists():raise ValueError('Fresh R64 output required: '+str(p))
    p.mkdir(parents=True);return p

def binding():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.html')})
