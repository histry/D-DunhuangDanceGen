from pathlib import Path
import json,hashlib,subprocess
import numpy as np
from dunhuang.io import sha,checked
from dunhuang.research.complete_r36.core import read,write,relative_fk,identity
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash
ROOT=Path('outputs/paired_benchmark_r59_01')
CFG=read(Path(__file__).with_name('protocol.json'))
SCIENCE=('common.py','data.py','scoring.py','metrics.py','diagnostics.py','protocol.json')
def fresh(p):
 p=Path(p)
 if p.exists():raise ValueError('Fresh R59 output required: '+str(p))
 p.mkdir(parents=True);return p
def binding():
 deps=['src/dunhuang/research/paired_event_r41/model.py','src/dunhuang/research/transition_r37/content.py','src/dunhuang/research/bridge_r38/content.py','src/dunhuang/research/clock_residual_r55/common.py','src/dunhuang/research/strong_baseline_r52/viewer.html']
 return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),science={n:sha(Path(__file__).with_name(n)) for n in SCIENCE},dependencies={p:sha(p) for p in deps})
def seal_member(root,name):
 p=Path(root)/name;m=read(Path(root)/'artifacts_manifest.json')['files'];assert sha(p)==next(r['sha256'] for r in m if r['path']==name);return p
