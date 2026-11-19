"""Frozen read-only parents and deterministic scientific archives."""
from pathlib import Path
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read, write, sha, identity, relative_fk, rotation_sample, Library
from dunhuang.research.warp_coupling_r53.common import save_arrays, array_hash

ROOT = Path('outputs/phase_integrity_r60_01')
HERE = Path(__file__).parent
CFG = read(HERE/'protocol.json')
PARENTS = ['event_coverage_r47_01', 'event_feasibility_r48_01',
           'strong_baseline_r52_01', 'warp_coupling_r53_01', 'execution_planning_r58_01',
           'paired_benchmark_r59_01']

def fresh(path):
    p = Path(path)
    if p.exists(): raise ValueError('Fresh R60 output required: '+str(p))
    p.mkdir(parents=True)
    return p

def binding():
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.html')})

def seal_parents():
    rows = []
    for name in PARENTS:
        root = Path('outputs')/name
        for p in sorted(root.rglob('*')):
            if p.is_file(): rows.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
    assert rows
    return rows

def check_seal(rows):
    for r in rows: assert sha(r['path']) == r['sha256'], r['path']
    return len(rows)

def proof_interval(interval, component):
    a,b = map(int,interval); lo,hi = map(int,component)
    if not lo <= a < b <= hi: raise ValueError('Cannot cross accepted support or invent frames')
    return a,b

def crossings(interval, protected):
    a,b = interval
    return [dict(span=[int(x),int(y)], sides=[s for s,t in [('left',a),('right',b)] if x<t<y])
            for x,y in protected if x<a<y or x<b<y]

def phase_slice(phases, interval):
    a,b = interval
    return [p for p in phases if max(a,p['evidence_span'][0]) < min(b,p['evidence_span'][1])]
