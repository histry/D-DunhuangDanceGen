"""R68 isolated IO, execution identity and frozen imported dependencies."""
import hashlib
import json
from pathlib import Path
from dunhuang.research.sdmr_warp_rank.common import binding as parent_binding

HERE=Path(__file__).parent
ROOT=Path('outputs/local_timing_r68_01')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def fresh(p):
    p=Path(p)
    if p.exists():raise ValueError('Fresh R68 output directory required: '+str(p))
    p.mkdir(parents=True);return p
CFG=read(HERE/'protocol.json')
def binding():
    old=parent_binding();deps=old['dependencies']
    for name in ['src/dunhuang/research/sdmr_warp_rank/common.py','src/dunhuang/research/sdmr_warp_rank/connectivity.py','src/dunhuang/research/rhythm_r66/roster.json','scripts/local_timing_r68.py']:
        deps[name]=sha(name)
    return dict(commit=old['commit'],code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.html')},dependencies=deps)
def seal(out):
    p=Path(out);write(p/'manifest.json',dict(binding=binding(),files={f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file()}))
def verify_manifest(folder):
    p=Path(folder);m=read(p/'manifest.json')
    for name,h in m['files'].items():
        child=(p/name).resolve()
        if not child.is_relative_to(p.resolve()) or not child.is_file() or sha(child)!=h:raise ValueError('Frozen artifact changed: '+name)
    return m
