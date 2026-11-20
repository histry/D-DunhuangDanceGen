"""R69 fixed-source baseline comparison; separate from production routing."""
import hashlib,json,subprocess
from pathlib import Path
from dunhuang.research.sdmr_warp_rank.common import binding as parent_binding

HERE=Path(__file__).parent
ROOT=Path('outputs/retiming_baselines_r69_01')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_bytes((json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def fresh(path):
 p=Path(path)
 if p.exists():raise ValueError('Fresh R69 output required')
 p.mkdir(parents=True);return p
CFG=read(HERE/'protocol.json')
def binding():
 old=parent_binding();deps=old['dependencies']
 for name in ['src/dunhuang/research/sdmr_warp_rank/common.py','src/dunhuang/research/sdmr_warp_rank/connectivity.py','src/dunhuang/research/local_timing_r68/common.py','src/dunhuang/research/local_timing_r68/warp.py','src/dunhuang/research/local_timing_r68/study.py','src/dunhuang/research/local_timing_r68/viewer.py','scripts/retiming_baselines_r69.py']:
  deps[name]=sha(name)
 return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.html')},dependencies=deps)
def verify_manifest(folder,expected=None):
 p=Path(folder).resolve();m=p/'manifest.json'
 if expected is not None and sha(m)!=expected:raise ValueError('Parent manifest identity changed')
 d=read(m)
 for name,h in d['files'].items():
  f=(p/name).resolve()
  if not f.is_relative_to(p) or not f.is_file() or sha(f)!=h:raise ValueError('Sealed parent bytes changed: '+name)
 return d
def seal(folder):
 p=Path(folder)
 # Runtime is deliberately separate from the deterministic science file map.
 files={f.relative_to(p).as_posix():sha(f) for f in sorted(p.rglob('*')) if f.is_file() and f.name!='runtime.json'}
 write(p/'manifest.json',dict(binding=binding(),files=files,runtime_excluded='WALL_TIME_DIAGNOSTIC_NOT_SCIENTIFIC_SELECTION'))
