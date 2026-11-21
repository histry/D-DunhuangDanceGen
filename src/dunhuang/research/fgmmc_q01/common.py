from pathlib import Path
import json,hashlib
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash,identity
HERE=Path(__file__).parent;ROOT=Path('outputs/fgmmc_q01');CFG=json.loads((HERE/'protocol.json').read_text())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fresh(p):p=Path(p);p.mkdir(parents=True,exist_ok=False);return p
def binding():
 return dict(code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json']},dependencies={n:sha(Path('src/dunhuang')/n) for n in ['alignment/model.py','alignment/partial_path.py','planning/anchor_reachability.py','research/transition_r37/content.py','research/source_time_q01/study.py']})
def seal(out):write(out/'manifest.json',dict(binding=binding(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}))
def verify(out):
 m=read(out/'manifest.json');assert m['binding']==binding()
 for n,h in m['files'].items():assert sha(out/n)==h,n
 return m
