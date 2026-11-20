"""R66 isolated IO and frozen dependency bindings."""
import hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).parent
ROOT=Path('outputs/sdmr_warp_rank_01')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def fresh(p):
 p=Path(p)
 if p.exists():raise ValueError('Fresh R66 directory required: '+str(p))
 p.mkdir(parents=True);return p
CFG=read(HERE/'protocol.json')
def binding():
 deps=['src/dunhuang/research/timing_r64/music.py','src/dunhuang/research/paired_event_r41/events.py','src/dunhuang/research/manifold_bridge_r47/config.json','src/dunhuang/research/music_structure_r56/music.py','src/dunhuang/research/music_structure_r56/protocol.json','src/dunhuang/planning/planner.py','src/dunhuang/research/complete_r36/music.py','src/dunhuang/research/complete_r36/core.py','src/dunhuang/research/warp_coupling_r53/common.py','src/dunhuang/contracts.py','src/dunhuang/io.py','src/dunhuang/research/budget_r63/population.py','src/dunhuang/research/budget_r63/engine.py','src/dunhuang/research/timing_validity_r66/engine.py','src/dunhuang/research/warp_coupling_r53/warp_lattice.py','src/dunhuang/research/event_alignment_r49/common.py','src/dunhuang/research/paired_event_r41/model.py','src/dunhuang/research/transition_r37/content.py','src/dunhuang/research/duration_r62/cross_source.py','src/dunhuang/research/duration_r62/protocol.json','src/dunhuang/research/manifold_bridge_r47/geometry.py','src/dunhuang/research/bridge_dynamics_r48/study.py','src/dunhuang/research/edit_support_r61/fixed_gap.py','src/dunhuang/research/rhythm_r66/metrics.py','src/dunhuang/research/rhythm_r66/population_b.py','src/dunhuang/research/paired_benchmark_r59/data.py','src/dunhuang/research/source_tasks/common.py']
 return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.html')},dependencies={p:sha(p) for p in deps})
