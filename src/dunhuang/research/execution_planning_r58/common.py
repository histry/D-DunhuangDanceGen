from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha,relative_fk,rotation_sample,multi_warp
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash,identity
ROOT=Path('outputs/execution_planning_r58_01')
PARENT=Path('outputs/music_structure_r56_02')
PREVIOUS=Path('outputs/tempo_attribution_r57_01')
CFG=read(Path(__file__).with_name('protocol.json'))
SCIENCE=('common.py','profiles.py','graph.py','study.py','recovery.py','protocol.json')
def fresh(path):
 p=Path(path)
 if p.exists():raise ValueError('Fresh R58 output required: '+str(p))
 p.mkdir(parents=True);return p
DEPENDENCIES=('src/dunhuang/research/geometry_recovery_r57/study.py','src/dunhuang/research/tempo_attribution_r57/common.py','src/dunhuang/research/music_structure_r56/planning.py','src/dunhuang/research/music_structure_r56/metrics.py','src/dunhuang/research/music_structure_r56/common.py','src/dunhuang/research/music_structure_r56/music.py')
def binding():return dict(dependencies={p:sha(p) for p in DEPENDENCIES},science={n:sha(Path(__file__).with_name(n)) for n in SCIENCE},parent_manifest_sha256=sha(PARENT/'artifacts_manifest.json'),r57_manifest_sha256=sha(PREVIOUS/'artifacts_manifest.json'),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
def execution_cost(ratios):
 terms=[]
 for order,value in ratios.items():
  if value['ratio'] is not None:terms.append(abs(float(np.log(max(value['ratio'],1e-8))))/int(order))
 return float(np.mean(terms)) if terms else 0.
def regions_for(nodes):
 rows=[]
 for k,n in enumerate(nodes):
  if k:
   a,b=nodes[k-1]['target'][1],n['target'][0]
   if b-a!=CFG['gap']:raise ValueError('Selected nodes have incompatible clocks')
   rows.append(dict(kind='transition',interval=[a,b]))
  rows.append(dict(kind='core',interval=n['target']))
 from dunhuang.research.whole_song_geometry.study import coverage
 coverage(rows,nodes[-1]['target'][1]);return rows
