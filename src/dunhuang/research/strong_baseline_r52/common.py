from pathlib import Path
from dunhuang.research.complete_r36.core import read,write,relative_fk,sha
from dunhuang.research.reperformance_r44.replay import digest
ROOT=Path('outputs/strong_baseline_r52_01')

def fresh(p):
 p=Path(p)
 if p.exists():raise ValueError('Fresh R52 path required: '+str(p))
 p.mkdir(parents=True);return p
