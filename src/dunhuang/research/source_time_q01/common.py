"""Thin integration binding; no change to source, retiming or evaluation baselines."""
from pathlib import Path
from dunhuang.research.paper1_auto_eval.common import read,write,sha,prepare,checked_parent
from dunhuang.research.paper1_auto_eval.q02_common import parents as Q01_parents
HERE=Path(__file__).parent;ROOT=Path('outputs/paper1_source_time_01');CFG=read(HERE/'protocol.json')

def binding():
    deps=['paper1_support/study.py','paper1_support/contracts.py','hold_timing/metrics.py','gtw_kinematic_baseline/features.py','complete_r36/core.py','duration_r62/cross_source.py','manifold_bridge_r47/geometry.py','paper1_auto_eval/experiment.py','paper1_auto_eval/music.py','paper1_auto_eval/observability.py','paper1_auto_eval/q02_protocol.json']
    return dict(code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json','.html']},dependencies={n:sha(HERE.parent/n) for n in deps})

def parents():
    p=Path('reports/paper1_auto_eval_release/preview_registration.json');assert sha(p)==CFG['preflight_sha256'];q=Path('reports/paper1_auto_eval_q02_release/main_manifest.json');assert sha(q)==CFG['Q02_manifest_sha256']
    m=read(q);out=Path('outputs/paper1_auto_eval_02/main')
    for n,h in m['files'].items():assert sha(out/n)==h,n
    return dict(Q01=Q01_parents(),Q02_manifest=sha(q),Q02_files=m['files'],preflight=sha(p))

def fresh(p):
    p=Path(p);p.mkdir(parents=True,exist_ok=False);return p

def seal(out):
    write(out/'manifest.json',dict(binding=binding(),parents=parents(),files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}))

def verify(out):
    m=read(Path(out)/'manifest.json');assert m['binding']==binding();assert m['parents']==parents()
    for n,h in m['files'].items():assert sha(Path(out)/n)==h,n
    return m
