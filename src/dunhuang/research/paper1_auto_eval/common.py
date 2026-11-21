"""Bindings for a read-only extension; Q01 and all baseline producers stay frozen."""
from pathlib import Path
from dunhuang.research.paper1_support.study import read,write,sha,uid,prepare

ROOT=Path('outputs/paper1_auto_eval_01');HERE=Path(__file__).parent;CFG=read(HERE/'protocol.json');PARENT=Path('outputs/paper1_support_01/controls')
def checked_parent():
    assert sha(PARENT/'manifest.json')==CFG['q01_controls_manifest'];m=read(PARENT/'manifest.json')
    for n,h in m['files'].items():assert sha(PARENT/n)==h,n
    return read(PARENT/'registration.json'),read(PARENT/'records.json')
def fresh(path):
    p=Path(path);p.mkdir(parents=True,exist_ok=False);return p
def seal(out):
    deps=['paper1_support/study.py','paper1_support/contracts.py','gtw_kinematic_baseline/features.py','gtw_kinematic_baseline/reviews.py','gtw_kinematic_baseline/viewer.html','music_events/catalog.py','paired_event_r41/events.py','timing_validity_r66/engine.py','warp_coupling_r53/warp_lattice.py']
    base=HERE.parent
    write(out/'manifest.json',dict(code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json']},dependencies={n:sha(base/n) for n in deps},files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}))
