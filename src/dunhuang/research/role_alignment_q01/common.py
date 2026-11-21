from pathlib import Path
from dunhuang.research.reachability_q01.common import verify as old_verify,ROOT as PARENT
from dunhuang.research.source_time_q01.common import ROOT as SOURCE_TIME,prepare,read,write,sha,fresh
HERE=Path(__file__).parent;ROOT=Path('outputs/paper1_role_alignment_01');CFG=read(HERE/'protocol.json')
def binding():
    deps=['planning/anchor_reachability.py','research/paired_event_r41/events.py','research/manifold_bridge_r47/config.json','research/source_time_q01/study.py','research/paper1_auto_eval/experiment.py','research/music_events/infer.py','research/timing_validity_r66/engine.py']
    base=HERE.parents[1];return dict(code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json']},dependencies={n:sha(base/n) for n in deps})
def parents():
    result={}
    for phase in ['preview','controls']:old_verify(PARENT/phase);result[phase]=sha(PARENT/phase/'manifest.json')
    p=Path('reports/paper1_reachability_release/manifest.json');assert sha(p)==CFG['reachability_public_manifest_sha256'];result['public']=sha(p);return result
def seal(out):write(out/'manifest.json',dict(binding=binding(),parents=parents(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}))
def verify(out):
    m=read(Path(out)/'manifest.json');assert m['binding']==binding() and m['parents']==parents()
    for n,h in m['files'].items():assert sha(Path(out)/n)==h,n
    return m
