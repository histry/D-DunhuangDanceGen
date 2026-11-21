from pathlib import Path
from dunhuang.research.source_time_q01.common import read,write,sha,prepare,fresh,verify as old_verify,ROOT as OLD
HERE=Path(__file__).parent;ROOT=Path('outputs/paper1_reachability_01');CFG=read(HERE/'protocol.json')
def binding():
    deps=['planning/anchor_reachability.py','planning/feasible_time_map.py','research/source_time_q01/study.py','research/source_time_q01/common.py','research/paper1_auto_eval/experiment.py','research/paper1_auto_eval/music.py']
    base=HERE.parents[1];return dict(code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ['.py','.json']},dependencies={n:sha(base/n) for n in deps})
def parents():
    result={}
    for phase,key in [('preview','source_time_preview_manifest_sha256'),('controls','source_time_controls_manifest_sha256')]:
        m=old_verify(OLD/phase);assert sha(OLD/phase/'manifest.json')==CFG[key];result[phase]=sha(OLD/phase/'manifest.json')
    p=Path('reports/paper1_source_time_release/manifest.json');assert sha(p)==CFG['source_time_public_manifest_sha256'];result['public']=sha(p);return result
def seal(out):write(out/'manifest.json',dict(binding=binding(),parents=parents(),files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'}))
def verify(out):
    m=read(Path(out)/'manifest.json');assert m['binding']==binding() and m['parents']==parents()
    for n,h in m['files'].items():assert sha(Path(out)/n)==h,n
    return m
