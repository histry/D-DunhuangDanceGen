"""Canonical public scalars, never raw rotations, audio, reviewer identities or maps."""
from pathlib import Path
from .common import ROOT,HERE,read,write,sha

def run():
    assert read(ROOT/'receipts/independent.json')['status']=='PASS'
    out=Path('reports/paper1_auto_eval_release');out.mkdir(exist_ok=False)
    for phase in ['preview','controls']:
        m=read(ROOT/phase/'manifest.json')
        for n,h in m['files'].items():
            assert sha(ROOT/phase/n)==h
            if n.endswith('.json'):write(out/(phase+'_'+n),read(ROOT/phase/n))
        write(out/(phase+'_manifest.json'),m)
    for name in ['independent','human_status','preview_browser_acceptance']:write(out/(name+'.json'),read(ROOT/'receipts'/(name+'.json')))
    write(out/'protocol.json',read(HERE/'protocol.json'));write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print('AUTO_EVAL_PUBLIC_SCALARS_READY',flush=True)
if __name__=='__main__':run()
