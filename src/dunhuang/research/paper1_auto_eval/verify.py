"""Fresh scientific replay; immutable parents and exact output arrays are mandatory."""
from .common import ROOT,HERE,read,write,sha
from .experiment import preview
from .controls import run

def verify():
    results={}
    for phase,runner in [('preview',preview),('controls',run)]:
        parent=ROOT/phase;m=read(parent/'manifest.json')
        for n,h in m['files'].items():assert sha(parent/n)==h,('original changed',n)
        for n,h in m['dependencies'].items():assert sha(HERE.parent/n)==h,('dependency changed',n)
        for n,h in m['code'].items():assert sha(HERE/n)==h,('producer changed',n)
        dest=ROOT/(phase+'_replay');runner(dest);differences=[n for n,h in m['files'].items() if sha(dest/n)!=h]
        assert not differences,(phase,differences);results[phase]=dict(exact=True,differences=differences,manifest_sha256=sha(parent/'manifest.json'))
    write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,results=results,parent_outputs_unchanged=True,training_steps=0,human_records=0,can_route=False))
    print('AUTO_EVAL_REPLAY_PASS',flush=True)
if __name__=='__main__':verify()
