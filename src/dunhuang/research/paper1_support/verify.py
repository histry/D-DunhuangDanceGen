"""Fresh scientific replay, byte comparison, no optimizer or evaluation labels."""
from .study import ROOT,run,read,write,sha
def verify():
    results={}
    for phase in ['main','controls']:
        root=ROOT/phase;dest=ROOT/(phase+'_replay');run(phase,dest)
        m=read(root/'manifest.json');differences=[n for n,h in m['files'].items() if sha(dest/n)!=h]
        results[phase]=dict(exact=not differences,differences=differences,manifest_sha256=sha(root/'manifest.json'));assert not differences,results
    write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,results=results,optimizer_calls=0,training_steps=0,human_records=0,can_route=False))
    print('PAPER1_SUPPORT_REPLAY_PASS',flush=True)
if __name__=='__main__':verify()
