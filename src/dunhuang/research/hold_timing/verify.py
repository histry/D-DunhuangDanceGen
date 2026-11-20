"""Fresh complete scientific replay; no timing, browser or human truth substitution."""
from pathlib import Path
from .study import ROOT,run_audit,binding
from .controls import run as run_controls
from dunhuang.research.local_timing_r68.common import read,write,sha

def run():
 results={}
 for phase,runner in [('audit',run_audit),('controls',run_controls)]:
  original=ROOT/phase;manifest=read(original/'manifest.json');dest=ROOT/(phase+'_replay')
  runner(dest)
  differences=[n for n,h in manifest['files'].items() if sha(dest/n)!=h]
  results[phase]=dict(exact=not differences,differences=differences,manifest_sha256=sha(original/'manifest.json'))
  write(dest/'independent.json',dict(fresh_process=True,**results[phase]));assert not differences,(phase,differences)
 write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,results=results,training_steps=0,optimizer_calls=0,human_records=0,quality='NOT_EVALUABLE',can_route=False))
 print('HOLD_COMPLETE_REPLAY_PASS',flush=True)

if __name__=='__main__':run()
