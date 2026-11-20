"""Post-run verification of sealed auxiliary inputs; never a before-run claim."""
from pathlib import Path
from .study import ROOT,CFG,STUDIES
from dunhuang.research.local_timing_r68.common import read,write,sha

def run():
 roots=[(name,Path('outputs')/item[0],CFG[name+'_manifest']) for name,item in STUDIES.items()]
 roots.append(('MusicEvents',Path('outputs/music_events_01/main'),'ccc3ace99ac5313135c95f84c9a732f8f50312c00036900115118ddab26dbf52'))
 checks=[]
 for label,folder,expected in roots:
  assert sha(folder/'manifest.json')==expected
  m=read(folder/'manifest.json');different=[n for n,h in m['files'].items() if sha(folder/n)!=h];assert not different,(label,different)
  checks.append(dict(input=label,manifest_sha256=expected,files=len(m['files']),differences=different))
 write(ROOT/'receipts/auxiliary_input_verification.json',dict(status='PASS',fresh_process=True,verification_timing='POST_RUN',checks=checks,source_or_model_mutated=False))
 print('SEALED_AUXILIARY_INPUTS_PASS',flush=True)

if __name__=='__main__':run()
