"""Numerical replay is strict; separately bind irrelevant publication-label differences."""
import subprocess
from pathlib import Path
from dunhuang.research.complete_r36.core import read,write

SCIENCE_FILES=[
 'src/dunhuang/research/transition_r37/planning.py',
 'src/dunhuang/research/transition_r37/memory.py',
 'src/dunhuang/research/transition_r37/clock.py',
 'src/dunhuang/research/transition_r37/content.py',
 'src/dunhuang/research/complete_r36/core.py',
 'src/dunhuang/research/complete_r36/music.py',
 'src/dunhuang/planning/time_map.py','src/dunhuang/contracts.py']

def compare_planning(original,replayed,certificate=None):
    original=Path(original);replayed=Path(replayed);a=read(original/'result.json');b=read(replayed/'result.json');blobs={}
    for p in SCIENCE_FILES:
        old=subprocess.check_output(['git','rev-parse',a['code_commit']+':'+p],text=True).strip()
        new=subprocess.check_output(['git','rev-parse',b['code_commit']+':'+p],text=True).strip()
        if old!=new:raise ValueError('Scientific implementation differs: '+p)
        blobs[p]=old
    for p in original.glob('*.json'):
        x=p.read_text().replace(str(original),'<RUN>');y=(replayed/p.name).read_text().replace(str(replayed),'<RUN>')
        if p.name=='result.json':
            import json
            x=json.loads(x);y=json.loads(y);x.pop('code_commit');y.pop('code_commit')
        if x!=y:raise ValueError('Scientific planning replay differs: '+p.name)
    result=dict(status='ALL_SCIENTIFIC_JSON_EXACT; SOURCE_BLOBS_IDENTICAL',original_execution_label=a['code_commit'],replay_execution_label=b['code_commit'],
                ignored_fields=['result.json.code_commit'],scientific_git_blobs=blobs,optimizer_steps=0)
    if certificate:write(certificate,result)
    return result
