import sys
import numpy as np
from .common import ROOT,CFG,read,write,sha,fresh,seal,verify,prepare,binding,parents
from . import execution,paired
from dunhuang.research.warp_coupling_r53.common import save_arrays

def preview():
    out=fresh(ROOT/'preview');ctx=prepare();cases,catalogs=execution.register(ctx);write(out/'registration.json',dict(protocol=CFG,binding=binding(),parents=parents(),cases=cases,paired=paired.register(),registration_precedes_execution=True));write(out/'motion_catalogs.json',catalogs);rows,arrays,budget=execution.run(cases,ctx);write(out/'candidates.json',rows);write(out/'budget.json',budget);save_arrays(out/'arrays.npz',arrays);write(out/'decisions.json',execution.choose(cases,rows,CFG['preview_methods']));seal(out)
    from .view import build
    build('preview');print('ROLE_PREVIEW_READY',len(rows),flush=True)

def controls(dest=None):
    verify(ROOT/'preview')
    if dest is None:
        r=read(ROOT/'receipts/browser_acceptance.json');assert r['status']=='PASS' and r['manifest_sha256']==sha(ROOT/'preview/manifest.json')
    out=fresh(dest or ROOT/'controls');reg=read(ROOT/'preview/registration.json');rows=read(ROOT/'preview/candidates.json');d=execution.choose(reg['cases'],rows,CFG['execution_methods']);write(out/'decisions.json',d);equivalence=[]
    for r in [r for r in d if r['method']=='RoleOrdered']:
        s=next(s for s in d if s['case_key']==r['case_key'] and s['method']=='SourceAwareSequential');assert r['selected_uid']==s['selected_uid'];equivalence.append(dict(case_key=r['case_key'],identical=True))
    summary=paired.evaluate(out);write(out/'execution_summary.json',[dict(method=m,registered_tasks=sum(r['method']==m for r in d),G1_passes=sum(r['method']==m and r['G1_pass'] for r in d),selected_nonuniform=sum(r['method']==m and r['nonuniform'] is True for r in d),bound_target_checks_pass=sum(r['method']==m and r['bound_targets_pass'] is True for r in d),artistic_quality='NOT_EVALUABLE') for m in CFG['execution_methods']]);write(out/'sequential_equivalence.json',equivalence);write(out/'qualification.json',dict(clock_gate=all(r['clock_gate_pass'] for r in summary),independent_role_records=0,semantic_gate='NOT_EVALUABLE',independent_new_music_certified=False,formal_routing=False,whole_song='NOT_STARTED',training_steps=0,can_route=False));seal(out);print('ROLE_CONTROLS_READY',flush=True)

def replay():
    verify(ROOT/'preview');reg=read(ROOT/'preview/registration.json');cases,cats=execution.register(prepare());assert cases==reg['cases'] and cats==read(ROOT/'preview/motion_catalogs.json') and paired.register()==reg['paired'];rows,arrays,budget=execution.run(cases,prepare());out=fresh(ROOT/'replay');save_arrays(out/'arrays.npz',arrays);assert sha(out/'arrays.npz')==sha(ROOT/'preview/arrays.npz');assert rows==read(ROOT/'preview/candidates.json') and budget==read(ROOT/'preview/budget.json');controls(ROOT/'controls_replay');m=verify(ROOT/'controls');diff=[n for n,h in m['files'].items() if sha(ROOT/'controls_replay'/n)!=h];assert not diff,diff;write(ROOT/'receipts/independent.json',dict(status='PASS',fresh_process=True,arrays_exact=True,candidate_ledger_exact=True,scalar_differences=diff,audio_replay=read(ROOT/'receipts/audio_replay.json'),parents_unchanged=parents(),training_steps=0,can_route=False));print('ROLE_REPLAY_PASS',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='preview':preview()
    elif sys.argv[1]=='controls':controls()
    elif sys.argv[1]=='replay':replay()
    else:raise ValueError(sys.argv[1])
