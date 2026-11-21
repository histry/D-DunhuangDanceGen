"""Publish scalar source-time results after exact replay, never motion arrays."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.research.source_time_q01.common import ROOT,CFG,read,write,sha,fresh,verify
from dunhuang.research.source_time_q01 import baselines

def run():
    assert read(ROOT/'receipts/independent.json')['status']=='PASS';verify(ROOT/'preview');verify(ROOT/'controls');out=fresh('reports/paper1_source_time_release')
    for n in ['registration','geometry_budget','rejected','decisions','manifest']:
        write(out/('preview_'+n+'.json'),read(ROOT/'preview'/(n+'.json')))
    candidates=read(ROOT/'preview/candidates.json');public=[]
    for r in candidates:
        c=r['connection'];fields=['constructible','passes_original_screen','boundary_key','cache_hit','actual_geometry_calls','bridge_sha256','inherited','core_exact','metrics']
        public.append(dict(r,connection={k:c[k] for k in fields if k in c}))
    write(out/'candidates.json',public)
    for n in ['decisions','metrics','summary','exhaustive_equivalence','qualification','manifest']:write(out/('controls_'+n+'.json'),read(ROOT/'controls'/(n+'.json')))
    for n in ['independent','browser_acceptance','reference_G1_parity']:write(out/(n+'.json'),read(ROOT/'receipts'/(n+'.json')))
    reg=read(ROOT/'preview/registration.json');tasks=[]
    for c in reg['cases']:
        own=[r for r in candidates if r['case_key']==c['key']]
        for target in c['targets']:
            x=target['target_frame'];per=[]
            for source in [0,1]:
                rr=[r for r in own if r['source_order']==source];ok=[r for r in rr if baselines.basic(r)]
                per.append(dict(source_order=source,candidates=len(rr),G1_passes=sum(r['G1_pass'] for r in rr),support_passes=sum(r['moving_support_seconds']>=CFG['minimum_moving_support_seconds'] for r in rr),anchor_range_frames=[min(r['anchor_target_frame'] for r in rr),max(r['anchor_target_frame'] for r in rr)],screened_nearest_target_error_frames=None if x is None or not ok else min(baselines.error(r,x) for r in ok),fully_feasible_count=0 if x is None else sum(baselines.feasible(r,x) for r in rr),moving_support_seconds_range=[min(r['moving_support_seconds'] for r in rr),max(r['moving_support_seconds'] for r in rr)]))
            tasks.append(dict(case_key=c['key'],source_uid=c['source_uid'],target_fraction=target['fraction'],target_frame=x,source_results=per,scope='DESCRIPTIVE_FROZEN_POOL_ATTRIBUTION_NOT_CONTINUOUS_CAPACITY_BOUND'))
    write(out/'attribution.json',tasks);write(out/'protocol.json',CFG)
    launch=Path('outputs/paper1_source_time_launch');exits={name:int((launch/(name+'.exit')).read_text()) for name in ['preview','controls','view','view_fixed','reference','replay']};assert all(v==0 for k,v in exits.items() if k!='view');assert exits['view']==1
    write(out/'runtime.json',dict(status='PASS',scientific_commit='30f349e040330aba5c4efdbeaf557365d9b9bc7c',exit_status=exits,first_view_failure='SHELL_QUOTING_ONLY; SCIENTIFIC_CONTROLS_ALREADY_EXITED0',first_view_error_log_sha256=sha(launch/'view.log'),fixed_view_exit=0,contract_tests=15,source_arrays_and_scalar_replay='EXACT',reference_validation_calls=read(ROOT/'receipts/reference_G1_parity.json')['reference_G1_calls'],geometry_budget=read(ROOT/'preview/geometry_budget.json'),training_steps=0,can_route=False))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}));print('SOURCE_TIME_PUBLIC_JSON',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()))
if __name__=='__main__':run()
