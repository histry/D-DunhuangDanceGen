"""Export frozen scalar role-clock evidence, never rotations or playback coordinates."""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.research.role_alignment_q01.common import ROOT,CFG,read,write,sha,fresh,verify

def run():
    independent=read(ROOT/'receipts/independent.json');assert independent['status']=='PASS'
    verify(ROOT/'preview');verify(ROOT/'controls');out=fresh('reports/paper1_role_alignment_release')
    for phase,names in [('preview',['registration','budget','decisions','manifest']),('controls',['decisions','execution_summary','sequential_equivalence','paired_results','paired_summary','qualification','manifest'])]:
        for name in names:write(out/(phase+'_'+name+'.json'),read(ROOT/phase/(name+'.json')))
    candidates=read(ROOT/'preview/candidates.json');public=[]
    for row in candidates:
        c=row['connection'];fields=['constructible','passes_original_screen','boundary_key','cache_hit','actual_geometry_calls','bridge_sha256','inherited','core_exact','metrics']
        public.append(dict(row,connection={k:c[k] for k in fields if k in c}))
    write(out/'candidates.json',public)
    catalogs=read(ROOT/'preview/motion_catalogs.json')+read(ROOT/'controls/paired_motion_catalogs.json')
    write(out/'catalog_summary.json',[dict(source_uid=c['source_uid'],events=len(c['events']),supported=sum(e['supported'] for e in c['events']),role_counts={role:sum(e['role']==role for e in c['events']) for role in CFG['role_map']},semantic_accuracy='NOT_EVALUABLE') for c in catalogs])
    write(out/'audio_events.json',read(ROOT/'audio/events.json'))
    for name in ['browser_acceptance','independent','audio_replay']:write(out/(name+'.json'),read(ROOT/'receipts'/(name+'.json')))
    write(out/'protocol.json',CFG)
    launch=Path('outputs/paper1_role_alignment_launch');exits={n:int((launch/(n+'.exit')).read_text()) for n in ['preview_initial','preview','audio','controls','replay','tests']}
    assert exits['preview_initial']==1 and all(v==0 for n,v in exits.items() if n!='preview_initial')
    testlog=(launch/'tests.log').read_text();assert re.search(r'Ran 23 tests',testlog) and '\nOK' in testlog
    old=ROOT/'preview_initial_asset_failure';unchanged={n:sha(ROOT/'preview'/n)==sha(old/n) for n in ['arrays.npz','candidates.json','budget.json','decisions.json']};assert all(unchanged.values())
    witness=[r for r in candidates if 'GuardedRateWitness' in r['labels']];assert all(r['bound_targets_pass'] and r['protected_first_last_four_exact'] for r in witness)
    write(out/'runtime.json',dict(status='PASS',scientific_commit='4a7913a7195897b8fd095f7b7f7ff766e77f6213',initial_commit='6038d2599b49adc4a1d72de9c0c231a8e732b3f1',exit_status=exits,initial_failure='VIEWER_AUDIO_ASSET_DIRECTORY_ONLY',initial_failure_log_sha256=sha(launch/'preview_initial.log'),asset_fix_scientific_outputs_unchanged=unchanged,contract_tests=23,candidate_positions=len(candidates),unique_core_rotation_hashes=len({r['rotation_sha256'] for r in candidates}),guarded_witness_positions=len(witness),all_guarded_witnesses_targets_and_edges_pass=True,geometry_budget_per_scientific_execution=read(ROOT/'preview/budget.json'),scientific_execution_attempts=3,geometry_budget_note='INITIAL_PREVIEW_PLUS_ASSET_RERUN_PLUS_FRESH_REPLAY; EACH_ATTEMPT_6_G1_SOLVES_68_MEASURED_REQUESTS; NOT_6_TOTAL_LIFETIME_CALLS',training_steps=0,new_planner_runs=0,independent_role_records=0,semantic_accuracy='NOT_EVALUABLE',artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE',can_route=False))
    write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}));print('ROLE_PUBLIC_JSON',len(list(out.iterdir())),sum(p.stat().st_size for p in out.iterdir()))
if __name__=='__main__':run()
