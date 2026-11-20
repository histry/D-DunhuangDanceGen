"""Scalar-only release of a preregistered adapted baseline, not quality rankings."""
from pathlib import Path
from collections import Counter
import itertools,json,hashlib
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.gtw_kinematic_baseline.study import ROOT
from dunhuang.research.gtw_kinematic_baseline.reviews import summarize

def main():
    out=Path('reports/gtw_baseline_release');out.mkdir(exist_ok=False)
    for n in ['preview','preview_verify','view','source','controls','controls_verify']:
        assert (ROOT/'receipts'/(n+'_exit.txt')).read_text().strip()=='0'
    for phase in ['preview','controls']:
        independent=read(ROOT/(phase+'_replay')/'independent.json');assert independent['exact'] and independent['differences']==[]
        for name in ['registration.json','manifest.json','summary.json','selections.json','costs.json','records.json','runtime.json']:
            write(out/phase/name,read(ROOT/phase/name))
        write(out/(phase+'_independent.json'),independent)
    inventory=read(ROOT/'source_audit/inventory.json');write(out/'source_inventory.json',inventory)
    human=summarize([],read(ROOT/'preview/admin/map.json'));write(out/'human_state.json',human)
    selected=read(ROOT/'controls/selections.json');rows=read(ROOT/'controls/records.json');agreements=[]
    for protocol in ['A','B']:
        units=sorted({r['unit'] for r in selected if r['protocol']==protocol});eligible=[]
        for unit in units:
            own=[r for r in selected if r['unit']==unit and r['protocol']==protocol]
            assert len({r['candidate_pool_sha256'] for r in own})==1
            if all(r['selected_id'] is not None for r in own):eligible.append(own)
        for a,b in itertools.combinations(['GTWLocal','GTWExtended','GTWContext'],2):
            pairs=[(next(r for r in g if r['method']==a),next(r for r in g if r['method']==b)) for g in eligible]
            agreements.append(dict(protocol=protocol,methods=[a,b],common_executions=len(pairs),different_selected_ids=sum(x['selected_id']!=y['selected_id'] for x,y in pairs),
                groups_with_choice=sum(x['eligible_count']>1 for x,y in pairs),interpretation='CHOICE_DIFFERENCE_NOT_QUALITY_GAIN'))
    actual=[r for r in rows if r['status']=='EXECUTED'];fixed=[r for r in actual if r['method'] not in ['Uniform','E1W']]
    q=dict(protocol_summary=read(ROOT/'controls/summary.json'),agreements=agreements,displayed_executions=len(actual),
        actual_screen_passed=sum(r['connection']['passes_original_screen'] for r in actual),
        selected_strict_internal_guards_exact=all(r['protected_edges_exact'] for r in fixed),
        physical_retiming_2006='NOT_REPRODUCED_MISSING_REQUIRED_OBSERVATIONS',original_GTW_DP='NOT_REPRODUCED_FINITE_CANDIDATE_ADAPTATION',
        new_source_generalization='NOT_TESTED_ALL_SOURCE_AND_MUSIC_HISTORICALLY_EXPOSED',human_records=0,phase_labels=0,
        context_predictive_gain='NOT_EVALUABLE',naturalness='NOT_EVALUABLE',Native24='NOT_EVALUABLE',new_training_steps=0,R74='NOT_STARTED_G1_G2_G3_NOT_ESTABLISHED')
    write(out/'qualification.json',q)
    write(out/'preview_browser_acceptance.json',read(ROOT/'receipts/preview_browser_acceptance.json'))
    receipt={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted((ROOT/'receipts').iterdir()) if p.is_file()}
    write(out/'receipt_hashes.json',receipt)
    write(out/'manifest.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},scope='SCALAR_ONLY_NO_TRAJECTORY_MEDIA_OR_BLIND_MAP'))
    print(json.dumps(q,ensure_ascii=False))

if __name__=='__main__':main()
