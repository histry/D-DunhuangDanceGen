"""Release scalar evidence only; no motion arrays, media, private blind maps or identities."""
from pathlib import Path
import hashlib,json,subprocess
from collections import Counter

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes((json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def seal(out):write(out/'manifest.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},scope='SCALAR_METADATA_ONLY'))
def main():
    a=Path('outputs/judge_repair_r72_01');b=Path('outputs/compensation_r73_01');outa=Path('reports/r72_release');outb=Path('reports/r73_release')
    assert not outa.exists() and not outb.exists()
    for root,names in [(a,['dev','replay','diagnostic_media','human_media']),(b,['preview_v2','preview_verify','view','controls_v2','controls_verify'])]:
        for name in names:assert (root/'receipts'/(name+'_exit.txt')).read_text().strip()=='0'
    attempts=[json.loads(x) for x in (a/'generation_attempts.jsonl').read_text().splitlines()];assert len(attempts)<=48
    dev=read(a/'dev/summary.json');replay=read(a/'replay/independent.json');assert replay['fresh_process']
    rows=[read(p) for p in sorted((a/'dev').glob('s*.json')) if p.name!='summary.json']
    assert len(rows)==dev['calls']
    if not dev['development_2fps_gate_passed']:assert not (a/'validation').exists()
    reg=read(a/'registration.json');human=read('outputs/human_r72_01/benchmark/review_state.json')
    for name in ['registration.json','construction.json','jobs.json']:
        write(outa/name,read(a/name))
    write(outa/'generation_attempts.json',attempts)
    for phase in ['dev','replay','validation']:
        if (a/phase).exists():
            for p in sorted((a/phase).glob('*.json')):write(outa/phase/p.name,read(p))
    write(outa/'human_benchmark_state.json',human)
    auto=dict(development=dev,total_generation_attempts=len(attempts),hard_budget=48,
        raw_stop_reasons=dict(Counter(r.get('stop_reason','ERROR') for r in rows)),
        maximum_generated_tokens=max(r.get('generated_tokens',0) for r in rows),
        peak_allocated_bytes=max(r.get('peak_allocated_bytes',0) for r in rows),
        per_test=[dict(id=r['job']['job_id'],fps=r['job']['fps'],test=r['job']['test'],parsed=r.get('parsed'),expected=r['job']['expected'],stop=r.get('stop_reason'),tokens=r.get('generated_tokens')) for r in rows],
        fresh_process_replay=replay,validation='NOT_RUN_DEVELOPMENT_GATE_FAILED' if not dev['development_2fps_gate_passed'] else 'SEE_VALIDATION',
        automatic_judge_can_rank_quality=False,R69_bulk_inferences=0,human_agreement='NOT_EVALUABLE',valid_human_records=0,
        diagnostic_scope='SYNTHETIC_PERIODIC_POSE_ROTATIONS_NOT_ARTISTIC_QUALITY_TRUTH',R74='NOT_STARTED',new_training_steps=0)
    write(outa/'qualification.json',auto)
    preview=read(b/'preview/summary.json');controls=read(b/'controls/summary.json');groups=read(b/'controls/groups.json');records=read(b/'controls/records.json')
    for phase in ['preview','controls']:
        verify=read(b/(phase+'_replay')/'independent.json');assert verify['exact'] and verify['differences']==[]
        for name in ['summary.json','registration.json','groups.json','proposals.json','manifest.json']:
            write(outb/phase/name,read(b/phase/name))
        write(outb/(phase+'_independent.json'),verify)
    lookup={(r['case_key'],r['method']):r for r in records};comparisons=[]
    population=read(b/'controls/registration.json')['cases'];bykey={c['key']:c for c in population}
    matched=[g for g in groups if g['status']=='MATCHED_HIGH_LOW']
    for g in matched:
        key=g['case_key'];delta=g['delta'];h=lookup[key,f'High_{delta:+d}'];l=lookup[key,f'Low_{delta:+d}'];c=bykey[key]
        comparisons.append(dict(case_key=key,source_uid=c['source_uid'],source_interval=c['source_interval'],delta=delta,match=g['primary_match'],
            High=dict(core=h['core_dynamics'],pair=h.get('pair_dynamics'),dose=h['candidate']['dose'],activity=h['candidate']['activity'],actual_peak=h['candidate']['actual_peak_frame']),
            Low=dict(core=l['core_dynamics'],pair=l.get('pair_dynamics'),dose=l['candidate']['dose'],activity=l['candidate']['activity'],actual_peak=l['candidate']['actual_peak_frame'])))
    write(outb/'matched_comparisons.json',comparisons)
    scalar=[]
    for r in records:
        row={k:v for k,v in r.items() if k not in ['region_dynamics']}
        if 'region_dynamics' in r:
            d=r['region_dynamics'];row['attribution']={k:d[k] for k in ['footprint','stencils','geometry_retention','activity_normalized_oscillation']}
        scalar.append(row)
    write(outb/'records.json',scalar)
    actual=[r for r in records if r['method']!='E1W']
    q=dict(preview=preview,controls=controls,matched_source_intervals=len({(r['source_uid'],tuple(r['source_interval'])) for r in comparisons}),
        matched_by_dose={str(d):dict(Counter(g['status'] for g in groups if g['delta']==d)) for d in [-6,-3,3,6]},
        selected_nonbaseline_executions=len(actual),all_selected_bridges_byte_identical=all(r['bridge_identical_to_rebuilt_baseline'] is True for r in actual),
        all_selected_protected_clock_and_rotation_edges_exact=all(r['region_dynamics']['footprint']['protected_q_and_R_bytes_exact'] for r in actual),
        source_grouped_effect='DESCRIPTIVE_ONLY_REPEATED_DOSES_AND_MUSIC_NOT_INDEPENDENT',quality_winner=None,
        valid_human_records=0,naturalness_gain='NOT_EVALUABLE',semantic_phase_truth='UNKNOWN',Native24='NOT_EVALUABLE',
        R74='NOT_STARTED_INDEPENDENT_MECHANISM_EVIDENCE_MISSING',new_training_steps=0,
        correction='OUTPUT_ONLY_NUMPY_BOOL_SERIALIZATION; INITIAL_FAILED_ARCHIVE_RETAINED; CANDIDATES_AND_RULES_UNCHANGED')
    write(outb/'qualification.json',q)
    for root,out,names in [(a,outa,['dev','replay','diagnostic_media','human_media']),(b,outb,['initial_interruption','preview_v2','preview_verify','view','controls_failed_serialization','controls_v2','controls_verify'])]:
        receipt={}
        for n in names:
            for suffix in ['_exit.txt','.txt','.log']:
                p=root/'receipts'/(n+suffix)
                if p.exists():receipt[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size,exit_status=p.read_text().strip() if suffix=='_exit.txt' else None)
        write(out/'execution_receipts.json',receipt)
    write(outb/'preview_browser_acceptance.json',read(b/'receipts/preview_browser_acceptance.json'))
    seal(outa);seal(outb)
    print('R72_R73_SCALAR_RELEASE_READY',json.dumps(dict(calls=len(attempts),matched=len(matched),R74='NOT_STARTED')))

if __name__=='__main__':main()
