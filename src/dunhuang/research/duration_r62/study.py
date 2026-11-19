"""Effect-first local duration study, with exact replay and no planning promotion."""
import copy
import numpy as np
from .common import ROOT,CFG,binding,fresh,read,write,sha,parent_seal,check_seal

def run(out,controls=False,gate=True,render=True):
    if controls and gate:
        r=read(ROOT/'receipts/main_browser.json')
        assert r['passed'] and r['main_result_sha256']==sha(ROOT/'main/result.json')
    out=fresh(out)
    seals=parent_seal()
    from .continuous import run as continuous
    from .cross_source import run as cross
    c=continuous(out/'continuous',controls=controls,gate=False)
    x=cross(out/'cross_source',controls=controls,main=ROOT/'main/cross_source' if controls else None)
    assert len(c['records'])==12*len(CFG['durations'])
    result=dict(binding=binding(),controls=controls,protocol=CFG,
        continuous_summary=c['summary'],cross_summary=x.get('summary'),
        continuous_result_sha256=sha(out/'continuous/result.json'),cross_result_sha256=sha(out/'cross_source/result.json'),
        parent_seal=seals,parent_files_unchanged=check_seal(seals),
        G0='ALL_REGISTERED_SUPPORT_AND_FAILURES_REPORTED',
        G1='DESCRIPTIVE_DURATION_RESPONSE_NOT_FIXED_BUDGET_QUALITY_GAIN',
        G2='NO_INDEPENDENT_ADAPTIVE_SELECTION_EVALUATION; NOT_QUALIFIED',
        G3='CORE_TRANSITION_JOINT_PLANNING_NOT_RUN',
        training_steps=0,optimizer_calls=0,planner_runs=0,joint_budget_planning_runs=0,
        human_reviewed=False,verified_phase='UNKNOWN',Native24='NOT_EVALUABLE',can_route=False)
    write(out/'result.json',result)
    if render:
        from .view import build
        build(out,controls=controls)
    return result

def normalize(value):
    value=copy.deepcopy(value)
    if isinstance(value,dict) and 'binding' in value:value['binding']['commit']='EXACT_PACKAGE_CODE_HASHES_CHECKED'
    return value

def verify():
    out=fresh(ROOT/'independent');records=[]
    for mode in ('main','controls'):
        expected=ROOT/mode;old=read(expected/'result.json')
        assert binding()['code']==old['binding']['code']
        run(out/('rebuild_'+mode),controls=mode=='controls',gate=False,render=False)
        actual=out/('rebuild_'+mode)
        for sub in ('continuous','cross_source'):
            assert normalize(read(expected/sub/'result.json'))==normalize(read(actual/sub/'result.json'))
            assert sha(expected/sub/'arrays.npz')==sha(actual/sub/'arrays.npz')
        # Root child result SHA incorporates the original commit; all scientific values
        # and package hashes above are exact, so normalize only these linkage fields.
        rebuilt=read(actual/'result.json')
        for k in ('continuous_result_sha256','cross_result_sha256'):
            rebuilt[k]=old[k]
        assert normalize(rebuilt)==normalize(old)
        records.append(dict(mode=mode,scientific_values_exact=True,arrays_bytes_exact=True,
            source_result_sha256=sha(expected/'result.json'),source_commit=old['binding']['commit']))
    parents=read('outputs/phase_integrity_r60_02/population/result.json')['parent_seal']
    inherited=check_seal(parents)
    result=dict(records=records,replay_commit=binding()['commit'],frozen_parent_files_unchanged=inherited,
        fresh_source_decoding_and_curve_solves=True,optimizer_calls=0,training_steps=0,planner_runs=0,can_route=False)
    write(out/'result.json',result);print('R62_INDEPENDENT_EXACT',flush=True)
    return result
