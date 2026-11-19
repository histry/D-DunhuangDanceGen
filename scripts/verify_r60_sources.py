"""Independent fresh source and actual preview replay without bridge outcome selection."""
import subprocess
import numpy as np
from dunhuang.research.phase_integrity_r60.common import ROOT,read,write,sha,fresh,relative_fk,check_seal,crossings
from dunhuang.research.phase_integrity_r60.population import Assets

def main():
    out=fresh(ROOT/'independent_sources');p=read(ROOT/'population/result.json');a=Assets().build()
    assert a.units==p['units'] and list(a.sources.values())==p['sources']
    for old in a.old['sources']:
        assert old['protected_spans']==a.sources[old['source_uid']]['protected_spans']
        assert old['accepted_components']==a.sources[old['source_uid']]['accepted_components']
    matched=[]
    for u in a.units:
        q=u['pair']
        if q['status']!='MATCHED':continue
        lo,hi=u['support_component'];s=a.sources[u['source_uid']]
        assert q['A4'][1]-q['A4'][0]==q['A5'][1]-q['A5'][0]
        for arm in ('A4','A5'):assert q[arm][0]-lo>=30 and hi-q[arm][1]>=30
        for side in q['common_hidden_seams']:
            cut=q['A5'][0 if side=='left' else 1]
            assert any(x<cut<y for x,y in s['protected_spans'])
        assert not crossings(q['A4'],s['protected_spans']) and q['common_hidden_seams']
        matched.append(u['unit_uid'])
    with np.load(ROOT/'main/arrays.npz') as z:
        cases=read(ROOT/'main/cases.json')
        for row in cases:
            i=row['case'];start=row['origin'];R=a.R[row['source_uid']][start:start+row['frames']]
            assert np.array_equal(R,z['case%d_R'%i])
            assert np.array_equal(relative_fk(R,a.J,a.parents),z['case%d_P'%i])
    old=np.load('outputs/strong_baseline_r52_01/plans/arrays.npz')
    assert np.array_equal(a.J,old['J']) and np.array_equal(a.parents,old['parents'])
    n=check_seal(p['parent_seal'])
    write(out/'result.json',dict(source_recordings_redecoded=len(a.sources),complete_population_exact=True,
        R47_protected_spans_and_quality_support_exact=True,same_skeleton_as_R52=True,
        matched_pairs_checked=len(matched),all_context_and_intervened_seams_verified=True,
        actual_preview_cases=len(cases),preview_R_and_FK_arrays_exact=True,parent_files_unchanged=n,
        raw_scope='FRESH_NATIVE22_AND_AUTHENTICATED_R59_FULL_RECORD_ARRAY_DECODING; NOT_NEW_SEMANTIC_ANNOTATION',
        script_sha256=sha(__file__),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),optimizer_calls=0))
if __name__=='__main__':main()
