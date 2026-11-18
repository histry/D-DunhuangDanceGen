"""Independent numerical re-execution, official FK and frozen lineage verification."""
import argparse,datetime
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.reperformance_r44.replay import digest

PARENTS=('bridge_r38_01','stabilize_r39_01','contract_r40_01','paired_event_r41_01','source_context_r42_01','music_reperformance_r43_01','sdmr_event_rank_01','reperformance_r44_01','retrieval_audit_r44_01','phase_r45_01','review_r45_01','bridge_audit_r46_01','source_coverage_r46_01','event_coverage_r47_01','manifold_bridge_r47_01','music_phase_r47_01')

def official(main,out):
    import torch
    from dunhuang.contracts import sixd
    from dunhuang.research.bridge_r38.study import runtime
    from dunhuang.research.coverage_r34.common import forbid_optimizers
    forbid_optimizers();rt=runtime(read('src/dunhuang/research/bridge_r38/config.json'));root=Path(main);meta=read(root/'result.json');z=np.load(root/'arrays.npz');maximum=0.;count=0
    try:
        for r in meta['records']:
            for method in r['methods']:
                R=z[r['key']+'_'+method+'_R'];x=np.zeros((256,139),np.float32);x[:,7:]=sixd(R,row=True).reshape(256,132)
                with torch.no_grad():P=rt.f(torch.tensor(x[None],device='cuda'),rt.s)[0,:,:22].cpu().numpy()
                P=P-P[:,:1] # Official FK includes rest pelvis J[0]; saved contract is pelvis-relative.
                error=float(abs(P-z[r['key']+'_'+method+'_P']).max());assert error<2e-6;maximum=max(maximum,error);count+=1
        rt.check()
    finally:rt.close()
    write(out,dict(official_fk_executed=True,trajectories=count,maximum_error_m=maximum,root_scope='ZERO_ROOT_AND_EXPLICIT_OFFICIAL_PELVIS_ORIGIN_REMOVAL_FOR_BODY_RELATIVE_CHECK',physical_qualified=False))

def verify(out):
    G=Path('outputs/bridge_dynamics_r48_01');parents=[]
    for root in PARENTS:
        path=Path('outputs')/root;manifest=path/'artifacts_manifest.json';files=read(manifest)['files']
        for f in files:assert (path/f['path']).stat().st_size==f['bytes'] and digest(path/f['path'])==f['sha256'],str(path/f['path'])
        parents.append(dict(root=str(path),files=len(files),manifest_sha256=digest(manifest)))
    packs=[]
    for name in ('main','controls'):
        first=G/name;second=G/(name+'_replay');assert read(first/'result.json')==read(second/'result.json');a=np.load(first/'arrays.npz');b=np.load(second/'arrays.npz');assert set(a.files)==set(b.files)
        for key in a.files:assert np.array_equal(a[key],b[key],equal_nan=True),key
        packs.append(dict(name=name,arrays=len(a.files),exact_scientific_arrays=True));a.close();b.close()
    a=np.load(G/'main/arrays.npz');b=np.load(G/'controls/arrays.npz')
    for key in a.files:assert np.array_equal(a[key],b[key],equal_nan=True),key
    assert read('outputs/event_feasibility_r48_01/result.json')==read('outputs/event_feasibility_r48_replay/result.json')
    assert read('outputs/phase_discrimination_r48_01/result.json')==read('outputs/phase_discrimination_r48_replay/result.json')
    assert read('outputs/r48_publication/root_contact_audit.json')==read('outputs/r48_publication/root_contact_audit_replay.json')
    for name in ('main','controls'):assert read(G/(name+'_official_fk.json'))==read(G/(name+'_official_fk_replay.json'))
    receipt=read(G/'receipts/main_browser_acceptance.json');assert receipt['passed'] and receipt['protocol_sha256']==digest(G/'viewer/protocol.json');assert datetime.datetime.fromisoformat(receipt['observed_at_utc']).timestamp()<(G/'controls/result.json').stat().st_mtime
    for folder in (G/'viewer',G/'videos',G/'viewer_controls',Path('outputs/event_feasibility_r48_01/viewer')):
        for f in read(folder/'protocol.json')['files']:assert (folder/f['name']).stat().st_size==f['bytes'] and digest(folder/f['name'])==f['sha256']
    phase=read('outputs/phase_discrimination_r48_01/result.json');assert not phase['training_admitted'] and phase['optimizer_steps']==0 and phase['test_reads']==0;assert read('outputs/review_r45_01/evaluation/result.json')['raw_review_count']==0
    write(out,dict(all_scientific_values_exact=True,geometric_solver_reexecuted=True,neural_optimizer_calls=0,frozen_neural_inference_replayed=True,parent_files=sum(r['files'] for r in parents),parents=parents,packs=packs,official_fk=[read(G/(name+'_official_fk.json')) for name in ('main','controls')],first_effect_before_controls=True,whole_song_generations=0,raw_human_reviews=0,can_route=False))
    print('R48_INDEPENDENT_OK',sum(r['files'] for r in parents),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--official-main');a=p.parse_args();official(a.official_main,a.out) if a.official_main else verify(a.out)
