"""Fresh CPU replay comparison, SO3 budgets, state FK and all metrics recomputed."""
import argparse,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.research.complete_r36.core import read,write,sha,Library,relative_fk
from .generate_contract import verify_plan
from .evaluate import decode,metrics,world_fk

def exact(a,b):
    x=np.load(a);y=np.load(b);assert set(x.files)==set(y.files)
    for k in x.files:assert np.array_equal(x[k],y[k]),k

def run(cfg,root,out):
    root=Path(root);checks={};summary=[]
    for name in ('planning','smoothing','generation'):
        if name!='planning':exact(root/name/'arrays.npz',root/(name+'_replay')/'arrays.npz')
        for p in (root/name).glob('*.json'):
            x=p.read_text().replace(str(root/name),'<RUN>');y=(root/(name+'_replay')/p.name).read_text().replace(str(root/(name+'_replay')),'<RUN>');assert x==y,p.name
        checks[name+'_fresh_process_exact_replay']=True
    lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog'])
    for song in read(root/'planning/result.json')['songs']:verify_plan(read(song['new_plan']),lib)
    checks['source_support_maps_no_padding']=True
    z=np.load(root/'generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];meta=read(root/'generation/result.json')
    for row in meta['records']:
        key=row['key'];reference=z[key+'_reference_R'];record=dict(key=key,new_plan_consumed=row['consumes_new_r37_plan'],methods={})
        record['methods']['Reference']=[metrics(reference,reference,np.zeros((256,3)),J,parents,[128],row['gap'])]
        G=z[key+'_Geometry_R'];record['methods']['Geometry']=[metrics(G,reference,np.zeros((256,3)),J,parents,[128],row['gap'])]
        for arm in ('R36Like','Joint'):
            stats=[]
            anchor=reference if arm=='R36Like' else G
            for raw,stored in zip(z[key+'_'+arm+'_139'],z[key+'_'+arm+'_22']):
                R,Root=decode(raw);P=world_fk(R,Root,J,parents);assert np.allclose(P,stored,atol=2e-5)
                error=Rotation.from_matrix((anchor.swapaxes(-1,-2)@R).reshape(-1,3,3)).magnitude().reshape(256,22)
                assert (error<=z[key+'_'+arm+'_budget'][:,None]+2e-5).all()
                stats.append(metrics(R,reference,Root,J,parents,[128],row['gap']))
            record['methods'][arm]=stats
            # Four-frame raw-x0 vs projected-x0 uses identical state and stencil.
            trace=z[key+'_'+arm+'_trace'];v=[]
            for step in trace:
                samples=[]
                for seed in step:
                    pair=[]
                    for raw4 in seed:
                        # Stored trace is NORMALIZED; body channels are R31 identity-normalized.
                        R,_=decode(raw4);P=relative_fk(R,J,parents)
                        pair.append(float(np.linalg.norm(np.diff(P,n=3,axis=0)*27000,axis=-1).mean()))
                    samples.append(pair)
                v.append(samples)
            record[arm+'_same_state_pre_post_boundary_jerk']=v
        summary.append(record)
    checks['saved_motion_fk_matches_viewer_tracks']=True;checks['budgets_satisfied']=True;checks['same_state_projection_diagnostics_saved']=True
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),checks=checks,generation=summary,
                optimizer_steps=0,training_reproduction_claim=False,whole_song_generations=0,operational_qualified=False,sdmr_can_route=False,native24='NOT_EVALUABLE_22_OF24')
    write(out,result)
    files=[]
    for name in ('planning','planning_replay','smoothing','smoothing_replay','generation','generation_replay'):
        for p in sorted((root/name).rglob('*')):
            if p.is_file():files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    write(root/'artifacts_manifest.json',dict(files=files,optimizer_steps=0,operational_qualified=False))
    print('R37_VERIFIED',len(files),checks,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(read('src/dunhuang/research/transition_r37/config.json'),a.root,a.out)
