"""Root/body vector decomposition and non-executable counterfactual responsibility."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
from dunhuang.io import checked,write,sha
from dunhuang.contracts import matrices
from dunhuang.evaluation_benchmarks.native_shadow import LegacyBackend

def read(p):return json.loads(Path(p).read_text())

def decompose(x,fk):
    world=fk.astype(np.float64);root=x[:,4:7].astype(np.float64)
    relative=world-root[:,None]
    r=matrices(x[:,7:13]).astype(np.float64)
    body=np.einsum('tji,tkj->tki',r,relative)
    jw=np.diff(world,n=3,axis=0)*30**3
    jr=np.diff(root,n=3,axis=0)[:,None]*30**3
    jb=np.diff(relative,n=3,axis=0)*30**3
    internal=np.diff(body,n=3,axis=0)*30**3
    wr=float(np.mean(jw*jw));rr=float(np.mean(jr*jr));br=float(np.mean(jb*jb));cross=float(2*np.mean(jr*jb))
    error=float(abs(jw-jr-jb).max())
    if error>1e-6:raise ValueError('Root/body vector identity failed')
    def dist(a):
        v=np.linalg.norm(a,axis=-1)
        return dict(p95=float(np.percentile(v,95)),max=float(v.max()))
    feet=world[:,[7,8,10,11]]
    return dict(world_jerk=dist(jw),translation_root_jerk=dist(jr),root_relative_jerk=dist(jb),
        root_frame_internal_jerk=dist(internal),vector_identity_max_abs=error,
        squared_component_mean=dict(world=wr,root_translation=rr,root_relative=br,signed_cross_term=cross,
            reconstruction_error=abs(wr-rr-br-cross)),
        foot_world_speed_p95=float(np.percentile(np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0)*30,axis=-1),95)),
        root_relative_includes_root_orientation=True,root_frame_removes_orientation=True,
        energy_components_are_not_causal_percentages=True)

def run(assets,prior,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);backend=LegacyBackend(assets);prior=Path(prior)
    if read(prior/'legacy_identity.json')!=backend.identity:raise ValueError('Frozen Native identity differs')
    protocol=dict(schema='root_contact_responsibility_v1',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        native_protocol_sha256=sha(prior/'protocol.json'),native_identity=backend.identity['id'],
        counterfactuals_not_executable_candidates=True,optimizer_calls=0,new_generations=0,legacy_gate_unchanged=True)
    write(out/'protocol.json',protocol)
    count=0;rows=[]
    for row in map(json.loads,(prior/'train/windows.jsonl').read_text().splitlines()):
        x=np.load(checked(row['motion']));fk=backend.fk(x);value=decompose(x,fk)
        rows.append(dict(window_uid=row['window_uid'],recording_uid=row['recording_uid'],
            source_group=row['source_group'],motion=row['motion'],legacy=row['legacy'],analysis=value))
        count+=1
    write(out/'source_rows.json',rows)
    generated=read(prior/'generated/results.json');results=[]
    for i,g in enumerate(generated):
        x=np.load(checked(g['motion']));fk=np.load(checked(g['fk']))
        original=backend.gate(backend.audit(x,backend.cfg,precomputed_joints=fk))
        if original!=g['analysis']['legacy']:raise ValueError('Original Native replay differs')
        item=dict(name=g['name'],motion=g['motion'],legacy=original,original=decompose(x,fk),counterfactuals={})
        folder=out/('case_%03d'%i);folder.mkdir()
        for name in ('hold_translation','hold_root_rigid','hold_body_local'):
            copy=x.copy()
            if name in ('hold_translation','hold_root_rigid'):copy[:,4:7]=x[0,4:7]
            if name=='hold_root_rigid':copy[:,7:13]=x[0,7:13]
            if name=='hold_body_local':copy[:,13:]=x[0,13:]
            positions=backend.fk(copy);np.save(folder/(name+'.npy'),copy)
            item['counterfactuals'][name]=dict(analysis=decompose(copy,positions),
                legacy_engineering_observation=backend.gate(backend.audit(copy,backend.cfg,precomputed_joints=positions)),
                not_a_candidate=True,physical_executability_unestablished=True)
        results.append(item)
    write(out/'generated.json',results)
    def grouped(key):
        groups={}
        for r in rows:groups.setdefault(r['recording_uid'],[]).append(r['analysis'][key]['max'])
        return dict(recordings=len(groups),recording_balanced_mean=float(np.mean([np.mean(v) for v in groups.values()])),
            window_median=float(np.median([r['analysis'][key]['max'] for r in rows])))
    write(out/'summary.json',dict(status='COMPLETED_ROOT_BODY_RESPONSIBILITY',source_windows=count,
        recordings=len({r['recording_uid'] for r in rows}),generated_cases=len(results),
        measurements={k:grouped(k) for k in ('world_jerk','translation_root_jerk','root_relative_jerk','root_frame_internal_jerk')},
        independent_root_bottleneck_established=False,rootnet_added=False,
        counterfactuals_not_repairs=True,legacy_gate_unchanged=True,optimizer_calls=0,new_generations=0))
    print('ROOT_RESPONSIBILITY_COMPLETED',count,len(results),flush=True)
def main():
    p=argparse.ArgumentParser()
    for k in ('assets','prior','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run(read(a.assets),a.prior,a.out)
if __name__=='__main__':main()
