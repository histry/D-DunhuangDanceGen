"""Effect-first phase candidate, then same-anchor and fixed-population controls."""
import argparse,subprocess,shutil,copy
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.reperformance_r44.archive import Arrays
from dunhuang.research.reperformance_r44.motion import retime,world,kinematics
from .clock import compile_clock,matched_maps
from .audio import extract,timing

def build(out,controls=False):
    cfg=read(Path(__file__).with_name('config.json'));parent=Path(cfg['parent']);out=Path(out)
    if out.exists():raise ValueError('Fresh R45 output')
    out.mkdir(parents=True);z=Arrays(parent/'main/arrays.npz');old=read(parent/'main/result.json');J=z['skeleton_J22'];parents=z['parents'];values=dict(skeleton_J22=J,parents=parents);rows=[];audio_cache={}
    for row in old['records']:
        key=row['key'];raw=z[key+'_Geometry_139'];src=parent/'main'/row['audio'];audio_name=key+'.wav';shutil.copyfile(src,out/audio_name)
        if str(src) not in audio_cache:audio_cache[str(src)]=extract(src,cfg)
        A=audio_cache[str(src)];q,records,banks=compile_clock(raw[0],J,parents,A,cfg);methods=dict(Geometry=raw.copy(),R44C2=z[key+'_C2_139'],Phase=np.stack([retime(x,q) for x in raw]),R43Core=z[key+'_R43Core_139']);clocks=dict(Geometry=np.arange(256,dtype=float),R44C2=z[key+'_C2_clock'],Phase=q,R43Core=np.arange(256,dtype=float));alternative=None;B=None;control_maps={}
        if controls:
            linear,c2=matched_maps(row['phase_maps']);np.testing.assert_allclose(c2,z[key+'_C2_clock'],atol=1e-10,rtol=0);reconstructed=np.stack([retime(x,c2) for x in raw]);assert np.array_equal(reconstructed,methods['R44C2']);alternative=key+'_B.wav';alt=Path('outputs/transition_r37_02/generation')/('old_failed.wav' if row['source'].get('target_music')=='new_plan' else 'new_plan.wav');shutil.copyfile(alt,out/alternative)
            if str(alt) not in audio_cache:audio_cache[str(alt)]=extract(alt,cfg)
            B=audio_cache[str(alt)];qb,rb,_=compile_clock(raw[0],J,parents,B,cfg,copy.deepcopy(banks));qn,rn,_=compile_clock(raw[0],J,parents,None,cfg,copy.deepcopy(banks));assert np.array_equal(qn,np.arange(256));methods.update(MatchedLinear=np.stack([retime(x,linear) for x in raw]),MatchedC2=reconstructed,WrongMusic=np.stack([retime(x,qb) for x in raw]),NoMusic=raw.copy(),FixedMapMusicB=methods['Phase'].copy());clocks.update(MatchedLinear=linear,MatchedC2=c2,WrongMusic=qb,NoMusic=qn,FixedMapMusicB=q.copy());control_maps=dict(WrongMusic=rb,NoMusic=rn);assert np.array_equal(methods['Phase'],methods['FixedMapMusicB'])
            for name,maps in control_maps.items():
                for a,b in zip(records,maps):assert [[x['target'],x['eligible']] for x in a['bank']]==[[x['target'],x['eligible']] for x in b['bank']]
        metrics={};curves={};protected=np.ones(256,bool)
        for a,b in cfg['core_intervals']:protected[a:b]=False
        for name,x in methods.items():
            xyz=np.stack([world(v,J,parents) for v in x]);values[key+'_'+name+'_139']=x;values[key+'_'+name+'_22']=xyz;values[key+'_'+name+'_clock']=clocks[name];m=[kinematics(v,J,parents) for v in x];metrics[name]={k:float(np.mean([v[k] for v in m])) for k in m[0]};metrics[name].update(timing(x[0],A,J,parents,cfg));metrics[name]['protected_exact']=bool(np.array_equal(x[:,protected],raw[:,protected]));curves[name]=np.r_[np.zeros(3),np.sqrt(np.mean(np.diff(xyz,n=3,axis=1)**2,axis=(0,2,3)))*27000].tolist()
            if name!='R43Core':assert metrics[name]['protected_exact']
        values[key+'_reference_R']=z[key+'_reference_R'];rows.append(dict(key=key,label=key,source=row['source'],methods=list(methods),audio=audio_name,alternative_audio=alternative,seeds=row['seeds'],music=A,alternative_music=B,phase_maps=records,matched_anchor_maps=row['phase_maps'],control_maps=control_maps,metrics=dict(methods=metrics,curves=curves),scope='R45 actual source-path retiming; fixed Geometry Bridge. Phase uses a music-independent source-feasible clock bank. Not learned music generation. Common22/world Root/contact limits inherited.'))
        print('R45_EFFECT',key,'phase_frames',int(np.sum(q!=np.arange(256))),'eligible',[sum(r['eligible'] for r in c['bank']) for c in records],flush=True)
    np.savez_compressed(out/'arrays.npz',**values);write(out/'result.json',dict(records=rows,config=cfg,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),parent_arrays_sha256=sha(parent/'main/arrays.npz'),optimizer_steps=0,training_steps=0,new_neural_samples=0,whole_song_generated=False,geometry_seed_duplicates_not_independent=True,operational_qualified=False));z.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');a=p.parse_args();build(a.out,a.controls)
