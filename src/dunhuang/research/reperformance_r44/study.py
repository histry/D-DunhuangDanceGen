"""Frozen-pose musical clocks. Hidden targets are evaluation-only; no network sampling."""
import argparse, subprocess, shutil
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from dunhuang.research.complete_r36.core import read,write,sha
from .archive import Arrays
from .phase import maps,source_phases
from .motion import decode,retime,world,kinematics


def audio(path,cfg):
    # Use the already audited runtime and fail if its files or version drift.
    from dunhuang.research.song_transfer.music import runtime
    runtime(read('inputs/clock_assets_v2.json'))
    import librosa
    y,sr=librosa.load(path,sr=cfg['sample_rate']);hop=cfg['hop_length']
    onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
    _,beats=librosa.beat.beat_track(onset_envelope=onset,sr=sr,hop_length=hop,trim=False)
    # Independent evaluation proxy: raw waveform energy peaks, not beat detector labels.
    rms=np.array([np.sqrt(np.mean(y[t:t+hop]**2)) for t in range(0,len(y),hop)])
    accents,_=find_peaks(rms,distance=5,prominence=max(1e-5,float(np.std(rms))*.15))
    return dict(beats=np.asarray(beats,float).tolist(),accent_frames=accents.tolist(),sha256=sha(path),librosa=librosa.__version__,sampling_rate=sr,hop_length=hop)


def proxy(raw,accents,cfg):
    R,_=decode(raw);peaks=[]
    for a,b in cfg['core_intervals']:peaks.extend(p['frame'] for p in source_phases(R,a,b,cfg)[0])
    if not len(accents) or not peaks:return None
    return float(np.mean(np.min(abs(np.asarray(peaks)[:,None]-np.asarray(accents)[None]),axis=1))/30)


def build(out,controls=False):
    cfg=read(Path(__file__).with_name('config.json'));parent=Path(cfg['parent']);out=Path(out)
    if out.exists():raise ValueError('Fresh R44 output required')
    out.mkdir(parents=True);z=Arrays(parent/'main/arrays.npz');meta=read(parent/'main/result.json');J=z['skeleton_J22'];parents=z['parents'];values=dict(skeleton_J22=J,parents=parents);records=[];cache={}
    for row in meta['records']:
        key=row['key'];raw=z[key+'_Geometry_139'];R,_=decode(raw[0]);src=parent/'main'/row['audio'];name=key+'.wav';shutil.copyfile(src,out/name)
        if str(src) not in cache:cache[str(src)]=audio(src,cfg)
        A=cache[str(src)];linear,q,anchors=maps(R,A['beats'],cfg);methods={'Geometry':raw.copy(),'E1W':np.stack([retime(x,linear) for x in raw]),'C2':np.stack([retime(x,q) for x in raw]),'R43Core':z[key+'_Core_139']};clocks={'Geometry':np.arange(256,dtype=float),'E1W':linear,'C2':q,'R43Core':np.arange(256,dtype=float)}
        B=None
        if controls:
            alt=Path('outputs/transition_r37_02/generation')/('old_failed.wav' if row.get('target_music')=='new_plan' else 'new_plan.wav');B=audio(alt,cfg);_,qb,ab=maps(R,B['beats'],cfg);methods.update(WrongMapB=np.stack([retime(x,qb) for x in raw]),FixedMapMusicB=methods['C2'].copy(),NoMusicTime=raw.copy());clocks.update(WrongMapB=qb,FixedMapMusicB=q.copy(),NoMusicTime=np.arange(256,dtype=float));shutil.copyfile(alt,out/(key+'_B.wav'))
            assert np.array_equal(methods['C2'],methods['FixedMapMusicB'])
        scores={};curves={};protected=np.ones(256,bool)
        for a,b in cfg['core_intervals']:protected[a:b]=False
        for method,x in methods.items():
            xyz=np.stack([world(v,J,parents) for v in x]);values[key+'_'+method+'_139']=x;values[key+'_'+method+'_22']=xyz;values[key+'_'+method+'_clock']=clocks[method]
            metrics=[kinematics(v,J,parents) for v in x];scores[method]={k:float(np.mean([v[k] for v in metrics])) for k in metrics[0]};scores[method]['rms_accent_timing_proxy_seconds']=proxy(x[0],A['accent_frames'],cfg);scores[method]['protected_exact']=bool(np.array_equal(x[:,protected],raw[:,protected]));curves[method]=np.r_[np.zeros(3),np.sqrt(np.mean(np.diff(xyz,n=3,axis=1)**2,axis=(0,2,3)))*27000].tolist()
            if method!='R43Core':assert scores[method]['protected_exact']
        values[key+'_reference_R']=z[key+'_reference_R'];records.append(dict(key=key,label=key,methods=list(methods),seeds=row['seeds'],audio=name,alternative_audio=key+'_B.wav' if controls else None,source=row,phase_maps=anchors,music=A,alternative_music=B,metrics=dict(methods=scores,curves=curves),scope='Frozen source pose path; Core clock only; Geometry Bridge unchanged. R43Core is saved neural failure, not R44 generation. Root model prior; ChangE world Root/contact UNKNOWN.'))
        print('R44_RETIME',key,'changed',int(np.sum(q!=np.arange(256))),flush=True)
    np.savez_compressed(out/'arrays.npz',**values);write(out/'result.json',dict(records=records,config=cfg,parent_arrays_sha256=sha(parent/'main/arrays.npz'),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),training_steps=0,optimizer_steps=0,new_neural_samples=0,whole_song_generated=False,human_reviewed=False,operational_qualified=False,geometry_seed_duplicates_not_independent=True));z.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--controls',action='store_true');a=p.parse_args();build(a.out,a.controls)
