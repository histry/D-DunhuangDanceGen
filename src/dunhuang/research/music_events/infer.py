"""Frozen official models; full waveforms, model observations, no Planner writes."""
from pathlib import Path
import json,os,sys,time,subprocess,random
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks
from .catalog import event,agreement,scale
from .compat import install
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash
ROOT=Path('outputs/music_events_01').resolve();ASSETS=ROOT/'assets';CFG=read(Path(__file__).with_name('protocol.json'))
os.environ.setdefault('TORCH_HOME',str(ASSETS/'torch'))

def binding():return {n:sha(Path(__file__).with_name(n)) for n in ['infer.py','catalog.py','compat.py','protocol.json']}

def assets():
    import torch
    from huggingface_hub import model_info,hf_hub_download
    manifest=ASSETS/'model_manifest.json'
    if manifest.exists():
        d=read(manifest)
        for r in d['files']:assert sha(r['path'])==r['sha256']
        return d
    hf_rev=model_info('taejunkim/allinone').sha;files=[]
    # Official loader filenames, frozen repository checkout.
    import ast
    tree=ast.parse((ASSETS/'all-in-one/src/allin1/models/loaders.py').read_text())
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAME_TO_FILE' for t in n.targets))
    for name in names.values():
        p=Path(hf_hub_download('taejunkim/allinone',name,revision=hf_rev,cache_dir=ASSETS/'hf'))
        files.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    checkpoint=ASSETS/'beat_this-final0.ckpt'
    if not checkpoint.exists():torch.hub.download_url_to_file('https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt',str(checkpoint))
    files.append(dict(path=str(checkpoint),sha256=sha(checkpoint),bytes=checkpoint.stat().st_size))
    d=dict(hf_repo='taejunkim/allinone',hf_revision=hf_rev,files=files,beat_this_checkpoint=str(checkpoint));write(manifest,d);return d

def acoustic(path):
    import librosa
    from dunhuang.research.music_structure_r56.music import analyze
    old,meta=analyze(path);y,sr=librosa.load(path,sr=22050,mono=True);hop=220
    onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
    t=librosa.frames_to_time(np.arange(len(onset)),sr=sr,hop_length=hop)
    clock=np.arange(meta['frames'])/30
    onset=scale(np.interp(clock,t,onset));flux=scale(old['accent_curve']);rise=scale(np.maximum(np.diff(old['rms'],prepend=old['rms'][0]),0))
    accent=(onset+flux+rise)/3
    peaks=find_peaks(accent,prominence=.1,distance=6)[0]
    return dict(onset=onset,flux=flux,rms_rise=rise,accent=accent,chroma_novelty=old['held_boundary'],energy=old['rms'],accent_frames=peaks,chroma_frames=old['proxy_boundaries']),meta

def analyze(path,out,registration,run_demucs=True):
    import torch
    from beat_this.inference import Audio2Frames
    from beat_this.model.postprocessor import Postprocessor
    path=Path(path).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    assert sha(path)==registration['audio_sha256'];y,sr=sf.read(path,always_2d=True,dtype='float32');samples=len(y);duration=samples/sr
    torch.manual_seed(CFG['seed']);np.random.seed(CFG['seed']);random.seed(CFG['seed']);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    modelassets=assets();start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    tracker=Audio2Frames(modelassets['beat_this_checkpoint'],device='cuda',float16=False)
    logits=tracker(y.mean(1),sr);beats,downbeats=Postprocessor(type='minimal',fps=50)(*logits)
    activ=dict(beat_this_beat=logits[0].sigmoid().cpu().numpy(),beat_this_downbeat=logits[1].sigmoid().cpu().numpy())
    del tracker;torch.cuda.empty_cache();install()
    # Pin official HF checkpoint revision; disable network during model loading.
    import allin1
    import allin1.models.loaders as loader
    from huggingface_hub import hf_hub_download
    loader.hf_hub_download=lambda repo_id,filename,cache_dir=None: hf_hub_download(repo_id,filename,revision=modelassets['hf_revision'],cache_dir=ASSETS/'hf',local_files_only=True)
    demix=out/'demix';spec=out/'spec'
    if run_demucs:
        subprocess.run([sys.executable,'-m','demucs.separate','--out',str(demix),'--name','htdemucs','--device','cuda','--shifts','0',str(path)],check=True)
    result=allin1.analyze(str(path),model='harmonix-all',device='cuda',include_activations=True,demix_dir=demix,spec_dir=spec,keep_byproducts=True,multiprocess=False)
    activ.update({'allin1_'+k:v for k,v in result.activations.items()})
    ac,meta=acoustic(path);activ.update({'acoustic_'+k:v for k,v in ac.items()});rows=[]
    def add(times,kind,source):
        for i,t in enumerate(times):
            key=source+'_'+kind;curve=activ.get(key);rate=50 if source=='beat_this' else 100
            probability=None if curve is None or np.asarray(curve).ndim!=1 else float(curve[min(len(curve)-1,max(0,int(np.rint(t*rate))))])
            rows.append(event(t,kind,source,samples,sr,i,raw_model_probability=probability,probability_is_calibrated_accuracy=False))
    add(beats,'beat','beat_this');add(downbeats,'downbeat','beat_this');add(result.beats,'beat','allin1');add(result.downbeats,'downbeat','allin1')
    segments=[dict(start=float(s.start),end=float(s.end),model_label=s.label,semantic_truth='UNKNOWN_NOT_DUNHUANG_FUNCTION_TRUTH') for s in result.segments]
    interior=[s['start'] for s in segments if 0<s['start']<duration and s['model_label'] not in ['start','end']]
    add(interior,'section_boundary','allin1');add(ac['accent_frames']/30,'accent_candidate','acoustic');add(ac['chroma_frames']/30,'chroma_boundary_candidate','acoustic')
    phrase=[t for t in ac['chroma_frames']/30 if interior and min(abs(t-s) for s in interior)<=.5 and len(result.downbeats) and min(abs(t-s) for s in result.downbeats)<=.5]
    add(phrase,'phrase_boundary_candidate','multi_cue')
    legacy=registration.get('legacy_clock')
    if legacy:
        for key,kind in [('beat_seconds','beat'),('accent_seconds','accent_candidate')]:add(legacy[key],kind,'R64')
        add(np.asarray(legacy['phrase_proxy_frames'])/30,'phrase_proxy','R64');add(np.asarray(legacy['section_proxy_frames'])/30,'section_proxy','R64')
    comparison=dict(beat_this_vs_allin1_beat=agreement(beats,result.beats),beat_this_vs_allin1_downbeat=agreement(downbeats,result.downbeats))
    if legacy:comparison['beat_this_vs_R64']=agreement(beats,legacy['beat_seconds']);comparison['allin1_vs_R64']=agreement(result.beats,legacy['beat_seconds'])
    evidence=dict(schema='MusicEventCatalogV1',audio_path=str(path),audio_sha256=sha(path),sample_rate=sr,audio_samples=samples,duration_seconds=duration,frames=(samples*30+sr-1)//sr,fps=30,
        events=rows,segments=segments,comparison=comparison,phrase_status='UNVALIDATED_CANDIDATES' if phrase else 'UNKNOWN_NO_MULTICUE_BOUNDARY',
        independent_annotations=None,accuracy='NOT_EVALUABLE',human_reviewed=False,can_route=False,model_manifest_sha256=sha(ASSETS/'model_manifest.json'),binding=binding())
    save_arrays(out/'activations.npz',activ);write(out/'catalog.json',evidence)
    write(out/'runtime.json',dict(seconds=time.perf_counter()-start,parent_peak_allocated_bytes=torch.cuda.max_memory_allocated(),demucs_subprocess_peak_not_included=True))
    write(out/'demix_manifest.json',dict(files={str(p.relative_to(out)):sha(p) for p in sorted(demix.rglob('*.wav'))}))
    return evidence

def main(replay=False):
    clocks=read('outputs/timing_r64_01/main/music_clocks.json');out=ROOT/('replay' if replay else 'main');out.mkdir(exist_ok=False)
    if replay:assert read(ROOT/'main/manifest.json')['binding']==binding()
    write(out/'registration.json',dict(protocol=CFG,binding=binding(),audio_sources=[dict(key=k,path=v['audio_path'],sha256=v['audio_sha256']) for k,v in clocks.items()]))
    for k,c in sorted(clocks.items()):
        analyze(c['audio_path'],out/k,dict(audio_sha256=c['audio_sha256'],legacy_clock=c));print('MUSIC_SONG_COMPLETE',k,flush=True)
    weight_files={str(p.relative_to(ASSETS)):sha(p) for p in sorted((ASSETS/'torch').rglob('*')) if p.is_file()}
    write(out/'additional_model_weights.json',dict(files=weight_files,scope='OFFICIAL_DEMUCS_TORCH_CACHE; WEIGHTS_NOT_PUBLISHED'))
    names={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['runtime.json','manifest.json']}
    write(out/'manifest.json',dict(binding=binding(),files=names))
    if replay:
        old=read(ROOT/'main/manifest.json');diff=[n for n,h in old['files'].items() if sha(out/n)!=h]
        # Raw activations may be subject to GPU arithmetic: never hide these differences.
        write(out/'independent.json',dict(fresh_process=True,exact=not diff,differences=diff,source_manifest_sha256=sha(ROOT/'main/manifest.json')))
        if diff:raise ValueError('Independent full-model replay differs; inspect recorded differences')

if __name__=='__main__':main('--replay' in sys.argv)
