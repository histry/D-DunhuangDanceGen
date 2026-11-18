"""Audio-only planning and separate waveform/FK evaluation proxies."""
import numpy as np
from scipy.signal import find_peaks
from dunhuang.research.complete_r36.core import sha,relative_fk
from dunhuang.research.reperformance_r44.motion import decode

_ready=False
def extract(path,cfg):
    global _ready
    if not _ready:
        from dunhuang.research.song_transfer.music import runtime
        from dunhuang.research.complete_r36.core import read
        runtime(read('inputs/clock_assets_v2.json'));_ready=True
    import librosa
    y,sr=librosa.load(path,sr=cfg['sample_rate']);hop=cfg['hop_length'];flux=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop);_,beats=librosa.beat.beat_track(onset_envelope=flux,sr=sr,hop_length=hop,trim=False);scale=float(np.quantile(flux,.95));onset=np.clip(flux/max(scale,1e-10),0,1);ticks=np.arange(len(flux));beat=np.exp(-.5*(np.min(abs(ticks[:,None]-np.asarray(beats)[None]),axis=1)/2)**2) if len(beats) else np.zeros(len(flux));rms=[];transient=[]
    for i in range(0,len(y),hop):
        chunk=y[i:i+hop];rms.append(float(np.sqrt(np.mean(chunk**2))));transient.append(float(np.sqrt(np.mean(np.diff(chunk)**2))) if len(chunk)>1 else 0.)
    def peaks(x):return find_peaks(x,distance=8,prominence=max(1e-6,.15*float(np.std(x))))[0].tolist()
    r44peaks=find_peaks(rms,distance=5,prominence=max(1e-5,.15*float(np.std(rms))))[0].tolist()
    return dict(onset=onset.tolist(),beat_strength=beat.tolist(),beats=np.asarray(beats).tolist(),valid=bool(scale>1e-8),rms_accent_frames=r44peaks,waveform_transient_frames=peaks(transient),audio_sha256=sha(path),librosa=librosa.__version__,sample_rate=sr,hop=hop,planning='SPECTRAL_FLUX_PLUS_BEAT; NO_EVALUATION_PEAKS_USED',evaluation='RMS_PEAK_AND_WAVEFORM_FIRST_DIFFERENCE_ENERGY; PROXIES_NOT_SEMANTIC_TRUTH')

def timing(raw,signal,J,parents,cfg):
    from dunhuang.research.reperformance_r44.phase import source_phases
    R,_=decode(raw);speedpeaks=[];accpeaks=[]
    for a,b in cfg['core_intervals']:
        speedpeaks.extend(p['frame'] for p in source_phases(R,a,b,cfg)[0]);P=relative_fk(R,J,parents)[a:b,[20,21]];acc=np.r_[0.,0.,np.linalg.norm(np.diff(P,n=2,axis=0)*900,axis=-1).mean(-1)];ids,_=find_peaks(acc,distance=12,prominence=max(.1,.15*float(np.std(acc))));accpeaks.extend(int(a+i) for i in ids if 8<i<len(acc)-9)
    def error(left,right):return float(np.min(abs(np.array(left)[:,None]-np.array(right)[None]),axis=1).mean()/30) if len(left) and len(right) else None
    return dict(rms_timing_seconds=error(speedpeaks,signal['rms_accent_frames']),wrist_acceleration_timing_seconds=error(accpeaks,signal['waveform_transient_frames']),speed_peak_count=len(speedpeaks),wrist_acceleration_peak_count=len(accpeaks),proxy_truth='NOT_HUMAN_OR_SEMANTIC_EVENT_CORRESPONDENCE')
