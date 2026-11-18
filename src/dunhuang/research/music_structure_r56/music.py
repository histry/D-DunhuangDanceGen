"""Audio-only structure proposals; separate proposal and evaluation feature families."""
import math
import soundfile as sf
from scipy.signal import stft,resample_poly,find_peaks
from scipy.ndimage import gaussian_filter1d
from .common import *

def unit_scale(x):
 x=np.asarray(x,float);q=float(np.quantile(x,.95));return np.clip(x/max(q,1e-12),0,1)

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def novelty(features):
 x=normalize(features);cs=np.vstack((np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)));n=len(x);scores=[]
 for radius in CFG['novelty_context_frames']:
  r=np.arange(n);lo=np.maximum(0,r-radius);hi=np.minimum(n,r+radius);left=(cs[r]-cs[lo])/np.maximum(r-lo,1)[:,None];right=(cs[hi]-cs[r])/np.maximum(hi-r,1)[:,None];v=np.sum((left-right)**2,axis=1);v[:radius]=0;v[-radius:]=0;v=gaussian_filter1d(v,CFG['novelty_smoothing_frames']);scores.append(v/max(float(v.max()),1e-12))
 value=np.mean(scores,axis=0);return value/max(float(value.max()),1e-12)

def analyze(path):
 wave,sr=sf.read(path,always_2d=True,dtype='float64');frames=int(np.ceil(len(wave)*30/sr));mono=wave.mean(axis=1);g=math.gcd(sr,CFG['audio_sr']);mono=resample_poly(mono,CFG['audio_sr']//g,sr//g);rate=CFG['audio_sr'];f,t,z=stft(mono,fs=rate,nperseg=CFG['stft_window'],noverlap=CFG['stft_window']-CFG['stft_hop'],boundary='zeros',padded=True);power=np.abs(z)**2;centers=np.geomspace(55,7600,50);bank=[]
 for a,b,c in zip(centers[:-2],centers[1:-1],centers[2:]):bank.append(np.maximum(0,np.minimum((f-a)/(b-a),(c-f)/(c-b))))
 spec=np.log1p(10000*(np.asarray(bank)@power)).T;chroma=np.zeros((len(t),12));valid=(f>=55)&(f<=5000);pitch=np.rint(69+12*np.log2(f[valid]/440)).astype(int)%12
 for k in range(12):chroma[:,k]=power[valid][pitch==k].sum(axis=0)
 chroma=np.log1p(chroma*10000);clock=np.arange(frames)/30
 spec=np.stack([np.interp(clock,t,x) for x in spec.T],axis=1);chroma=np.stack([np.interp(clock,t,x) for x in chroma.T],axis=1);rms=np.interp(clock,t,np.sqrt(power.sum(axis=0)));spectral_change=unit_scale(np.r_[0,np.maximum(np.diff(spec,axis=0),0).mean(axis=1)])
 proposal=novelty(spec);held=novelty(chroma);quiet=1-unit_scale(gaussian_filter1d(rms,15));reward=proposal+CFG['silence_weight']*quiet
 peaks=find_peaks(held,distance=CFG['proxy_boundary_distance_frames'],prominence=CFG['proxy_boundary_prominence'])[0];accents=find_peaks(spectral_change,distance=CFG['accent_distance_frames'],prominence=.1)[0]
 arrays=dict(spec=spec,chroma=chroma,rms=rms,proposal=proposal,held_boundary=held,boundary_reward=reward,accent_curve=spectral_change,proxy_boundaries=peaks.astype(np.int32),proxy_accents=accents.astype(np.int32))
 meta=dict(audio_sha256=sha(path),audio_samples=len(wave),sample_rate=sr,frames=frames,duration_seconds=len(wave)/sr,feature_version='SCIPY_STFT_LOG_FREQUENCY48_CHROMA12_V1',clock='EXACT_STFT_SECONDS_INTERPOLATED_TO_FRAME_CENTERS_30FPS',planner_boundary_features='LOG_FREQUENCY_NOVELTY_PLUS_SILENCE',evaluation_boundary_features='CHROMA_NOVELTY_AUTOMATIC_PROXY',planner_repetition_features='CHROMA_CORE_MEAN_STD',evaluation_repetition_features='LOG_FREQUENCY_CORE_MEAN_STD',phrase_truth='NOT_EVALUABLE_NO_INDEPENDENT_ANNOTATIONS',arrays={k:array_hash(v) for k,v in arrays.items()})
 return arrays,meta

def descriptors(features,spans):
 rows=[]
 for r in spans:
  if r['kind']=='core':
   a,b=r['interval'];v=features[a:b];rows.append(np.r_[v.mean(axis=0),v.std(axis=0)])
 x=np.asarray(rows);return (x-x.mean(axis=0))/np.maximum(x.std(axis=0),.01)
