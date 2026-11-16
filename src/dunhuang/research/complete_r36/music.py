"""Audio-only multi-scale structure proposals. No paired labels or SDMR scores."""
import numpy as np
from scipy.signal import stft,find_peaks,resample_poly
from scipy.ndimage import gaussian_filter1d
import soundfile as sf

def analyze(path,frames,beats):
    x,sr=sf.read(path,dtype='float32',always_2d=True);x=x.mean(1)
    if not np.isfinite(x).all() or len(x)<sr:raise ValueError('Finite readable audio required')
    if sr!=22050:
        import math
        g=math.gcd(sr,22050);x=resample_poly(x,22050//g,sr//g);sr=22050
    f,t,z=stft(x,sr,nperseg=2048,noverlap=1536);power=abs(z)**2
    bins=np.geomspace(40,10000,33);bands=np.array([np.log1p(1e4*power[(f>=a)&(f<b)].mean(0)) for a,b in zip(bins[:-1],bins[1:])]).T
    midi=np.zeros(len(f));valid=f>20;midi[valid]=np.rint(69+12*np.log2(f[valid]/440))
    chroma=np.stack([power[(midi.astype(int)%12==k)&valid].sum(0) for k in range(12)],1)
    chroma/=np.maximum(np.linalg.norm(chroma,axis=1,keepdims=True),1e-8)
    sample=np.arange(frames)/30;feature=np.stack([np.interp(sample,t,bands[:,j]) for j in range(32)],1)
    h=np.stack([np.interp(sample,t,chroma[:,j]) for j in range(12)],1)
    novelty=gaussian_filter1d(np.r_[0,np.linalg.norm(np.diff(feature,axis=0),axis=1)],5)
    novelty/=max(float(novelty.max()),1e-8)
    beat=np.array(sorted(set([0]+[int(v) for v in beats if 0<v<frames]+[frames])),int)
    if len(beat)<12:beat=np.arange(0,frames,15);beat=np.unique(np.r_[beat,frames])
    sync=np.array([np.r_[feature[a:b].mean(0),h[a:b].mean(0)] for a,b in zip(beat[:-1],beat[1:])])
    sync=(sync-sync.mean(0))/np.maximum(sync.std(0),.05);sync/=np.maximum(np.linalg.norm(sync,axis=1,keepdims=True),1e-8)
    similarity=np.clip(sync@sync.T,-1,1);contrast=np.zeros(len(beat))
    for i in range(4,len(sync)-4):
        left=sync[i-4:i];right=sync[i:i+4]
        contrast[i]=max(0.,float((left@left.T).mean()+(right@right.T).mean()-2*(left@right.T).mean()))
    contrast/=max(float(contrast.max()),1e-8)
    peaks=find_peaks(contrast,distance=4,prominence=.1)[0]
    phrases=[dict(frame=int(beat[i]),strength=float(contrast[i]),status='ACOUSTIC_PHRASE_CANDIDATE') for i in peaks]
    # Repetition observations are descriptive; no cultural or semantic labels.
    repeats=[]
    for i in range(0,len(sync)-8,4):
        matches=[(float(np.mean([similarity[i+k,j+k] for k in range(8)])),j) for j in range(i+16,len(sync)-8,4)]
        if matches:
            value,j=max(matches)
            if value>.65:repeats.append(dict(first=[int(beat[i]),int(beat[i+8])],second=[int(beat[j]),int(beat[j+8])],score=value,scope='UNCALIBRATED_AUDIO_REPETITION'))
    energy=np.array([np.sqrt(np.mean(x[round(a*sr/30):max(round((a+1)*sr/30),round(a*sr/30)+1)]**2)) for a in range(frames)])
    if not np.isfinite(energy).all():raise ValueError('Audio duration does not cover registered timeline')
    ranks=np.argsort(np.argsort(energy)).astype(float)/max(frames-1,1)
    return dict(frames=frames,beats=beat.tolist(),phrases=phrases,repetitions=repeats,novelty=novelty.tolist(),energy_percentile=ranks.tolist(),
        structure_scope='AUDIO_ONLY_MULTI_SCALE_PROPOSALS_NOT_VERIFIED_PHRASES',sdmr_used=False)

def next_boundaries(a,music,cfg):
    N=music['frames'];beats=np.array(music['beats']);nextbeats=beats[beats>a]
    values={N}
    for k in cfg['beat_lengths']:
        if len(nextbeats)>=k:values.add(int(nextbeats[k-1]))
    values.update(p['frame'] for p in music['phrases'] if a+cfg['min_music_frames']<=p['frame']<=a+cfg['max_music_frames'])
    # Finite fallback avoids making an impossible terminal sliver; never truncate music.
    values.update((min(N,a+180),min(N,a+240),max(0,N-180)))
    return sorted(b for b in values if cfg['min_music_frames']<=b-a<=cfg['max_music_frames'])
