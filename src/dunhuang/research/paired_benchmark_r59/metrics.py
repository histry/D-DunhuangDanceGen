"""Clock target retrieval with exact expectation over score ties; UNKNOWN is not artistic-negative."""
import math
import numpy as np

def starts(lo,hi,n=150,stride=75):
 if hi-lo<n:return []
 return sorted(set(list(range(int(lo),int(hi)-n+1,stride))+[int(hi)-n]))
def candidate_pool(rows,q,kind='open',limit=64):
 a=rows[q];ids=[i for i,b in enumerate(rows) if a['dataset']==b['dataset'] and a['role']==b['role'] and (a['dataset']!='AIST++' or a['raw_motion_sha256']!=b['raw_motion_sha256'])]
 if kind=='same_music':ids=[i for i in ids if rows[i]['audio_sha256']==a['audio_sha256']]
 if kind=='activity_hard':ids=sorted(ids,key=lambda i:(abs(a['music_activity']-rows[i]['body_activity']),rows[i]['uid']))[:limit]
 return ids

def relation(a,b,iou=.6,gap=15):
 if a['dataset']!=b['dataset'] or a['role']!=b['role']:return 'UNKNOWN'
 if a['dataset']=='AIST++' and a['raw_motion_sha256']==b['raw_motion_sha256']:return 'UNKNOWN'
 A,B=a['audio_interval'],b['audio_interval'];inter=max(0,min(A[1],B[1])-max(A[0],B[0]));union=max(A[1],B[1])-min(A[0],B[0]);same=a['audio_sha256']==b['audio_sha256']
 if same and inter/max(union,1e-9)>=iou:return 'CLOCK_TARGET'
 if same and max(A[0],B[0])-min(A[1],B[1])>=gap:return 'CLOCK_OTHER_TIME'
 if not same:return 'OTHER_AUDIO_ASSOCIATION'
 return 'UNKNOWN'

def retrieval(scores,positive):
 x=np.asarray(scores);p=np.asarray(positive,bool)
 if x.ndim!=1 or x.shape!=p.shape or not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty score and target vectors required')
 if not p.any():return None
 # Fixed 1e-7 rounding bins make equality transitive and disclose numerical score resolution.
 x=np.rint(x.astype(float)/1e-7).astype(np.int64);first=max(x[p]);higher=int((x>first).sum());tie=x==first;n=int(tie.sum());positives=int(p[tie].sum())
 def recall(k):
  take=min(n,max(0,k-higher));miss=1.
  for j in range(take):miss*=max(0,n-positives-j)/(n-j)
  return 1-miss
 survive=1.;mrr=0.
 for r in range(1,n-positives+2):
  prob=survive*positives/(n-r+1);mrr+=prob/(higher+r);survive*=max(0,n-positives-r+1)/(n-r+1)
 return dict(recall_at1=recall(1),recall_at5=recall(5),mrr=mrr,chance_at1=float(p.mean()),target_count=int(p.sum()),pool_size=len(p))

def clock_metrics(scores,offsets,truth):
 x=np.asarray(scores);best=np.isclose(x,x.max(),rtol=0,atol=1e-7);o=np.asarray(offsets);return dict(accuracy=float(np.mean(o[best]==truth)),mae_frames=float(np.mean(abs(o[best]-truth))),tie_count=int(best.sum()))

def paired_ci(differences,seed=59059,draws=3000):
 x=np.asarray(differences,float)
 if not len(x):return dict(groups=0,mean=None,ci95=None)
 rng=np.random.default_rng(seed);means=np.mean(x[rng.integers(len(x),size=(draws,len(x)))],axis=1)
 return dict(groups=len(x),mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),unit='MUSIC_GROUP_MEAN_AFTER_RECORDING_BALANCE',scope='HISTORICALLY_EXPOSED_GROUP_POPULATION_NOT_NEW_MUSIC_GENERALIZATION')

def shared_support(motion_frames,audio_feature_frames,energy_frames):
 counts=[int(motion_frames),int(audio_feature_frames),int(energy_frames)]
 if min(counts)<1:raise ValueError('No genuine physical common prefix')
 n=min(counts)
 return dict(frames=n,unmatched_motion_tail=counts[0]-n,unmatched_audio_feature_tail=counts[1]-n,unmatched_audio_energy_tail=counts[2]-n,padding_frames=0,resampling=False)
