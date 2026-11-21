"""Metadata partition first; no historical model or normalizer is fitted on holdout."""
import re,collections
import numpy as np
from .common import *
from dunhuang.research.transition_r37.content import observations

def register():
 assert sha(CFG['data'])==CFG['data_sha256'];meta=read(CFG['data']);a=[s for s in meta['sources'] if s['dataset']=='AIST++' and s['role']=='train'];key=lambda x:__import__('hashlib').sha256((CFG['split_salt']+x).encode()).hexdigest();genre=lambda s:s['recording_uid'].split(':')[1].split('_')[0];dancer=lambda s:re.search(r'_d(\d+)_',s['recording_uid']).group(1)
 genres=sorted({genre(s) for s in a},key=key);dancers=sorted({dancer(s) for s in a},key=key);gr={v:('train' if i<6 else 'dev' if i<8 else 'holdout') for i,v in enumerate(genres)};dr={v:('train' if i<20 else 'dev' if i<25 else 'holdout') for i,v in enumerate(dancers)};allrows=[]
 for s in a:
  if gr[genre(s)]==dr[dancer(s)] and s['frames']>=180:allrows.append(dict(s,split=gr[genre(s)],genre_uid=genre(s),dancer_uid=dancer(s)))
 roles=collections.defaultdict(set)
 for s in allrows:
  if s.get('choreography_uid') not in [None,'UNKNOWN']:roles[s['choreography_uid']].add(s['split'])
 bad={k for k,v in roles.items() if len(v)>1};allrows=[s for s in allrows if s.get('choreography_uid') not in bad];sources=[]
 for split in ['train','dev','holdout']:
  own=[s for s in allrows if s['split']==split]
  for group in sorted({s['group'] for s in own}):sources+=sorted([s for s in own if s['group']==group],key=lambda s:s['source_uid'])[:CFG['sources_per_music']]
 for s in sorted(meta['sources'],key=lambda s:s['source_uid']):
  if s['dataset']=='FineDance':sources.append(dict(s,split='fine_diagnostic',genre_uid='UNKNOWN',dancer_uid='UNKNOWN_UNVERIFIED'))
 summary={r:dict(sources=sum(s['split']==r for s in sources),music_groups=len({s['group'] for s in sources if s['split']==r})) for r in ['train','dev','holdout','fine_diagnostic']}
 for r,n in CFG['minimum_groups'].items():assert summary[r]['music_groups']>=n,(r,summary)
 isolation={}
 for field in ['audio_sha256','raw_motion_sha256','recording_uid','choreography_uid','group','dancer_uid']:
  seen={}
  for s in sources:
   if s['split']=='fine_diagnostic':continue
   v=s.get(field)
   if v not in [None,'UNKNOWN']:assert seen.setdefault(v,s['split'])==s['split'],field
  isolation[field]=True
 return dict(sources=sources,summary=summary,isolation=isolation,genre_partition=gr,dancer_partition=dr,purged_cross_split_known_choreographies=sorted(bad),historical_r59_exposure=True,historical_pretraining_not_used=True,new_unseen_music_certified=False,initial_metadata_only_split_rejected='HASH_TRACK_AND_DANCER_PLUS_CHOREOGRAPHY_PURGE_LEFT_NO_DEV; NO_TENSORS_OR_MODEL_SCORES_READ',fine_dancer_isolation='NOT_CERTIFIED_DIAGNOSTIC_ONLY',skeleton=meta['skeleton'],parent_normalizer=meta['normalizer'])

class Store:
 def __init__(self,reg):self.reg=reg;self.cache={};self.meta={s['source_uid']:s for s in reg['sources']};assert sha(reg['parent_normalizer']['path'])==reg['parent_normalizer']['sha256'];self.old=read(reg['parent_normalizer']['path'])['statistics'];meta=read(CFG['data']);parent=Path('outputs/paired_event_r41_01/data/manifest.json');assert sha(parent)==meta['parent_manifest_sha256'];self.audio_hashes={s['audio_sha256']:s['features_sha256'] for s in read(parent)['sources']}
 def raw(self,uid):
  if uid in self.cache:return self.cache[uid]
  s=self.meta[uid];assert sha(s['arrays'])==s['arrays_sha256'];z=np.load(s['arrays']);R=z['R'];D,DM=observations(R,np.asarray(self.reg['skeleton']['J']),np.asarray(self.reg['skeleton']['parents']));DM[0,198:272]=False;DM[:,-2:]=False
  if s['dataset']=='AIST++':
   p=Path('outputs/transfer_audit_01/a2/audio')/(s['audio_sha256']+'.npz');assert sha(p)==self.audio_hashes[s['audio_sha256']];v=np.load(p);x,mask=v['values'],v['mask'];ticks=np.arange(len(R))+s['audio_offset_frames'];lo=np.floor(ticks).astype(int).clip(0,len(x)-1);hi=np.ceil(ticks).astype(int).clip(0,len(x)-1);f=(ticks-lo)[:,None];M=(x[lo]*(1-f)+x[hi]*f).astype(np.float32);MM=mask[lo]&mask[hi]
  else:
   stat=self.old['music'];M=(z['M']*np.asarray(stat['std'])+np.asarray(stat['mean'])).astype(np.float32);MM=np.broadcast_to(z['MV'][:,None],M.shape).copy()
  n=min(len(M),len(D));out=dict(M=M[:n],MM=MM[:n],D=D[:n],DM=DM[:n],MV=MM[:n].any(-1));assert all(np.isfinite(out[k]).all() for k in ['M','D']);self.cache[uid]=out;return out
 def crop(self,row,delta=0,donor=None,normal=None,zero=False,reverse=False):
  z=self.raw(row['source_uid']);m=self.raw((donor or row)['source_uid']);a=row['start'];am=(donor or row)['start']+delta;T=CFG['frames'];M=m['M'][am:am+T].copy();MM=m['MM'][am:am+T];D=z['D'][a:a+T].copy();DM=z['DM'][a:a+T].copy();assert len(M)==len(D)==T and am>=0
  if normal:
   M=np.where(MM,(M-normal['M_mean'])/normal['M_std'],0).astype(np.float32);D=np.where(DM,(D-normal['D_mean'])/normal['D_std'],0).astype(np.float32)
  if zero:M[:]=0
  if reverse:D=D[::-1].copy();DM=DM[::-1].copy()
  return M,D,DM,MM.any(-1)

def windows(reg):
 rows=[]
 for s in reg['sources']:
  lo=s['shared_start']+15;hi=s['frames']-CFG['frames']-15
  if hi<lo:continue
  for start in np.unique(np.linspace(lo,hi,CFG['windows_per_source']).round().astype(int)):
   rows.append(dict(uid=identity([s['source_uid'],int(start),CFG['frames']]),source_uid=s['source_uid'],split=s['split'],dataset=s['dataset'],group=s['group'],audio_sha256=s['audio_sha256'],start=int(start)))
 return rows

def statistics(store,rows):
 out={};proof=[]
 for name in ['M','D']:
  size=50 if name=='M' else 274;n=np.zeros(size);total=np.zeros(size);square=np.zeros(size)
  for r in rows:
   if r['split']!='train':continue
   z=store.raw(r['source_uid']);a=r['start'];v=z[name][a:a+150].astype(float);mask=z[name+'M' if name=='D' else 'MM'][a:a+150];n+=mask.sum(0);total+=(v*mask).sum(0);square+=(v*v*mask).sum(0)
  mean=total/np.maximum(n,1);std=np.maximum(np.sqrt(np.maximum(0,square/np.maximum(n,1)-mean*mean)),1e-5);out[name+'_mean']=mean;out[name+'_std']=std
 return out

def nuisance(M):
 from scipy.signal import find_peaks
 x=np.asarray(M,float);x=(x-x.mean(0))/np.maximum(x.std(0),1e-5);v=np.linalg.norm(np.diff(x,axis=0),axis=1);u=v-v.mean();ac=np.correlate(u,u,'full')[len(u)-1:];lag=10+int(np.argmax(ac[10:46]));peaks,_=find_peaks(v,distance=5,prominence=max(float(np.std(v)),1e-5));return np.array([1800/lag,len(peaks)/5,float(np.std(v))])

def donors(store,rows):
 vectors={r['uid']:nuisance(store.crop(r)[0]) for r in rows};training=np.stack([vectors[r['uid']] for r in rows if r['split']=='train']);std=np.maximum(training.std(0),1e-5);result={}
 for r in rows:
  others=[s for s in rows if s['split']==r['split'] and s['audio_sha256']!=r['audio_sha256']];assert others
  q=min(others,key=lambda s:(float(np.linalg.norm((vectors[r['uid']]-vectors[s['uid']])/std)),s['uid']));result[r['uid']]=dict(donor_uid=q['uid'],distance=float(np.linalg.norm((vectors[r['uid']]-vectors[q['uid']])/std)),query_proxy=vectors[r['uid']].tolist(),donor_proxy=vectors[q['uid']].tolist(),scope='NEAREST_TRAIN_STANDARDIZED_AUDIO_CHANGE_PERIOD_DENSITY_NOT_CERTIFIED_BPM_OR_ARTISTIC_NEGATIVE')
 return result
