"""Reconstruct the full fixed raw-record grid before any positive label or model score is consulted."""
import collections,sys,shutil,ast
import numpy as np
import soundfile as sf
from dunhuang.contracts import matrices
from dunhuang.research.transition_r37.content import observations
from dunhuang.research.paired_event_r41.events import validate_roles
from .common import *
from .metrics import starts

def normalize(x,mask,statistics,key):
 s=statistics[key];return np.where(mask,(x-np.asarray(s['mean']))/np.asarray(s['std']),0).astype(np.float32)

def energy(path):
 wave,sr=sf.read(path,always_2d=True,dtype='float32');wave=wave.mean(1);n=int(len(wave)*30/sr);v=np.array([np.sqrt(np.mean(wave[round(t*sr/30):round((t+1)*sr/30)]**2)) for t in range(n)]);assert np.isfinite(v).all()
 # Stable empirical midrank; silence/tied levels do not gain arbitrary temporal rank.
 from scipy.stats import rankdata
 return (rankdata(v,method='average')-1)/max(n-1,1)

def role_audit(sources):
 maps={key:{} for key in ('raw_motion_sha256','audio_sha256','group')}
 for r in sources:
  for k,m in maps.items():
   identity=r[k];assert m.setdefault(identity,r['role'])==r['role'],k
 return {k:len(v) for k,v in maps.items()}

def fine_assets():
 src=Path(CFG['fine_raw_root']);dst=Path(CFG['fine_copy']);dst.mkdir(parents=True,exist_ok=True);metadata=dst/'metadata';manifest=read(metadata/'file_manifest.json');lookup={r['path'].replace(chr(92),'/'):r for r in manifest};rows=[]
 official=metadata/'official_FineDance_dataset.py';assert sha(official)=='57e935ab937a701a99d4bf16dc6159225527a282011d2e1c8496b4a337d74035';tree=ast.parse(official.read_text(encoding='utf8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_train_test_list');scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(official),'exec'),scope);genre_train,genre_test,genre_ignore=scope['get_train_test_list']('cross_genre');dancer_train,dancer_test,dancer_ignore=scope['get_train_test_list']('cross_dancer')
 assert set(CFG['fine_ids'])<=set(genre_train)
 for name in CFG['fine_ids']:
  files={}
  for folder,ext in [('motion','.npy'),('music_wav','.wav'),('label_json','.json')]:
   rel=folder+'/'+name+ext;r=lookup[rel];origin=src/rel;target=dst/rel;target.parent.mkdir(exist_ok=True)
   if not target.exists():
    assert sha(origin)==r['sha256'];shutil.copyfile(origin,target)
   assert sha(target)==r['sha256'];files[folder]=dict(path=str(target),sha256=r['sha256'])
  rows.append(dict(sequence_id=name,files=files,genre_role='train',dancer_role='test' if name in dancer_test else 'train' if name in dancer_train else 'ignore'))
 return rows,dict(official_code_sha256=sha(official),file_manifest_sha256=sha(metadata/'file_manifest.json'),source_root=str(src),records=rows,primary_split='OFFICIAL_CROSS_GENRE_TRAIN_DIAGNOSTIC',cross_dancer_separate=True)

def prepare(out):
 out=fresh(out);(out/'sources').mkdir();parent=read(seal_member('outputs/paired_event_r41_01','data/manifest.json'));seals=read(Path(CFG['a2'])/'source_seals.json');byraw={s['raw_motion']['sha256']:s for s in seals};sk=np.load(Path(CFG['aist'])/'skeleton.npz');J,parents=sk['J'],sk['parents'];normal=Path(CFG['normalizer']);assert sha(normal)==parent['normalizer']['sha256'];stats=read(normal)['statistics'];audio={};sources=[];rows=[];checks=[]
 def add(source,R,root,M,MM,E):
  uid=source['source_uid'];D,DM=observations(R,J,parents);DM[0,198:272]=False;P=relative_fk(R,J,parents);A=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(1)/2
  n=min(len(R),len(M));assert n>=CFG['frames'];file=out/'sources'/(uid+'.npz');save_arrays(file,dict(M=normalize(M[:n],MM[:n],stats,'music'),D=normalize(D[:n],DM[:n],stats,'motion'),DM=DM[:n],MV=MM[:n].any(-1),R=R[:n],root=root[:n],E=E[:n],A=A[:n]));source.update(frames=n,arrays=str(file),arrays_sha256=sha(file));sources.append(source)
  for a in starts(source['shared_start'],n,CFG['frames'],CFG['stride']):
   b=a+CFG['frames'];rows.append(dict(uid=identity([uid,a,b,'R59_RAW_GRID']),source_uid=uid,dataset=source['dataset'],role=source['role'],group=source['group'],recording_uid=source['recording_uid'],choreography_uid=source.get('choreography_uid','UNKNOWN'),raw_motion_sha256=source['raw_motion_sha256'],audio_sha256=source['audio_sha256'],source_interval=[a,b],audio_interval=[a+source['audio_offset_frames'],b+source['audio_offset_frames']],frames=b-a,music_activity=float(E[a:b].mean()),body_activity=float(np.clip(A[a:b].mean(),0,1)),label_scope=source['clock_scope']))
 for number,s in enumerate(parent['sources']):
  proof=byraw[s['raw_motion_sha256']];raw=checked(proof['raw_motion']);native=np.load(checked(proof['native']));c=read(checked(proof['source_clock']));cal=c['calibration'];assert cal['verified'] and cal['origin_anchor_used'] and cal['maximum_residual_seconds']<=1/30;assert c['raw_motion']['sha256']==s['raw_motion_sha256'] and c['raw_audio']['sha256']==s['audio_sha256'];assert 30*cal['audio_offset_seconds']==s['audio_offset_frames'];assert proof['role']==s['role'] in ('train','val');R=matrices(native[:,7:139].reshape(-1,22,6));prior=np.load(checked(s['source_arrays']));assert np.array_equal(prior['R'],R)
  ak=s['audio_sha256'];offset=s['audio_offset_frames']
  if ak not in audio:
   path=checked(s['audio']);z=np.load(Path(CFG['a2'])/'audio'/(ak+'.npz'));assert sha(Path(CFG['a2'])/'audio'/(ak+'.npz'))==s['features_sha256'];audio[ak]=(z['values'],z['mask'],energy(path))
  v,mask,E=audio[ak];ticks=np.arange(len(R))+offset;good=(ticks>=0)&(ticks<=min(len(v),len(E))-1);valid=np.flatnonzero(good);lo,hi=int(valid[0]),int(valid[-1])+1;t=ticks[:hi];u=np.floor(t).astype(int).clip(0,len(v)-1);w=np.ceil(t).astype(int).clip(0,len(v)-1);f=(t-u)[:,None];M=(v[u]*(1-f)+v[w]*f).astype(np.float32);MM=mask[u]&mask[w];EE=np.interp(t,np.arange(len(E)),E);add(dict(source_uid=s['source_uid'],dataset='AIST++',role=s['role'],group=s['source_group'],recording_uid=s['recording_uid'],choreography_uid=s['choreography_uid'],raw_motion_sha256=s['raw_motion_sha256'],raw_motion=proof['raw_motion'],audio_sha256=ak,audio=s['audio'],audio_offset_frames=offset,shared_start=lo,clock_scope='VERIFIED_OFFICIAL_SOURCE_CLOCK',clock_proof=proof['source_clock'],root_scope='OBSERVED_FITTED_SOURCE_TRANSLATION_NOT_CONTACT_TRUTH'),R[:hi],native[:hi,4:7],M,MM,EE)
  checks.append(dict(raw_sha256=s['raw_motion_sha256'],native_sha256=proof['native']['sha256'],clock_sha256=proof['source_clock']['sha256'],original_fps=60,research_fps=30,raw_bytes_verified=True,rotations_equal_sealed_native_and_R41=True))
  if (number+1)%200==0:print('R59_AIST_SOURCES',number+1,len(rows),flush=True)
 fine,fproof=fine_assets();runtime=read('inputs/clock_audio_runtime_v2/receipt.json')
 for item in runtime['files']+runtime.get('wheels',[])+[runtime['wheel']]:checked(item)
 sys.path.insert(0,str(Path(runtime['import_root']).resolve()));legacy=Path('third_party/danceRAG').resolve();assert subprocess.check_output(['git','-C',str(legacy),'rev-parse','HEAD'],text=True).strip()=='465f35ba88bbc45d3415e623a60c55a2c8fdb49f';sys.path.insert(0,str(legacy))
 import librosa
 from evaluation.paired_alignment_observations_v3 import full_audio_observations
 assert librosa.__version__=='0.11.0';fine_receipts=[]
 for f in fine:
  file=checked(f['files']['motion']);native=np.load(file);assert native.ndim==2 and native.shape[1]==315 and np.isfinite(native).all();R=matrices(native[:,3:135].reshape(-1,22,6),row=True);assert np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))<2e-4;wav=checked(f['files']['music_wav']);M,MM,receipt=full_audio_observations(wav);E=energy(wav);assert abs(len(native)-len(M))<=2
  uid='fine_'+f['sequence_id'];add(dict(source_uid=uid,dataset='FineDance',role='genre_train_diagnostic',group='finedance_archive:'+f['sequence_id'],recording_uid=uid,raw_motion_sha256=f['files']['motion']['sha256'],raw_motion=f['files']['motion'],audio_sha256=f['files']['music_wav']['sha256'],audio=f['files']['music_wav'],audio_offset_frames=0.,shared_start=0,clock_scope='OFFICIAL_ARCHIVE_ID_AND_RAW_COMMON_START_NOT_AIST_WAVEFORM_CALIBRATED',root_scope='RAW_TRANSLATION_COORDINATES_NOT_NATIVE_WORLD_ADMITTED',dancer_role=f['dancer_role']),R,native[:,:3],M,MM,E);fine_receipts.append(dict(sequence_id=f['sequence_id'],extractor=receipt,raw_frames=len(native),audio_feature_frames=len(M),raw_origin_preserved=True,both_sides_trimmed=False));print('R59_FINE',uid,len(rows),flush=True)
 role=role_audit(sources);queries=[]
 for key in sorted({(r['dataset'],r['role'],r['group']) for r in rows}):
  eligible=[i for i,r in enumerate(rows) if (r['dataset'],r['role'],r['group'])==key];uids=sorted({rows[i]['raw_motion_sha256'] for i in eligible})[:CFG['query_sources_per_group']]
  for uid in uids:
   opts=sorted([i for i in eligible if rows[i]['raw_motion_sha256']==uid],key=lambda i:rows[i]['source_interval']);queries.extend(opts[int(k)] for k in np.unique(np.linspace(0,len(opts)-1,min(len(opts),CFG['queries_per_source'])).round().astype(int)))
 result=dict(binding=binding(),rows=rows,sources=sources,queries=queries,skeleton=dict(J=J.tolist(),parents=parents.tolist()),normalizer=dict(path=str(normal),sha256=sha(normal)),role_isolation=role,AIST_raw_audits=checks,FineDance_asset_proof=fproof,FineDance_audio_receipts=fine_receipts,parent_manifest_sha256=sha(Path(CFG['aist'])/'manifest.json'),pool_rule=CFG['pool_rule'],source_counts=dict(collections.Counter(r['dataset']+'_'+r['role'] for r in sources)),groups={d:len({r['group'] for r in sources if r['dataset']==d}) for d in ('AIST++','FineDance')},shared_choreography_across_AIST_roles=parent['shared_choreography_across_roles'],historically_exposed_AIST_val=True,unseen_pretraining_lineage_certified=False,full_official_FineDance_test=False,test_tensor_reads=0,training_updates=0);write(out/'manifest.json',result);print('R59_DATA',len(rows),len(queries),result['source_counts'],flush=True);return result

class Data:
 def __init__(self,path):
  self.path=Path(path);self.meta=read(self.path/'manifest.json');self.rows=self.meta['rows'];self.sources={s['source_uid']:s for s in self.meta['sources']};self.cache={};self.sk=self.meta['skeleton']
 def source(self,uid):
  if uid not in self.cache:
   r=self.sources[uid];p=Path(r['arrays']);assert sha(p)==r['arrays_sha256'];self.cache[uid]=dict(np.load(p))
  return self.cache[uid]
 def inputs(self,i):
  r=self.rows[i];z=self.source(r['source_uid']);a,b=r['source_interval'];return [z[k][a:b] for k in ('M','D','DM','MV')]


def verify(out,source):
 out=fresh(out);source=Path(source);expected=read(source/'manifest.json');actual=prepare(out/'rebuild');actual['binding']['commit']=expected['binding']['commit']
 for a,b in zip(actual['sources'],expected['sources']):
  assert a['source_uid']==b['source_uid'];assert sha(a['arrays'])==sha(b['arrays']);a['arrays']=b['arrays']
 assert actual==expected
 write(out/'result.json',dict(raw_sources_rebuilt=len(actual['sources']),complete_grid_and_manifest_exact=True,all_source_array_bytes_exact=True,source_manifest_sha256=sha(source/'manifest.json'),optimizer_calls=0))
