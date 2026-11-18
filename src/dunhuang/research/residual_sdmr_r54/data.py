"""Build duration/Heuristic-bin tasks from physical clocks, never from neural scores."""
from collections import defaultdict,Counter
import soundfile as sf
from .common import *
from dunhuang.io import checked
from dunhuang.research.paired_event_r41.population import Population
from dunhuang.research.paired_event_r41.events import validate_roles

def build(out):
 out=fresh(out);source=sealed('outputs/paired_event_r41_01','data/manifest.json');old=sealed('outputs/sdmr_event_rank_01','benchmark/manifest.json');pop=Population(CFG['data']);meta=read(old);rows=meta['rows'];validate_roles(rows);skeleton=Path(CFG['data'])/'skeleton.npz';z=np.load(skeleton);J,parents=z['J'],z['parents'];sources={s['source_uid']:s for s in pop.manifest['sources']};audio={};speed={};fingerprints={};new=[]
 for i,r in enumerate(rows):
  s=sources[r['source_uid']];uid=s['source_uid'];ak=r['audio_sha256']
  if ak not in audio:
   wave,sr=sf.read(checked(s['audio']),always_2d=True,dtype='float32');wave=wave.mean(1);n=int(len(wave)*30/sr);e=np.array([np.sqrt(np.mean(wave[round(t*sr/30):round((t+1)*sr/30)]**2)) for t in range(n)]);assert np.isfinite(e).all();audio[ak]=np.argsort(np.argsort(e)).astype(float)/max(n-1,1)
  if uid not in speed:
   # Full-record derivatives before cutting any retrieval subwindow.
   src=np.load(checked(s['source_arrays']));P=relative_fk(src['R'],J,parents);speed[uid]=np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(1)
  a,b=r['source_interval'];u,v=r['audio_frame_interval'];ticks=np.arange(r['frames'])+u
  if ticks.min()<0 or ticks.max()>len(audio[ak])-1:raise ValueError('Real audio support escaped')
  row=dict(r,music_activity=float(np.interp(ticks,np.arange(len(audio[ak])),audio[ak]).mean()),body_activity=float(np.clip(speed[uid][a:b].mean()/2,0,1)));new.append(row)
  unit=pop.unit(r['parent_index']);a,b=r['slice'];features=unit['music'][a:b,18:50];sig=np.concatenate([s.mean(0) for s in np.array_split(features,4)]);fingerprints[i]=sig/max(np.linalg.norm(sig),1e-12)
  if (i+1)%2000==0:print('R54_P0_ROWS',i+1,flush=True)
 rows=new;queries=[]
 for role in ('train','val'):
  for group in sorted({r['source_group'] for r in rows if r['role']==role}):
   sources_q=sorted({r['raw_motion_sha256'] for r in rows if r['role']==role and r['source_group']==group})[:CFG['sources_per_group']]
   for source_q in sources_q:
    for length in CFG['lengths']:
     ids=sorted([i for i,r in enumerate(rows) if r['raw_motion_sha256']==source_q and r['frames']==length],key=lambda i:(rows[i]['source_interval'],rows[i]['uid']))
     if ids:queries.extend(ids[int(k)] for k in np.unique(np.linspace(0,len(ids)-1,min(len(ids),CFG['queries_per_source_length'])).round().astype(int)))
 tasks=[];missing=[];bylength=defaultdict(list)
 for i,r in enumerate(rows):bylength[r['role'],r['frames']].append(i)
 for q in queries:
  r=rows[q];pool=bylength[r['role'],r['frames']];h={i:float(heuristic(r['music_activity'],rows[i]['body_activity'])) for i in pool};pos=sorted([i for i in pool if relation(r,rows[i])=='CLOCK_POSITIVE'],key=lambda i:rows[i]['uid'])
  if not pos:missing.append(dict(query=q,reason='NO_CROSS_RECORDING_CLOCK_POSITIVE'));continue
  bin_id=int(np.floor(h[pos[0]]/CFG['heuristic_bin_width']));pos=[i for i in pos if int(np.floor(h[i]/CFG['heuristic_bin_width']))==bin_id][:CFG['maximum_positives']]
  negative=[]
  for i in pool:
   rel=relation(r,rows[i])
   if rel not in ('CLOCK_NON_TARGET_TIME','DIFFERENT_AUDIO_ASSOCIATION'):continue
   if rel=='CLOCK_NON_TARGET_TIME' and fingerprints[q]@fingerprints[i]>=CFG['repeat_cosine']:continue
   negative.append(i)
  for family in ('cross_music','same_music'):
   broad=[i for i in negative if (rows[i]['audio_sha256']!=r['audio_sha256'])==(family=='cross_music')]
   hard=[i for i in broad if int(np.floor(h[i]/CFG['heuristic_bin_width']))==bin_id];hard=sorted(hard,key=lambda i:rows[i]['uid'])[:CFG['maximum_negatives']]
   if len(hard)<CFG['minimum_negatives']:missing.append(dict(query=q,family=family,reason='INSUFFICIENT_SAME_HEURISTIC_BIN',count=len(hard)));continue
   seed=int(identity([CFG['seed'],r['uid'],family])[:12],16);rng=np.random.default_rng(seed);broad=sorted(broad,key=lambda i:rows[i]['uid']);unconditional=[broad[k] for k in rng.choice(len(broad),len(hard),replace=False)];ids=sorted(pos+hard,key=lambda i:rows[i]['uid']);other=sorted(pos+unconditional,key=lambda i:rows[i]['uid']);p=[i in pos for i in ids]
   donor=next(i for i in pool if rows[i]['audio_sha256']!=r['audio_sha256']);tasks.append(dict(uid=identity([r['uid'],family,bin_id]),query=q,candidates=ids,unconditional_candidates=other,positive=p,unconditional_positive=[i in pos for i in other],shuffled_positive=permuted_labels(p,seed),role=r['role'],group=r['source_group'],recording=r['raw_motion_sha256'],family=family,frames=r['frames'],heuristic=[h[i] for i in ids],unconditional_heuristic=[h[i] for i in other],heuristic_bin=bin_id,donor=donor,labels='PHYSICAL_CLOCK_OR_AUDIO_ASSOCIATION_NOT_ARTISTIC_INCOMPATIBILITY'))
 assert tasks
 used=sorted({i for t in tasks for i in [t['query'],t['donor']]+t['candidates']+t['unconditional_candidates']});arrays=dict(J=J,parents=parents)
 for i in used:
  r=rows[i];unit=pop.unit(r['parent_index']);a,b=r['slice']
  for dest,key in [('M','music'),('D','motion'),('DM','motion_mask')]:arrays[dest+str(i)]=unit[key][a:b]
  arrays['MV'+str(i)]=unit['music_mask'][a:b].any(-1)
 save_arrays(out/'arrays.npz',arrays);groups={role:sorted({t['group'] for t in tasks if t['role']==role}) for role in ('train','val')};assert len(groups['train'])>=CFG['minimum_train_groups'];checkpoint=sealed('outputs/paired_event_r41_01','main/model.pt');binding=dict(protocol_sha256=sha(Path(__file__).with_name('protocol.json')),r41_manifest_sha256=sha(source),r43_manifest_sha256=sha(old),checkpoint_sha256=sha(checkpoint),normalizer=pop.manifest['normalizer'],skeleton_sha256=sha(skeleton),code=code_binding(),arrays_sha256=sha(out/'arrays.npz'))
 result=dict(binding=binding,rows=rows,tasks=tasks,queries=queries,missing=missing,groups=groups,role_isolation=validate_roles(rows),shared_choreography_across_roles=pop.manifest['shared_choreography_across_roles'],statistics=dict(Counter((t['role']+'_'+t['family']) for t in tasks)),activity='R36_AUDIO_RMS_PERCENTILE_AND_R53_CLIPPED_COMMON22_FK_SPEED; WHOLE_RECORD_FIRST',candidate_scope='OFFLINE_PAIRED_DIAGNOSTIC_CONDITIONING_ON_POSITIVE_HEURISTIC_BIN; NOT_DEPLOYABLE_SHORTLIST',human_reviewed=False,test_tensor_reads=0,finedance_admitted=False,can_route=False)
 write(out/'manifest.json',result);print('R54_P0_COMPLETE',result['statistics'],{k:len(v) for k,v in groups.items()},flush=True)
