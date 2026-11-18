from .common import *
from dunhuang.research.warp_coupling_r53.population import Population

class Study:
 def __init__(self):
  self.parent=read(PARENT/'controls/result.json');self.pop=Population();self.units={u['unit_uid']:u for u in read('outputs/event_coverage_r47_01/index/result.json')['units']};self.cache={}
 def record(self,song,arm):return next(r for r in self.parent['records'] if r['song']==song and r['arm']==arm)
 def pool(self,song,arm,slot):return self.parent['graphs'][self.record(song,arm)['graph_key']]['pools'][slot]
 def sample(self,node,target,kind='warp',transport=None):
  u=node['unit'];a,b=u['source_interval'];A,B=target;raw=self.pop.lib.rotations[u['source_uid']]
  if kind=='native':q=np.arange(a,b,dtype=float);w=np.array([[a,0],[b-1,b-a-1]],float);origin=0
  else:
   if kind=='warp':w=np.asarray(node['warp']['time_warp'],float)
   elif kind=='linear':w=np.array([[a,A],[b-1,B-1]],float)
   elif kind=='transport':
    w=np.asarray(transport['warp']['time_warp'],float).copy();w[:,1]=A+(w[:,1]-w[0,1])*(B-A-1)/(w[-1,1]-w[0,1])
   else:raise ValueError('Unknown time map')
   rates=np.diff(w[:,1])/np.diff(w[:,0]);lo,hi=CFG['transport_rate_limits']
   if (rates<lo-1e-10).any() or (rates>hi+1e-10).any():return None
   if not (np.isclose(w[0,0],a) and np.isclose(w[-1,0],b-1) and np.isclose(w[0,1],A) and np.isclose(w[-1,1],B-1)):raise ValueError('Warp support mismatch')
   q=np.interp(np.arange(A,B),w[:,1],w[:,0]);origin=A
  if (q<a-1e-9).any() or (q>b-1+1e-9).any():raise ValueError('Cannot extrapolate source')
  key=(u['unit_uid'],kind,origin,tuple(w.ravel().tolist()),tuple(q.tolist()))
  if key not in self.cache:
   R=raw[a:b].copy() if kind=='native' else rotation_sample(raw,q);P=relative_fk(R,self.pop.net.joints,self.pop.net.parents);m=stats(P,w[1:-1,1]-origin);self.cache[key]=dict(R=R,P=P,q=q,time_warp=w.tolist(),metrics=m)
  return self.cache[key]
 def index(self,song,arm,slot):return {n['unit']['unit_uid']:n for n in self.pool(song,arm,slot)}
 def target(self,song,arm,slot):return self.record(song,arm)['events'][slot]['target_interval']

def main(out,render=True):
 out=fresh(out);s=Study();song=CFG['main_song'];slot=CFG['main_slot'];left=s.index(song,'B0',slot);right=s.index(song,'B1',slot);common=sorted(left.keys()&right.keys());assert common
 arrays=dict(J=s.pop.net.joints,parents=s.pop.net.parents);records=[]
 for k,uid in enumerate(common):
  node=right[uid];methods={};native=s.sample(node,[0,1],'native')
  for name,arm,kind in (('Native','B1','native'),('B0','B0','warp'),('LinearB1','B1','linear'),('B1','B1','warp')):
   item=s.sample((left if arm=='B0' else right)[uid],s.target(song,arm,slot),kind);assert item is not None;arrays[f'case{k}_{name}_R']=item['R'];arrays[f'case{k}_{name}_P']=item['P'];methods[name]=dict(frames=len(item['R']),metrics=item['metrics'],amplification=amplification(item['metrics'],native['metrics']),time_warp=item['time_warp'],source_positions=item['q'].tolist())
  records.append(dict(key=f'case{k}',event_uid=uid,source_uid=node['unit']['source_uid'],source_interval=node['unit']['source_interval'],source_sha256=s.pop.lib.rows[node['unit']['source_uid']][0]['reference']['sha256'],song=song,slot=slot,methods=methods,audio_origin=0,native_music_pairing='UNPAIRED_SOURCE_ONLY',selection='ALL_COMMON_LEGAL_SLOT0_EVENTS_BEFORE_METRICS'))
  print('R57_MAIN',uid,flush=True)
 result=dict(records=records,binding=binding(),parent=s.pop.binding,root_contact='UNKNOWN',native24='NOT_EVALUABLE',human_reviewed=False,neural_training_steps=0,can_route=False);save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
 if render:
  from .view import viewer
  viewer(out,result,arrays,s.pop)
 return result

def controls(out):
 assert read(ROOT/'receipts/main_browser.json')['passed'];out=fresh(out);s=Study();rows=[];factorials=[];unsupported=[];frozen=[]
 for song in CFG['songs']:
  rr={a:s.record(song,a) for a in CFG['arms']}
  for arm in CFG['arms']:
   m=rr[arm]['metrics']['kinematics'];frozen.append(dict(song=song,arm=arm,whole=m['whole'],region_means={kind:{'D%d_Q95'%k:float(np.mean([r['metrics']['D%d_Q95'%k] for r in m['regions'] if r['kind']==kind])) for k in (1,2,3)} for kind in ('core','transition','left_boundary')}))
  for slot in range(len(rr['B0']['events'])):
   idx={a:s.index(song,a,slot) for a in CFG['arms']};common=sorted(idx['B0'].keys()&idx['B1'].keys())
   for uid in sorted(idx['B0'].keys()|idx['B1'].keys()):
    if uid not in common:unsupported.append(dict(song=song,slot=slot,event_uid=uid,legal_B0=uid in idx['B0'],legal_B1=uid in idx['B1']));continue
    native=s.sample(idx['B0'][uid],[0,1],'native');item=dict(song=song,slot=slot,event_uid=uid,source_uid=idx['B0'][uid]['unit']['source_uid'],source_interval=idx['B0'][uid]['unit']['source_interval'],native=native['metrics'])
    for arm in CFG['arms']:
     target=s.target(song,arm,slot)
     for kind in ('warp','linear'):
      q=s.sample(idx[arm][uid],target,kind);item[arm+'_'+kind]=dict(metrics=q['metrics'],amplification=amplification(q['metrics'],native['metrics']),time_warp=q['time_warp'])
    transported=s.sample(idx['B1'][uid],s.target(song,'B1',slot),'transport',idx['B0'][uid]);item['transport_B0_shape_to_B1']=None if transported is None else dict(metrics=transported['metrics'],amplification=amplification(transported['metrics'],native['metrics']),time_warp=transported['time_warp']);n0=np.diff(s.target(song,'B0',slot))[0];n1=np.diff(s.target(song,'B1',slot))[0];item['continuous_uniform_speed_cubic_prediction']=float(((n0-1)/(n1-1))**3);rows.append(item)
   chosen=[rr[a]['events'][slot]['event_uid'] for a in CFG['arms']];cells={};identities={}
   for e,uid in enumerate(chosen):
    for t,arm in enumerate(CFG['arms']):
     if uid in idx[arm]:
      q=s.sample(idx[arm][uid],s.target(song,arm,slot));cells[str(e)+str(t)]=q['metrics']['D3_Q95'];identities[str(e)+str(t)]=dict(event_uid=uid,target=s.target(song,arm,slot))
   factorials.append(dict(song=song,slot=slot,cells=cells,identities=identities,**shapley(cells)))
  print('R57_CONTROLS',song,len(rows),flush=True)
 # Read-only G1 retry of EXACT R56 G0-error edges; no old ledger changes.
 from dunhuang.research.manifold_bridge_r47.geometry import segmented
 from dunhuang.research.bridge_dynamics_r48.study import metrics as gap_metrics
 diagnostics=[];curves={}
 for graphkey,graph in sorted(s.parent['graphs'].items()):
  for edge in graph['edges']:
   if edge.get('rejection')!='G0_AMBIGUOUS_SO3_CHART_CURRENT_METHOD_REJECTED':continue
   k,i,j=edge['slot'],edge['left'],edge['right'];L=graph['pools'][k][i];R=graph['pools'][k+1][j];cores=[r['interval'] for r in graph['spans'] if r['kind']=='core'];left=s.sample(L,cores[k])['R'];right=s.sample(R,cores[k+1])['R'];observed=np.full((CFG['gap']+6,22,3,3),np.nan);observed[:3]=left[-3:];observed[-3:]=right[:3];Q,guard=segmented(observed,(3,3+CFG['gap']),tuple(CFG['G1_counts']),np.pi-CFG['chart_radius_margin']);m=None;passed=False
   if Q is not None:
    assert np.array_equal(Q[:3],observed[:3]) and np.array_equal(Q[-3:],observed[-3:]);full=np.concatenate((left[-4:-3],Q,right[3:4]));m=gap_metrics(full,s.pop.net.joints,s.pop.net.parents,(4,4+CFG['gap']));ref=s.pop.reference;passed=bool(m['gap_D2']<=ref['D2'] and m['gap_D3']<=ref['D3'] and m['shoulder_wrist_min_m']>=ref['shoulder_wrist_lower_m']);curves[f'edge{len(diagnostics)}']=Q
   diagnostics.append(dict(graph_key=graphkey,slot=k,left=i,right=j,G1=guard,metrics=m,passes_original_screen=passed,R56_ledger_unchanged=True))
 valid=[r for r in factorials if r['status']=='SHARED_SUPPORT_FACTORIAL_DECOMPOSITION'];summary={}
 for song in CFG['songs']:
  group=[r for r in valid if r['song']==song];summary[str(song)]=dict(shared_slots=len(group),total_slots=13,mean_factorial={key:float(np.mean([r[key] for r in group])) if group else None for key in ('selection','timing','interaction','total')},scope='REGISTERED_SHARED_SUPPORT_ONLY_NO_GENERAL_INDEPENDENCE_CLAIM')
 result=dict(binding=binding(),parent=s.pop.binding,paired_events=rows,factorial=factorials,not_common_legal=unsupported,frozen_A0=frozen,G1_diagnostic=diagnostics,summary=summary,world_root_contact='UNKNOWN',physical_qualified=False,can_route=False,training_steps=0,optimizer_calls=0);write(out/'result.json',result);save_arrays(out/'arrays.npz',curves);return result

def verify(out,source):
 out=fresh(out);old=Path(source);expected=read(old/'result.json');before=binding();assert expected['binding']['code']==before['code'];from dunhuang.research.coverage_r34.common import forbid_optimizers
 forbid_optimizers();run=controls if 'paired_events' in expected else lambda p:main(p,False);actual=run(out/'rebuild');actual['binding']['commit']=expected['binding']['commit'];assert actual==expected;assert sha(old/'arrays.npz')==sha(out/'rebuild/arrays.npz');write(out/'result.json',dict(result_exact=True,arrays_bytes_exact=True,source_commit=expected['binding']['commit'],replay_commit=before['commit'],optimizer_calls=0,can_route=False))
