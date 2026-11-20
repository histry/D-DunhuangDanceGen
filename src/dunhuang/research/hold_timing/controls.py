"""Common frozen domain: hold-priority hypotheses, dual readout, explicit fusion abstention."""
from pathlib import Path
from collections import Counter
import sys
import numpy as np
from .study import ROOT,CFG,binding,load_study,summarize_motion
from . import metrics
from dunhuang.research.local_timing_r68.common import read,write,sha
from dunhuang.research.compensation_r73 import warp
from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost
from dunhuang.research.sdmr_warp_rank import connectivity,encoder,readout
from dunhuang.research.sdmr_warp_rank.common import CFG as R67
from dunhuang.research.complete_r36.core import rotation_sample
from dunhuang.research.warp_coupling_r53.common import save_arrays,array_hash

def run(destination=None):
 out=Path(destination) if destination else ROOT/'controls';out.mkdir(parents=True,exist_ok=False)
 gate=read(ROOT/'receipts/preview_browser_acceptance.json');assert gate['status']=='PASS' and gate['manifest_sha256']==sha(ROOT/'audit/manifest.json')
 ctx=connectivity.prepare();cases=read(ROOT/'audit/registration.json')['cases'];cal=read(ROOT/'audit/calibration.json');natural=read(ROOT/'audit/natural_sources.json');proposals=read('outputs/compensation_r73_01/controls/proposals.json')
 parent=Path('outputs/gtw_baseline_01/controls');m=read(parent/'manifest.json');assert sha(parent/'manifest.json')==CFG['GTW_manifest'] and sha(parent/'costs.json')==m['files']['costs.json']
 from dunhuang.research.gtw_kinematic_baseline.study import binding as gtw_binding
 assert m['binding']==gtw_binding()
 oldcosts=read(parent/'costs.json');net=encoder.FrozenEncoder();assert np.array_equal(net.joints[:22],ctx['J'][:22]) and np.array_equal(net.parents[:22],ctx['parents'][:22])
 path=Path('outputs/exposed_full_song_01/music/arrays.npz');assert sha(path)==R67['change_raw50_feature_arrays_sha256'];music=np.load(path);rows=[];candidates=[];arrays={};scorecache={};tokencache={};proofs=[]
 for ci,c in enumerate(cases):
  uid=c['source_uid'];source=ctx['lib'].rotations[uid];a,b=c['source_interval'];A,B=c['target_interval'];song=int(c['songidx']);donor=(song+1)%3
  model=ReferenceCost(source[max(0,a-30):min(len(source),b+30)],ctx['J'],ctx['parents']);bank=[];seen={}
  def add(cid,q,gtw,origin):
   q=np.asarray(q);qh=array_hash(q)
   if qh in seen:seen[qh]['aliases'].append(cid);return
   R=rotation_sample(source,q);h=array_hash(R);holds=metrics.occupancy(q,natural[uid]['regions']['body']['low_runs'],[a,b]);v=dict(id=cid,aliases=[],q=q,R=R,clock_sha256=qh,rotation_sha256=h,gtw=gtw,origin=origin,mapped_source_holds=holds,clock_shift=float(np.sqrt(np.mean((q-q0)**2))));seen[qh]=v;bank.append(v)
  q0=warp.old.baseline(c['time_warp'],c['target_interval'])['q'];R0=rotation_sample(source,q0);add('E1W',q0,model.score(R0)[0],'FROZEN_BASELINE')
  uniform=warp.old.uniform([a,b],c['target_interval'])
  if uniform['status']=='LEGAL':add('Uniform',uniform['q'],model.score(rotation_sample(source,uniform['q']))[0],'FROZEN_UNIFORM')
  for p in proposals:
   if p['case_key']!=c['key'] or p['status']!='LEGAL':continue
   spec={k:p[k] for k in ['width','segments','activity','kind','rank']};solved=warp.solve(c,spec,p['delta']);assert solved['status']=='LEGAL' and array_hash(solved['q'])==p['q_sha256']
   R=rotation_sample(source,solved['q']);assert array_hash(R)==p['core_R_sha256']
   old=next(v for v in oldcosts if v['id']==p['id'] and v['unit'].startswith(c['key']+'_'));assert old['core_sha256']==array_hash(R)
   add(p['id'],solved['q'],old['scores'],'EXACT_R73_LEGAL_PROPOSAL_WITH_SAVED_GTW_COST')
  # Compare same body/clock across audio. Audio cannot change the motion or GTW cost.
  musicvectors={};musicevidence={};proxy={}
  for label,si in [('actual',song),('wrong',donor)]:
   M=music[str(si)+'__music50'][A:B];MM=music[str(si)+'__music_mask'][A:B];assert len(M)==240
   musicvectors[label]=net.encode_music_raw(M,MM);musicevidence[label]=musicvectors[label][2]
   with np.load(Path('outputs/music_events_01/main')/str(si)/'activations.npz') as z:
    E=z['acoustic_energy'];O=z['acoustic_onset'];energy_rank=np.searchsorted(np.sort(E),E,side='right')/len(E);proxy[label]=(energy_rank+O)/2
  motions=[]
  for v in bank:
   h=v['rotation_sha256']
   if h not in tokencache:tokencache[h]=net.encode_rotations(v['R'])
   motions.append(tokencache[h])
  support=np.logical_and.reduce([v[1] for v in musicvectors.values()]+[v[1] for v in motions])
  for label,tokens in musicvectors.items():
   result=readout.score_bank(tokens[0],[x[0] for x in motions],tokens[1],[x[1] for x in motions],centres=np.arange(6,234),fixed_support=support)
   for j,v in enumerate(bank):
    v[label+'_Single64']=result['scores']['Single64'][j];v[label+'_Multi64']=result['scores']['Multi64'][j]
  for v in bank:
   v['hold_cost']=metrics.excess(v['mapped_source_holds'],cal['fixed_cap_seconds']);v['music_hold_cost']=metrics.excess(v['mapped_source_holds'],cal['fixed_cap_seconds'],proxy['actual'],A);v['wrong_music_hold_cost']=metrics.excess(v['mapped_source_holds'],cal['fixed_cap_seconds'],proxy['wrong'],A)
  def choose(field,maximize=False):
   valid=[v for v in bank if (v['gtw'].get(field) if field.startswith('GTW') else v.get(field)) is not None]
   def key(v):
    val=v['gtw'][field] if field.startswith('GTW') else v[field]
    return ((-val if maximize else val),v['gtw']['GTWLocal'],v['clock_shift'],v['id'])
   return min(valid,key=key) if valid else None
  get=lambda name:next((v for v in bank if v['id']==name or name in v['aliases']),None)
  chosen=[('E1W',get('E1W')),('Uniform',get('Uniform')),('GTWLocal',choose('GTWLocal')),('GTWContext',choose('GTWContext')),('HoldFirstFixed',choose('hold_cost')),('HoldFirstMusicProxy',choose('music_hold_cost')),('WrongMusicHoldProxy',choose('wrong_music_hold_cost')),('SDMRSingle64',choose('actual_Single64',True)),('SDMRMulti64',choose('actual_Multi64',True)),('WrongMusicMulti64',choose('wrong_Multi64',True)),('GuardedFusion',get('E1W'))]
  pareto=[v['id'] for v in bank if v['actual_Multi64'] is not None and not any(u['actual_Multi64'] is not None and u['gtw']['GTWLocal']<=v['gtw']['GTWLocal'] and u['actual_Multi64']>=v['actual_Multi64'] and (u['gtw']['GTWLocal']<v['gtw']['GTWLocal'] or u['actual_Multi64']>v['actual_Multi64']) for u in bank)]
  poolhash=array_hash(np.stack([v['q'] for v in bank]));proofs.append(dict(case_key=c['key'],candidate_count=len(bank),common_pool_sha256=poolhash,common_token_support=int(support.sum()),music=musicevidence,GTW_Multi64_pareto_ids=pareto,fusion='ABSTAIN_UNCALIBRATED_MUSIC_MARGIN_NO_HUMAN_TOLERANCE'))
  for v in bank:candidates.append({k:value for k,value in dict(v,case_key=c['key']).items() if k not in ['R','q']})
  for method,v in chosen:
   row=dict(case_key=c['key'],case_index=ci,method=method,status='MISSING_NO_FINITE_OR_CALIBRATED_CAP' if v is None else 'EXECUTED',common_pool_sha256=poolhash,candidate_count=len(bank),human_records=0,quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE',can_route=False)
   if v is None:rows.append(row);continue
   ctx['cache'].clear();conn,geo=connectivity.check(c,v['R'],ctx);h=v['rotation_sha256']
   if h not in scorecache:scorecache[h]=summarize_motion(v['R'],ctx,cal)
   full=None
   if 'full_R' in geo:
    fh=array_hash(geo['full_R'])
    if fh not in scorecache:scorecache[fh]=summarize_motion(geo['full_R'],ctx,cal)
    full=dict(rotation_sha256=fh,metrics=scorecache[fh])
   rows.append(dict(row,selected_id=v['id'],rotation_sha256=h,clock_sha256=v['clock_sha256'],gtw=v['gtw'],hold_cost=v['hold_cost'],music_hold_cost=v['music_hold_cost'],SDMRSingle64=v['actual_Single64'],SDMRMulti64=v['actual_Multi64'],core=scorecache[h],full_pair=full,connection=conn,
    protected_edges_exact=np.array_equal(v['R'][:4],R0[:4]) and np.array_equal(v['R'][-4:],R0[-4:]),fusion_status='ABSTAIN_UNCALIBRATED' if method=='GuardedFusion' else 'UNVALIDATED_RESEARCH_SELECTION'))
   for name,value in dict(R=v['R'],q=v['q'],**geo).items():arrays[c['key']+'__'+method+'__'+name]=value
  print('HOLD_COMMON_POOL',ci+1,len(cases),len(bank),flush=True)
 music.close();net.check();save_arrays(out/'arrays.npz',arrays);write(out/'records.json',rows);write(out/'candidates.json',candidates);write(out/'readout.json',proofs)
 write(out/'registration.json',dict(audit_manifest_sha256=sha(ROOT/'audit/manifest.json'),protocol=CFG,binding=binding(),controls_code_sha256=sha(Path(__file__)),encoder=net.binding,weights_before=net.before,weights_after=net.check(),reference_scope='SAME_FULL_SOURCE_INTERVAL_PLUS_AVAILABLE30_FRAMES',alignment_task='HOLD_SELECTION_NO_FIXED_ACTUAL_PEAK_STRATUM_NOT_ORIGINAL_GTW_PROTOCOL_A',music_quality='NOT_EVALUABLE',can_route=False))
 write(out/'summary.json',dict(cases=len(cases),positions=len(rows),actual=sum(r['status']=='EXECUTED' for r in rows),geometry=ctx['counters'],candidates=len(candidates),unique_motion_profiles=len(scorecache),distinct_encoded_motions=len(tokencache),training_steps=0,optimizer_calls=0,human_records=0,preference='NOT_EVALUABLE',R74='NOT_STARTED',can_route=False))
 write(out/'manifest.json',dict(files={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}))

if __name__=='__main__':run()
