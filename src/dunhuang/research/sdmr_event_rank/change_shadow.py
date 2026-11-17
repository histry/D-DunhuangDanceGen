"""Real full ChangE events, legal E1-W maps and frozen R41 scoring. No semantic labels."""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,sha,Library,rotation_sample,multi_warp,relative_fk
from dunhuang.research.transition_r37.clock import source_positions
from dunhuang.research.transition_r37.content import observations
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.paired_event_r41.population import Population
from dunhuang.research.paired_event_r41.train import digest
from dunhuang.research.source_context_r42.contracts import exact_arrays
from .features import load_model
from .scorer import R41EventScorer,scores,batch_partial


def run(evaluation,out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh native ChangE shadow')
    out.mkdir(parents=True);cfg=read(Path(__file__).with_name('config.json'));pop=Population(cfg['data']);lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);catalog=read('outputs/complete_r36_01/planning/catalog.json')['units'];plans=read('outputs/transition_r37_02/planning/result.json')['songs'];music=np.load('outputs/exposed_full_song_01/music/arrays.npz');inventory=read('outputs/exposed_full_song_01/music/inventory.json');songs=read('outputs/transition_r35_01/planning/songs.json');sk=np.load(Path(cfg['data'])/'skeleton.npz');J=sk['J'];parents=sk['parents'];cal=read(Path(evaluation)/'result.json')['train_hybrid_calibration'];forbid_optimizers();model=load_model(cfg);before=digest(model);service=R41EventScorer(model,cfg,pop.manifest['normalizer']);arrays=dict(skeleton_J22=J,parents=parents);records=[];unavailable=[]
    expected_skeleton=np.load('outputs/transition_r35_01/generation/arrays.npz');assert np.array_equal(J,expected_skeleton['skeleton_J22']) and np.array_equal(parents,expected_skeleton['parents'])
    normal=pop.statistics
    def normalized(values,mask,key):return np.where(mask,(values-np.array(normal[key]['mean']))/np.array(normal[key]['std']),0).astype(np.float32)
    for song in plans:
        item=next(s for s in songs if s['name']==song['name']);asset=next(s for s in inventory if s['name']==song['name']);assert asset['audio']['sha256']==item['audio']['sha256'];checked(item['audio']);plan=read(song['new_plan']);events=plan['events'];chosen=np.unique(np.linspace(0,len(events)-1,min(8,len(events))).round().astype(int))
        for event_index in chosen:
            selected=events[int(event_index)];A,B=map(int,selected['music_interval']);L=B-A;key=song['id']+'_'+str(int(event_index));case=int(item['case']);other=(case+1)%len(inventory)
            if not 90<=L<=420 or B>len(music[str(other)+'__music50']):unavailable.append(dict(key=key,frames=L,reason='REGISTERED_FULL_EVENT_OR_COUNTERFACTUAL_SUPPORT_UNAVAILABLE_NOT_CROPPED'));continue
            alternatives=[]
            for entry in catalog:
                if entry['unit_uid']==selected['unit_uid']:continue
                warp=multi_warp(entry['source_interval'],[A,B],entry.get('phases',[]),item['beats'])
                if warp is not None:alternatives.append(dict(entry,music_interval=[A,B],**warp))
            # Fixed source-diverse union before reading a learned score.
            alternatives.sort(key=lambda r:(r['source_uid'],r['unit_uid']));pool=[copy.deepcopy(selected)];used={selected['unit_uid']};seen={selected['source_uid']}
            for r in alternatives:
                if r['source_uid'] not in seen:pool.append(r);seen.add(r['source_uid']);used.add(r['unit_uid'])
            for r in alternatives:
                if r['unit_uid'] not in used and len(pool)<24:pool.append(r);used.add(r['unit_uid'])
            if len(pool)<2:unavailable.append(dict(key=key,frames=L,reason='NO_REAL_LEGAL_COMPETITOR'));continue
            mp=music[str(case)+'__music50'][A:B];mm=music[str(case)+'__music_mask'][A:B];sw=music[str(other)+'__music50'][A:B];sm=music[str(other)+'__music_mask'][A:B];pairs=[];provenance=[]
            for j,e in enumerate(pool):
                ticks=np.arange(A,B);q=source_positions(e,ticks);R=rotation_sample(lib.rotations[e['source_uid']],q);values,mask=observations(R,J,parents);mask[0,198:272]=False;arrays[key+f'_c{j}_xyz']=relative_fk(R,J,parents).astype(np.float32);arrays[key+f'_c{j}_source_positions']=q
                assert len(values)==L and q.min()>=e['source_interval'][0] and q.max()<=e['source_interval'][1]-1+1e-6
                motion=normalized(values,mask,'motion');pairs.append((motion,mask));provenance.append(dict(normalizer_outlier_fraction=float(np.mean(abs(motion[mask])>5)),event_uid=e['unit_uid'],source_uid=e['source_uid'],recording_uid=e.get('recording_uid',e['source_uid']),source_interval=e['source_interval'],music_interval=[A,B],time_warp=e['time_warp'],clock_contract=e.get('clock_contract',e.get('time_contract')),reference=lib.rows[e['source_uid']][0]['reference'],phase_anchors=e.get('anchor_source_frames',e.get('phase_anchors',[])),prior_selected=j==0,semantic_truth='UNKNOWN'))
            with torch.no_grad():
                D=torch.tensor(np.stack([v[0] for v in pairs]),device='cuda');DM=torch.tensor(np.stack([v[1] for v in pairs]),device='cuda');M=torch.tensor(np.repeat(normalized(mp,mm,'music')[None],len(pool),0),device='cuda');MV=torch.tensor(np.repeat(mm.any(-1)[None],len(pool),0),device='cuda');mg,dg,mt,dt,mv,dv=service.encode_normalized([M,D,DM,MV]);tokens=[]
                for j in range(len(pool)):tokens.append(dict(mg=mg[j].cpu().numpy(),dg=dg[j].cpu().numpy(),mt=mt[j].cpu().numpy(),dt=dt[j].cpu().numpy(),mv=mv[j].cpu().numpy(),dv=dv[j].cpu().numpy(),positions=np.arange(6,L-6,dtype=np.int32)))
                queries={'actual':tokens[0]}
                for control,raw,mask in [('music_zero',np.zeros_like(mp),mm),('music_swap',sw,sm)]:
                    # Zero means normalized encoder content zero, observation support preserved.
                    m=np.zeros_like(mp) if control=='music_zero' else normalized(raw,mask,'music');mg,_,mt,_,mv,_=service.encode_normalized([torch.tensor(m[None],device='cuda'),D[:1],DM[:1],torch.tensor(mask.any(-1)[None],device='cuda')]);queries[control]=dict(tokens[0],mg=mg[0].cpu().numpy(),mt=mt[0].cpu().numpy(),mv=mv[0].cpu().numpy())
            variants={}
            for control,query in queries.items():
                ps,pc,cost=batch_partial(query,tokens,cfg);v=[]
                for j,c in enumerate(tokens):
                    s,_=scores(query,c,cfg,False);hybrid=.5*((np.array([s['Global'],s['Diagonal']])-np.array(cal['mean']))/np.array(cal['std'])).sum();v.append([s['Global'],s['Diagonal'],s['Offset'],float(ps[j]),float(hybrid),float(pc[j])])
                variants[control]=v;arrays[key+'_'+control+'_scores']=np.array(v)
            for j,t in enumerate(tokens):
                for name in ('mt','dt','mv','dv','positions'):arrays[key+f'_c{j}_'+name]=t[name]
            records.append(dict(key=key,song=song['name'],case=case,event_index=int(event_index),frames=L,query_audio=item['audio'],audio_frame_interval=[A,B],query_prior_source=provenance[0],candidates=provenance,variants=variants,music_swap_audio=inventory[other]['audio'],can_route=False,human_reviewed=False,semantic_truth='UNKNOWN',scope='HISTORICALLY_EXPOSED_DUNHUANG_FULL_EVENTS; PRIOR_SELECTION_IS_NOT_TRUTH; FIXED_MAP_UNDER_MUSIC_CONTROLS'))
            np.savez_compressed(out/'arrays.npz',**arrays);print('E2_CHANGE_SHADOW',key,L,len(pool),flush=True)
    assert digest(model)==before;write(out/'result.json',dict(records=records,unavailable=unavailable,checkpoint_sha256=sha(cfg['checkpoint']),normalizer=pop.manifest['normalizer'],music_feature_arrays_sha256=sha('outputs/exposed_full_song_01/music/arrays.npz'),evaluation_sha256=sha(Path(evaluation)/'result.json'),derivatives='RECOMPUTED_AFTER_ACTUAL_SO3_E1W_WARP; FIRST_DERIVATIVE_UNKNOWN',optimizer_steps=0,can_route=False,operational_qualified=False,semantic_ground_truth=False))
    if replay:
        exact_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');assert read(out/'result.json')==read(Path(replay)/'result.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.evaluation,a.out,a.replay_from)
