"""Complete pre-truncation funnel; no larger geometry or learned diversity reward."""
from itertools import product
from collections import Counter
import numpy as np
from dunhuang.research.complete_r36.core import multi_warp,rotation_sample,relative_fk,write,read
from dunhuang.research.event_alignment_r49.rank import main_scores
from dunhuang.research.transition_planning_r50.common import assets,layout
from dunhuang.research.transition_planning_r50.planning import bridge
from .common import fresh,regions


def solve(pools,edges,reference,top4=False,no_repeat=False,wrong_music=False,scoring="full"):
    def value(c):
        return c["parts"][0][scoring] if scoring in ("local","heuristic") else c["scores"][int(wrong_music)]
    choices=[]
    for p in pools:
        order=sorted(range(len(p)),key=lambda i:(-value(p[i]),p[i]['unit']['unit_uid']))
        if top4:
            ids=[];seen=set()
            for i in order:
                uid=p[i]['unit']['source_uid']
                if uid not in seen:ids.append(i);seen.add(uid)
                if len(ids)==4:break
            choices.append(ids)
        else:choices.append(order)
    best=None
    for ids in product(*choices):
        uids=[pools[k][i]['unit']['unit_uid'] for k,i in enumerate(ids)]
        if no_repeat and len(set(uids))!=len(uids):continue
        es=[edges.get((k,ids[k],ids[k+1])) for k in range(len(ids)-1)]
        if any(e is None or e[0] is None for e in es):continue
        score=sum(value(pools[k][i]) for k,i in enumerate(ids))
        score-=.01*sum(e[1]['methods'][e[1]['selected']]['gap_D3']/max(reference['D3'],1e-9) for e in es)
        candidate=(score,tuple(-choices[k].index(i) for k,i in enumerate(ids)),ids)
        if best is None or candidate[:2]>best[:2]:best=candidate
    return None if best is None else dict(ids=list(best[2]),score=best[0])


def run(out,controls=False):
    if controls:return control_run(out)
    out=fresh(out);net,lib,units,tracks=assets();ref=read('outputs/bridge_dynamics_r48_01/main/result.json')['reference']
    bank=[]
    for uid in sorted(lib.rows):
        choices=sorted((u for u in units if u['source_uid']==uid and 135<=u['duration_frames']<=399),key=lambda u:u['source_interval'])
        bank.extend(choices[k] for k in sorted(set((len(choices)//3,2*len(choices)//3))))
    assert len(bank)==16
    old={r['key']:r for r in read('outputs/transition_planning_r50_01/result.json')['cases']}
    records=[];arrays={'J':net.joints,'parents':net.parents}
    for song,track in enumerate(tracks):
      for count in (2,3):
       for gap in (24,48,72):
        key=f'song{song}_events{count}_gap{gap}';base=old[key];spans=base['regions'];cores=[r['interval'] for r in spans if r['kind']=='core'];pools=[];rejected=[];catalog_pools=[]
        for slot,(a,b) in enumerate(cores):
            pool=[]
            music=[]
            for t in ((track,tracks[(song+1)%3]) if controls else (track,)):
                music.append((net.music_tokens(t['values'][a:b],t['mask'][a:b]),np.asarray(t['signal']['energy'][a:b]),float(np.mean(t['structure']['energy_percentile'][a:b]))))
            for unit in units:
                phases=[dict(p,type=p['body_region']+'_'+p['type']) for p in unit['phases']]
                warp=multi_warp(unit['source_interval'],[a,b],phases,track['structure']['beats'])
                if warp is None:
                    rejected.append(dict(slot=slot,event_uid=unit['unit_uid'],reason='E1W_TIME_INFEASIBLE'));continue
                w=np.array(warp['time_warp']);q=np.interp(np.arange(a,b),w[:,1],w[:,0]);parts=[main_scores(net,lib.rotations[unit['source_uid']],q,*m) for m in music]
                if any(p is None for p in parts):
                    rejected.append(dict(slot=slot,event_uid=unit['unit_uid'],reason='COMMON_MUSIC_SUPPORT_MISSING'));continue
                penalty=float(np.mean(abs(np.log(np.diff(w[:,1])/np.diff(w[:,0])))))
                scores=[p['local']+.1*p['multi']-.05*penalty for p in parts]
                pool.append(dict(unit=unit,warp=warp,R=rotation_sample(lib.rotations[unit['source_uid']],q),scores=scores,parts=parts,warp_cost=penalty))
            catalog_pools.append([{k:v for k,v in c.items() if k!="R"} for c in pool])
            pools.append([c for c in pool if c["unit"]["unit_uid"] in {u["unit_uid"] for u in bank}])
        edges={}
        for k in range(count-1):
          for i,l in enumerate(pools[k]):
           for j,r in enumerate(pools[k+1]):
            if l['unit']['source_uid']!=r['unit']['source_uid']:edges[k,i,j]=bridge(l['R'],r['R'],gap,net,ref)
        arms={'Top4':dict(top4=True),'FullPool':dict()}
        if controls:arms.update(FullPoolWrongMusic=dict(wrong_music=True),Top4WrongMusic=dict(top4=True,wrong_music=True),FullPoolNoExactRepeat=dict(no_repeat=True),LocalFullPool=dict(scoring="local"),HeuristicFullPool=dict(scoring="heuristic"))
        selected={}
        for arm,flags in arms.items():
            result=solve(pools,edges,ref,**flags)
            if result is None:selected[arm]=dict(status='REJECTED');continue
            ids=result['ids'];chunks=[];events=[];connections=[]
            for k,i in enumerate(ids):
                c=pools[k][i];chunks.append(c['R']);events.append(dict(event_uid=c['unit']['unit_uid'],source_uid=c['unit']['source_uid'],source_interval=c['unit']['source_interval'],warp=c['warp']))
                if k<count-1:
                    q,e=edges[k,i,ids[k+1]];chunks.append(q);connections.append(e)
            R=np.concatenate(chunks);name=key+'_'+arm;arrays[name]=relative_fk(R,net.joints,net.parents).astype('<f4');arrays[name+'_R']=R
            selected[arm]=dict(status='NUMERICAL_SHORT_PLAN',**result,events=events,connections=connections,kinematics=regions(R,net.joints,net.parents,spans,origin=spans[0]["interval"][0]))
            if arm=='Top4':assert [v['event_uid'] for v in events]==[v['event_uid'] for v in base['selected']],'Frozen R50 choice mismatch'
        raw_pools=[[{k:v for k,v in c.items() if k!='R'} for c in p] for p in pools]
        edge_rows=[dict(slot=k,left=i,right=j,**e) for (k,i,j),(q,e) in sorted(edges.items())]
        records.append(dict(key=key,song=song,events=count,gap=gap,regions=spans,music_interval=base['music_interval'],pools=raw_pools,catalog_pools=catalog_pools,rejected=rejected,edges=edge_rows,selected=selected))
        print('R51_FUNNEL',key,[len(p) for p in pools],Counter(e['selected'] or 'REJECT' for e in edge_rows),flush=True)
    np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,model=net.binding(),source_bank=[u['unit_uid'] for u in bank],optimizer_steps=0,can_route=False))
    from .view import plans
    plans(out,tracks)


def control_run(out):
    """Reuse the fixed eligibility graph; re-score music and reconstruct selected curves."""
    out=fresh(out);net,lib,units,tracks=assets();reference=read('outputs/bridge_dynamics_r48_01/main/result.json')['reference']
    original=read('outputs/validation_r51_01/plans/result.json');records=[];arrays={'J':net.joints,'parents':net.parents}
    arms={'Top4':dict(top4=True),'FullPool':dict(),'FullPoolWrongMusic':dict(wrong_music=True),'Top4WrongMusic':dict(top4=True,wrong_music=True),'FullPoolNoExactRepeat':dict(no_repeat=True),'LocalFullPool':dict(scoring='local'),'HeuristicFullPool':dict(scoring='heuristic')}
    for r in original['records']:
        cores=[s['interval'] for s in r['regions'] if s['kind']=='core'];pools=[];catalog_pools=[]
        for slot,(a,b) in enumerate(cores):
            track=tracks[(r['song']+1)%3];music=net.music_tokens(track['values'][a:b],track['mask'][a:b]);energy=np.asarray(track['signal']['energy'][a:b]);activity=float(np.mean(track['structure']['energy_percentile'][a:b]));pool=[]
            for c in r['catalog_pools'][slot]:
                w=np.asarray(c['warp']['time_warp']);q=np.interp(np.arange(a,b),w[:,1],w[:,0]);part=main_scores(net,lib.rotations[c['unit']['source_uid']],q,music,energy,activity)
                assert part is not None and part['valid_frames']==c['parts'][0]['valid_frames'],'Keep fixed valid token support'
                pool.append(dict(c,R=rotation_sample(lib.rotations[c['unit']['source_uid']],q),parts=[c['parts'][0],part],scores=[c['scores'][0],part['local']+.1*part['multi']-.05*c['warp_cost']]))
            catalog_pools.append([{k:v for k,v in c.items() if k!='R'} for c in pool])
            registered={c['unit']['unit_uid'] for c in r['pools'][slot]}
            pools.append([c for c in pool if c['unit']['unit_uid'] in registered])
        edges={(e['slot'],e['left'],e['right']):(True if e['selected'] else None,{k:v for k,v in e.items() if k not in ('slot','left','right')}) for e in r['edges']}
        selected={}
        for arm,flags in arms.items():
            result=solve(pools,edges,reference,**flags)
            if result is None:selected[arm]=dict(status='REJECTED');continue
            chunks=[];events=[];connections=[];ids=result['ids']
            for k,i in enumerate(ids):
                c=pools[k][i];chunks.append(c['R']);events.append(dict(event_uid=c['unit']['unit_uid'],source_uid=c['unit']['source_uid'],source_interval=c['unit']['source_interval'],warp=c['warp']))
                if k<len(ids)-1:
                    q,e=bridge(c['R'],pools[k+1][ids[k+1]]['R'],r['gap'],net,reference)
                    assert e==edges[k,i,ids[k+1]][1] and q is not None,'Frozen graph curve differs'
                    chunks.append(q);connections.append(e)
            R=np.concatenate(chunks);name=r['key']+'_'+arm;arrays[name]=relative_fk(R,net.joints,net.parents).astype('<f4');arrays[name+'_R']=R
            selected[arm]=dict(status='NUMERICAL_SHORT_PLAN',**result,events=events,connections=connections,kinematics=regions(R,net.joints,net.parents,r['regions'],origin=r['music_interval'][0]))
            if arm in ('Top4','FullPool'):assert {k:v for k,v in selected[arm].items() if k!='kinematics'}=={k:v for k,v in r['selected'][arm].items() if k!='kinematics'},'Main choice changed; regional evaluation is versioned separately'
        records.append(dict(r,pools=[[{k:v for k,v in c.items() if k!='R'} for c in p] for p in pools],catalog_pools=catalog_pools,selected=selected))
        print('R51_PLAN_CONTROLS',r['key'],flush=True)
    np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(original,records=records,main_graph_reused=True))
    from .view import plans
    plans(out,tracks)
