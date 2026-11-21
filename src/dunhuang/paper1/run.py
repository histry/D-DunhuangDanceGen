"""P0 preview first, then shared-bank scoring and fixed-neighbor G1 controls."""
import argparse,subprocess
import numpy as np
from .contracts import *
from .warp_bank import build
from .correspondence import audit
from dunhuang.research.complete_r36.core import rotation_sample,relative_fk
from dunhuang.research.warp_coupling_r53.population import Population,sealed
from dunhuang.research.whole_song_geometry.study import coverage

def preview(out):
    out=fresh(out);pop=Population();parent=Path('outputs/music_structure_r56_02');check_seal_parent=sealed(parent,'controls/result.json');old=read(check_seal_parent)
    parent_arrays=sealed(parent,'controls/arrays.npz');z=np.load(parent_arrays);units=read('outputs/event_coverage_r47_01/index/result.json')['units'];lookup={u['unit_uid']:u for u in units};bank=[lookup[k] for k in pop.old['source_bank']];assert len(bank)==16
    arrays=dict(J=pop.net.joints,parents=pop.net.parents);songs=[];failures=[]
    for song in CFG['songs']:
        record=next(r for r in old['records'] if r['song']==song and r['arm']==CFG['arm']);N=record['frames'];cov=coverage(record['regions'],N)
        event_path=Path(f'outputs/music_events_01/main/{song}/catalog.json');catalog=read(event_path)
        if catalog['audio_sha256']!=pop.tracks[song]['audio_sha256'] or catalog['frames']!=N:raise ValueError('Music Events audio identity/clock differs')
        frozen=check_seal(Path('outputs/music_events_01/main'));assert str(song)+'/catalog.json' in frozen['files']
        Rold=z[f'song{song}_B0_R'];Pold=z[f'song{song}_B0_P'];arrays[f'song{song}_baseline_R']=Rold;arrays[f'song{song}_baseline_P']=Pold
        arrays[f'song{song}_coverage']=cov;slots=[]
        for k,e in enumerate(record['events']):
            A,B=e['target_interval'];rows=[];rejections=[];selected_uid=None
            for u in bank:
                paths,proof=build(u,[A,B],pop.tracks[song]['structure']['beats'])
                if not paths:rejections.append(dict(event_uid=u['unit_uid'],**proof));continue
                for c in paths:
                    body=rotation_sample(pop.lib.rotations[u['source_uid']],c['q']);key=c['uid'];arrays[key]=body;arrays[key+'_q']=np.array(c.pop('q'))
                    c.update(rotation_sha256=array_hash(body),bank_proof=proof,source_R_sha256=array_hash(pop.lib.rotations[u['source_uid']]),
                             alignment=audit(u,c,body,catalog['events']))
                    if u['unit_uid']==e['event_uid'] and c['origin']=='E1W':
                        if c['time_warp']!=e['time_warp']['time_warp'] or not np.array_equal(body,Rold[A:B]):raise ValueError('Frozen selected Core changed')
                        selected_uid=key
                    rows.append(c)
            if selected_uid is None:raise ValueError('Frozen path left shared bank')
            slots.append(dict(slot=k,target_interval=[A,B],selected_baseline_uid=selected_uid,candidates=rows,rejections=rejections,
                              music_events=[v for v in catalog['events'] if A<=v['frame_float']<B]))
            print('P0_SLOT',song,k,len(rows),flush=True)
        # Preserve the entire old curve, including every historical G0/G1 Bridge.
        if not np.array_equal(relative_fk(Rold,pop.net.joints,pop.net.parents).astype('<f4'),Pold):raise ValueError('Frozen baseline FK differs')
        songs.append(dict(song=song,frames=N,regions=record['regions'],slots=slots,baseline_connections=record['connections'],
                          baseline_rotation_sha256=array_hash(Rold),baseline_exact=True,audio=pop.tracks[song],music_catalog_sha256=sha(event_path)))
    result=dict(task='Paper1-SDMR-Integration-P0',config=CFG,songs=songs,parent_result_sha256=sha(check_seal_parent),parent_arrays_sha256=sha(parent_arrays),
                parent_binding=pop.binding,science=binding(),baseline_exact=True,training_updates=0,new_geometry_calls=0,new_planner_runs=0,
                source_bank=pop.old['source_bank'],can_route=False,native24='NOT_EVALUABLE',artistic_quality='NOT_EVALUABLE')
    save_arrays(out/'arrays.npz',arrays);write(out/'result.json',result)
    from .view import build_view
    build_view(out,pop,arrays,result);seal(out);return result

def controls(out,preview_root):
    # Browser acceptance is a phase gate, not a scientific quality label.
    receipt=read(ROOT/'receipts/preview_browser.json');assert receipt['passed'] and receipt['preview_result_sha256']==sha(preview_root/'result.json')
    check_seal(preview_root);p=read(preview_root/'result.json');assert p['science']==binding();out=fresh(out);z=np.load(preview_root/'arrays.npz');pop=Population()
    from .sdmr_adapter import FrozenEncoder,score_executed_bank_shadow,recall_shadow
    from .execute import connection
    from dunhuang.research.gtw_kinematic_baseline.features import ReferenceCost
    net=FrozenEncoder();assert np.array_equal(net.joints,pop.net.joints) and np.array_equal(net.parents,pop.net.parents)
    feature_path=Path('outputs/exposed_full_song_01/music/arrays.npz');inventory=read(feature_path.with_name('inventory.json'));raw=np.load(feature_path)
    frozen_cfg=read('src/dunhuang/research/sdmr_warp_rank/protocol.json');assert sha(feature_path)==frozen_cfg['change_raw50_feature_arrays_sha256']
    refs={};cache={};calls=0;results=[]
    for song in p['songs']:
        s=song['song'];asset=next(x for x in inventory if int(x['case'])==s);assert asset['audio']['sha256']==song['audio']['audio_sha256']
        for slot in song['slots']:
            A,B=slot['target_interval'];T=B-A;rows=slot['candidates'];bodies=[z[r['uid']] for r in rows]
            # Full wave feature support: an absent final partial frame is masked,
            # never extrapolated or labelled as observed.
            music={};masks={}
            for name,donor in [('actual',s),('wrong',(s+1)%3),('zero',s)]:
                values=raw[str(donor)+'__music50'];mask=raw[str(donor)+'__music_mask'];v=np.zeros((T,50),np.float32);m=np.zeros((T,50),bool)
                end=min(B,len(values));n=max(0,end-A)
                if n:v[:n]=values[A:end];m[:n]=mask[A:end]
                if name=='zero':v[:]=0
                music[name]=v;masks[name]=m
            score=score_executed_bank_shadow(net,music,masks,bodies);source_scores=recall_shadow(score,rows);connections=[];gtw=[]
            k=slot['slot'];baseline=z[f'song{s}_baseline_R'];spans=[x['target_interval'] for x in song['slots']]
            for row,R in zip(rows,bodies):
                unit=row['event_uid']
                if unit not in refs:
                    a,b=row['source_interval'];refs[unit]=ReferenceCost(pop.lib.rotations[row['source_uid']][a:b],pop.net.joints,pop.net.parents)
                values,_=refs[unit].score(R,only='GTWLocal');gtw.append(values['GTWLocal']);local=[]
                for side in ('left','right'):
                    if side=='left' and k:lo,hi=spans[k-1];L,Rr=baseline[lo:hi],R
                    elif side=='right' and k+1<len(spans):lo,hi=spans[k+1];L,Rr=R,baseline[lo:hi]
                    else:continue
                    key=identity([array_hash(L),array_hash(Rr)])
                    if key not in cache:
                        if calls>=CFG['max_local_geometry_requests']:raise ValueError('Frozen local G1 budget exhausted')
                        cache[key]=connection(L,Rr,pop.net.joints,pop.net.parents,pop.reference);calls+=1
                    local.append(dict(side=side,cache_key=key,**cache[key]))
                connections.append(local)
            results.append(dict(song=s,slot=k,candidate_uids=[r['uid'] for r in rows],scores=score['scores'],GTWLocal=gtw,
                                source_recall_shadow=source_scores,encodings=score['motion_encoding'],music_details=score['details'],common_support=score['common_support'],connections=connections,
                                candidate_generation_score_blind=True,official_path_changed=False,can_route=False))
            print('P0_SCORES',s,k,len(rows),calls,flush=True)
    result=dict(slots=results,science=binding(),preview_manifest_sha256=sha(preview_root/'manifest.json'),feature_arrays_sha256=sha(feature_path),encoder_binding=net.binding,network_digest=net.check(),
                new_geometry_calls=calls,geometry_requests=sum(len(c) for r in results for c in r['connections']),local_neighbor_scope='ONE_CHANGED_CORE_WITH_FIXED_FROZEN_NEIGHBORS_NOT_FULL_ALTERNATIVE_PATH',
                training_updates=0,new_planner_runs=0,official_path_changed=False,can_route=False,artistic_quality='NOT_EVALUABLE',native24='NOT_EVALUABLE')
    write(out/'result.json',result);seal(out);return result

def replay(stage):
    from dunhuang.research.source_tasks.common import forbid_optimizers
    forbid_optimizers();old=ROOT/stage;new=ROOT/('independent_'+stage);check_seal(old)
    if stage=='preview':preview(new)
    else:controls(new,ROOT/'preview')
    a=read(old/'manifest.json');b=read(new/'manifest.json');assert a==b,'Fresh replay differs'
    write(ROOT/'receipts'/('independent_'+stage+'.json'),dict(status='PASS',fresh_process=True,all_files_exact=True,manifest_sha256=sha(old/'manifest.json'),optimizer_calls=0,can_route=False))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preview','controls','replay-preview','replay-controls']);args=parser.parse_args()
    if args.stage.startswith('replay-'):replay(args.stage[7:])
    elif args.stage=='preview':preview(ROOT/'preview')
    else:controls(ROOT/'controls',ROOT/'preview')

if __name__=='__main__':main()
