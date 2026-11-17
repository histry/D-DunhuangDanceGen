"""Equal-population content controls, event retrieval, and shadow-only partial paths."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,collections,subprocess
from pathlib import Path
import numpy as np
import torch
from dunhuang.io import sha
from dunhuang.research.complete_r36.core import read,write
from dunhuang.research.bridge_r38.content import OrderedRetrieval
from dunhuang.research.contract_r40.crops import tie_accuracy
from dunhuang.research.coverage_r34.common import forbid_optimizers,same_arrays
from dunhuang.alignment.partial_path import exact_path
from .model import scores_for_offsets,RADIUS
from .train import initial,seeded
from .population import Population
from .view import selected_sources

CONTROLS=('actual','music_zero','both_zero','music_swap','motion_swap')


def metrics(scores,rows,offsets):
    truth=np.broadcast_to(np.arange(5),(len(scores),5));hit=tie_accuracy(scores,truth);ties=np.isclose(scores,scores.max(-1,keepdims=True),atol=1e-7,rtol=0);weights=ties/ties.sum(-1,keepdims=True);o=np.array(offsets);mae=(weights*abs(o[None,None,:]-o[None,:,None])).sum(-1);expect=(weights*o).sum(-1);center=offsets.index(0);nonzero=[i for i in range(5) if i!=center];margins=np.stack([scores[:,i,i]-scores[:,i,center] for i in nonzero],1);groups={}
    for group in sorted({r['source_group'] for r in rows}):
        idx=np.array([r['source_group']==group for r in rows]);groups[group]=dict(units=int(idx.sum()),offset_accuracy=float(hit[idx].mean()),offset_mae_frames=float(mae[idx].mean()),correct_vs_zero_score_margin=float(margins[idx].mean()))
    return dict(units=len(rows),related_trials=len(rows)*5,offset_accuracy=float(hit.mean()),group_macro_accuracy=float(np.mean([g['offset_accuracy'] for g in groups.values()])),offset_mae_frames=float(mae.mean()),correct_vs_zero_score_margin=float(margins.mean()),nonzero_correct_beats_zero=float(np.mean((margins>1e-7)+.5*(abs(margins)<=1e-7))),equivariance_error_frames=float(np.mean(abs((expect-expect[:,center,None])-o[None]))),uniform_offset_chance=.2,groups=groups)


def global_metrics(scores,p,n,rows):
    result={};length=np.array([r['frames'] for r in rows]);audio=np.array([r['audio_sha256'] for r in rows]);activity=np.array([r['motion_activity'] for r in rows])
    for scope in ('all_clock_pairs','same_music_disjoint_time','offline_target_activity_matched'):
        hits=[];chance=[];group=collections.defaultdict(list)
        for j,r in enumerate(rows):
            pool=p[j]|n[j]
            if scope=='same_music_disjoint_time':pool &= audio==audio[j]
            if scope=='offline_target_activity_matched':pool &= (abs(length-length[j])<=max(15,.1*length[j])) & (abs(activity-activity[j])<=max(.2,.2*activity[j]))
            if not (p[j]&pool).any() or not (n[j]&pool).any():continue
            indices=np.flatnonzero(pool);v=scores[j,indices];winner=indices[np.isclose(v,v.max(),atol=1e-7,rtol=0)];value=float(p[j,winner].mean());hits.append(value);chance.append(float(p[j,pool].mean()));group[r['source_group']].append(value)
        result[scope]=dict(queries=len(hits),population_coverage=len(hits)/len(rows),recall_at1=float(np.mean(hits)) if hits else None,uniform_candidate_chance=float(np.mean(chance)) if chance else None,group_macro_recall_at1=float(np.mean([np.mean(v) for v in group.values()])) if group else None)
    return result


def run(data,root,out,replay=None):
    root=Path(root);out=Path(out)
    if out.exists():raise ValueError('Fresh evaluation directory')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/paired_event_r41/config.json');pop=Population(data);main=read(root/'main/result.json');checks=read(root/'checks/result.json');forbid_optimizers();seeded(cfg['seed']);rows=pop.rows
    # Main failure never triggers an undisclosed Val examination of a fitted checkpoint.
    val=[i for i,r in enumerate(rows) if r['role']=='val' and r['context_frames']>=121] if main['admitted'] else checks['fit_ids'];role='EXPOSED_VAL' if main['admitted'] else 'REGISTERED_TRAIN_FIT_ONLY_MAIN_BLOCKED'
    donors={i:min((j for j in val if rows[j]['audio_sha256']!=rows[i]['audio_sha256'] and rows[j]['context_frames']>=121),key=lambda j:rows[j]['event_uid']) for i in val};arrays=dict(indices=np.array(val),offsets=np.array(cfg['offset_candidates']));summary={};models={};model=initial(cfg);checkpoint_path=root/'main/model.pt' if main['admitted'] else root/'checks/train_fit/fit.pt';model.load_state_dict(torch.load(checkpoint_path,map_location='cuda',weights_only=True)['state']);models['R41' if main['admitted'] else 'TrainFitOnly']=model.eval()
    for name,file in [('FrozenR38','outputs/bridge_r38_01/content/model.pt'),('FrozenR40','outputs/contract_r40_01/content/model.pt')]:
        net=OrderedRetrieval().cuda().eval();net.load_state_dict(torch.load(file,map_location='cuda',weights_only=True)['state']);models[name]=net
    def tokens(net,inputs):
        if hasattr(net,'content'):m,d,*_=net.content(*inputs)
        else:_,_,m,d,_=net.encode(*inputs);m=m[:,RADIUS:-RADIUS];d=d[:,RADIUS:-RADIUS]
        return m,d
    offsets=cfg['offset_candidates'];energy_scores=np.zeros((len(val),5,5));clock_prior=np.zeros_like(energy_scores);clock_prior[:,:,offsets.index(0)]=1
    with torch.no_grad():
        for name,net in models.items():
            summary[name]={}
            for control in CONTROLS:
                scores=np.zeros((len(val),5,5));token_accuracy=np.zeros((len(val),5))
                for s in range(0,len(val),8):
                    ids=val[s:s+8]
                    for k,delta in enumerate(offsets):
                        origins=np.array([[15+delta,15]]*len(ids));inputs=pop.batch(ids,origins,control,donors);m,d=tokens(net,inputs);scores[s:s+len(ids),k]=scores_for_offsets(m,d,offsets).cpu().numpy();similarity=(m[:,24:54]@d.transpose(-1,-2)).cpu().numpy();target=np.broadcast_to(np.arange(30,60)+delta-RADIUS,(len(ids),30));token_accuracy[s:s+len(ids),k]=tie_accuracy(similarity,target).mean(-1)
                        if name==next(iter(models)) and control=='actual':
                            for j,i in enumerate(ids):
                                a=inputs[0][j,:,0].cpu().numpy();b=inputs[1][j,:,264].cpu().numpy()
                                for t,shift in enumerate(offsets):
                                    u=a[30:60];v=b[30+shift:60+shift];u=u-u.mean();v=v-v.mean();energy_scores[s+j,k,t]=float(np.mean(u*v)/max(np.sqrt(np.mean(u*u)*np.mean(v*v)),1e-8))
                arrays[name+'_'+control+'_scores']=scores;arrays[name+'_'+control+'_token']=token_accuracy;summary[name][control]=metrics(scores,[rows[i] for i in val],offsets);summary[name][control]['token_clock_recall_at1']=float(token_accuracy.mean());summary[name][control]['shared_token_keys']=78
                print('R41_EVAL',name,control,summary[name][control]['offset_accuracy'],flush=True)
    arrays['EnergyOnly_scores']=energy_scores;arrays['IndexPrior_scores']=clock_prior;summary['EnergyOnly']=metrics(energy_scores,[rows[i] for i in val],offsets);summary['IndexPrior']=metrics(clock_prior,[rows[i] for i in val],offsets)
    global_ids=[i for i,r in enumerate(rows) if r['role']=='val'] if main['admitted'] else val;p,n=pop.relations(global_ids);arrays['global_indices']=np.array(global_ids);arrays['global_positive']=p;arrays['global_negative']=n;global_result={}
    with torch.no_grad():
        for name,net in models.items():
            M=[];D=[]
            for s in range(0,len(global_ids),8):
                m,d,*_=net.encode(*pop.batch(global_ids[s:s+8]));M.append(m.cpu().numpy());D.append(d.cpu().numpy())
            scores=np.concatenate(M)@np.concatenate(D).T;arrays[name+'_global_scores']=scores;global_result[name]=global_metrics(scores,p,n,[rows[i] for i in global_ids])
    duration=np.array([rows[i]['frames'] for i in global_ids]);ds=-abs(duration[:,None]-duration);arrays['DurationOnly_global_scores']=ds;global_result['DurationOnly']=global_metrics(ds,p,n,[rows[i] for i in global_ids])
    train_energy=[]
    for i,r in enumerate(rows):
        if r['role']=='train':
            a,b=r['core_slice'];z=pop.unit(i);train_energy.append([float(z['music'][a:b,0].mean()),float(z['motion'][a:b,264].mean())])
    ee=np.array(train_energy);mu=ee.mean(0);sd=ee.std(0).clip(.0001);ve=[]
    for i in global_ids:
        a,b=rows[i]['core_slice'];z=pop.unit(i);ve.append([float(z['music'][a:b,0].mean()),float(z['motion'][a:b,264].mean())])
    ve=(np.array(ve)-mu)/sd;energy_global=-abs(ve[:,0,None]-ve[None,:,1]);arrays['EnergyOnly_global_scores']=energy_global;global_result['EnergyOnly']=global_metrics(energy_global,p,n,[rows[i] for i in global_ids])
    # Genuine motion-phase proxies are used for strata, never filtering after model scores.
    strata={};model_name=next(iter(models));actual=arrays[model_name+'_actual_scores']
    for flag in ('low_activity_proxy',):
        for value in (True,False):
            idx=[j for j,i in enumerate(val) if rows[i][flag]==value];strata[flag+'_'+str(value)]=metrics(actual[idx],[rows[val[j]] for j in idx],offsets) if idx else dict(units=0,not_evaluable=True)
    demos=[];selected={s['source_uid'] for s in selected_sources(pop.manifest)};demo_ids=[]
    for uid in sorted(selected):
        options=[i for i,r in enumerate(rows) if r['source_uid']==uid and r['context_frames']>=121]
        if options:demo_ids.extend(sorted(options,key=lambda i:rows[i]['event_uid']))
    if not main['admitted']:demo_ids=list(val)
    for i in demo_ids:
        if i not in donors:donors[i]=min((j for j,r in enumerate(rows) if r['role']==rows[i]['role'] and r['audio_sha256']!=rows[i]['audio_sha256'] and r['context_frames']>=121),key=lambda j:rows[j]['event_uid'])
    with torch.no_grad():
        for i in demo_ids:
            origins=np.array([[23,15]]);inputs=pop.batch([i],origins,'actual',donors);m,d=tokens(model,inputs);matrix=(m[0,24:54]@d[0].T).cpu().numpy();scores=scores_for_offsets(m,d,offsets)[0].cpu().numpy();mi=np.arange(30,60);dj=np.arange(6,84);allowed=abs((23+1+mi[:,None])-(15+1+dj[None]))<=cfg['path_band_seconds']*30+1e-6;decoded_path=exact_path(1-matrix,cfg['path_drop_cost'],cfg['path_drop_cost'],allowed)
            demos.append(dict(event_uid=rows[i]['event_uid'],source_uid=rows[i]['source_uid'],role=rows[i]['role'],model=model_name,true_offset=8,predicted_offset=offsets[int(scores.argmax())],heatmap=matrix.tolist(),true_motion_indices=(mi+8-RADIUS).tolist(),scores=scores.tolist(),path=decoded_path,scope='PHYSICAL_RECORDING_CLOCK_DIAGNOSTIC_ONLY; 30 QUERIES / MOTION KEYS6..83; ORANGE CLOCK LABEL NOT INPUT TO CONTENT; STRICT_ONE_TO_ONE_PARTIAL_PATH_NOT_FULL_DROP_DTW'))
    np.savez_compressed(out/'arrays.npz',**arrays);result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),summary=summary,global_retrieval=global_result,strata=strata,evaluation_role=role,main_trained=bool(main['admitted']),scope='事件内容与物理时钟研究评分；没有音乐—动作语义真值，保持Shadow。',demos=demos,source_manifest_sha256=sha(Path(data)/'manifest.json'),checkpoint_sha256=sha(checkpoint_path),control_population_identical=True,energy_calibration=dict(scope='TRAIN_EVENT_SCALARS_ONLY',mean=mu.tolist(),std=sd.tolist()),offline_target_activity_pool_uses_paired_target_metadata_not_deployable=True,partial_path_scope='SHADOW_ONE_TO_ONE_DIAGNOSTIC_WITH_KNOWN_CLOCK_BAND_NOT_ROUTING',original_music_clock_labels_not_inputs=True,synthetic_phase_truth=False,optimizer_steps=0,can_route=False,operational_qualified=False,whole_song_generations=0);write(out/'result.json',result)
    if replay:same_arrays(out/'arrays.npz',Path(replay)/'arrays.npz');old=read(Path(replay)/'result.json');current=read(out/'result.json');current.pop('code_commit');old.pop('code_commit');assert current==old
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.data,a.root,a.out,a.replay_from)
