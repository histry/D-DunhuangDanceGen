"""Frozen R41 Native vectors and Train-only query-bank affinity. No label fitting."""
import argparse,collections
from pathlib import Path
import numpy as np
import torch
from dunhuang.research.complete_r36.core import read,write,sha,Library,rotation_sample
from dunhuang.research.transition_r37.clock import source_positions
from dunhuang.research.transition_r37.content import observations
from dunhuang.research.coverage_r34.common import forbid_optimizers
from dunhuang.research.paired_event_r41.population import Population
from dunhuang.research.paired_event_r41.train import digest
from dunhuang.research.sdmr_event_rank.features import load_model
from dunhuang.research.sdmr_event_rank.scorer import R41EventScorer,scores
from dunhuang.research.reperformance_r44.archive import Arrays

METHODS=('Global','Diagonal','Partial','Debias','EnergyHeuristic')

def run(package,out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh frozen Native scores')
    out.mkdir(parents=True);package=Path(package);selection=read(package/'admin/manifest.json');cfg=read('src/dunhuang/research/sdmr_event_rank/config.json');pop=Population(cfg['data']);normal=pop.statistics;native=Path('outputs/sdmr_event_rank_01/change_shadow_v3');meta=read(native/'result.json');z=Arrays(native/'arrays.npz');J=z['skeleton_J22'];parents=z['parents'];lib=Library(read('src/dunhuang/research/complete_r36/config.json')['catalog']);music=Arrays('outputs/exposed_full_song_01/music/arrays.npz');old_eval=read('outputs/retrieval_audit_r44_01/evaluation/result.json');benchmark=read('outputs/retrieval_audit_r44_01/benchmark/manifest.json');fz=Arrays('outputs/retrieval_audit_r44_01/features/arrays.npz');banks=collections.defaultdict(list)
    for ids in old_eval['train_bank_indices'].values():
        for i in ids:
            r=benchmark['rows'][i];assert r['role']=='train' and r['source_group'] in benchmark['calibration_groups'];banks[r['source_group']].append(fz[f'a{i}_mg'])
    assert len(banks)==24;forbid_optimizers();model=load_model(cfg);before=digest(model);service=R41EventScorer(model,cfg,pop.manifest['normalizer']);arrays={};records=[];maxerror=0.;calls=0
    def norm(v,mask,key):return np.where(mask,(v-np.array(normal[key]['mean']))/np.array(normal[key]['std']),0).astype(np.float32)
    for row in meta['records']:
        key=row['key'];A,B=row['audio_frame_interval'];L=B-A;mp=music[str(row['case'])+'__music50'][A:B];mm=music[str(row['case'])+'__music_mask'][A:B];D=[];DM=[]
        for j,c in enumerate(row['candidates']):
            q=source_positions(c,np.arange(A,B));np.testing.assert_array_equal(q,z[key+f'_c{j}_source_positions']);R=rotation_sample(lib.rotations[c['source_uid']],q);v,mask=observations(R,J,parents);mask[0,198:272]=False;D.append(norm(v,mask,'motion'));DM.append(mask)
        with torch.no_grad():
            inputs=[torch.tensor(np.repeat(norm(mp,mm,'music')[None],len(D),axis=0),device='cuda'),torch.tensor(np.stack(D),device='cuda'),torch.tensor(np.stack(DM),device='cuda'),torch.tensor(np.repeat(mm.any(-1)[None],len(D),axis=0),device='cuda')];mg,dg,mt,dt,mv,dv=service.encode_normalized(inputs);calls+=1
        def token(j):return {n:v[j].cpu().numpy() for n,v in [('mg',mg),('dg',dg),('mt',mt),('dt',dt),('mv',mv),('dv',dv)]}|dict(positions=np.arange(6,L-6,dtype=np.int32))
        q=token(0);dg_array=dg.cpu().numpy();group_affinity=np.stack([(np.stack(v)@dg_array.T).mean(0) for v in banks.values()]);correction=np.sort(group_affinity,axis=0)[-10:].mean(0);values=[];old=np.asarray(row['variants']['actual'])
        for j in range(len(D)):
            s,_=scores(q,token(j),cfg,False);err=float(np.max(abs(np.array([s['Global'],s['Diagonal'],s['Offset']])-old[j,:3])));assert err<2e-6;maxerror=max(maxerror,err);energy=-abs(float(inputs[0][0,:,0].mean().cpu())-float(inputs[1][j,:,264].mean().cpu()));values.append([s['Global'],s['Diagonal'],float(old[j,3]),float(2*s['Global']-correction[j]),energy])
        ids=[r['candidate'] for r in selection['selection'] if r['key']==key];assert len(ids)==8 and len({row['candidates'][i]['reference']['sha256'] for i in ids})==8;arrays[key+'_scores']=np.asarray(values);arrays[key+'_mg']=mg.cpu().numpy();arrays[key+'_dg']=dg_array;arrays[key+'_bank_group_affinity']=group_affinity;records.append(dict(key=key,song=row['song'],candidate_source_uids=[c['source_uid'] for c in row['candidates']],candidate_event_uids=[c['event_uid'] for c in row['candidates']],balanced_candidates=ids,prior_selected_candidate=0,prior_in_balanced_pool=0 in ids,partial_coverage=old[:,5].tolist(),query_audio=row['query_audio'],audio_frame_interval=row['audio_frame_interval']));print('R45_FROZEN_SCORE',key,len(D),flush=True)
    assert digest(model)==before;np.savez_compressed(out/'arrays.npz',**arrays);write(out/'result.json',dict(records=records,methods=METHODS,checkpoint_sha256=sha(cfg['checkpoint']),frozen_model_digest=before,encoder_parameters=522632,encoder_forward_calls=calls,global_diagonal_offset_maximum_conformance_error=maxerror,conformance_tolerance=2e-6,train_bank_groups=list(banks),bank_policy='MEAN_AFFINITY_WITHIN_TRAIN_GROUP_OVER_AVAILABLE90_150_240_WINDOWS_THEN_TOP10_GROUPS; FIXED_R44_CALIBRATION_BANK',bank_feature_arrays_sha256=sha('outputs/retrieval_audit_r44_01/features/arrays.npz'),native_metadata_sha256=sha(native/'result.json'),review_manifest_sha256=sha(package/'admin/manifest.json'),energy_heuristic='ABS_DIFFERENCE_NORMALIZED_MUSIC_CHANNEL0_AND_MOTION_CHANNEL264_MEANS_NOT_FULL_PLANNER_COST',optimizer_steps=0,can_route=False,human_reviewed=False,operational_qualified=False));z.close();fz.close();music.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.package,a.out)
