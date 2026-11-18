"""Physical-clock stage diagnostics. No Oracle labels, semantic positives or optimizer."""
import argparse,collections,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.spatial.transform import Rotation
from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.research.phase_r45.audio import extract
from dunhuang.research.sdmr_event_rank.evaluate import group_ci

def union(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]:result[-1][1]=max(b,result[-1][1])
        else:result.append([a,b])
    return result

def rank_zero(scores):
    values=np.asarray(scores,float);correct=values[2];better=int(np.sum(values>correct+1e-10));tied=int(np.sum(abs(values-correct)<=1e-10));return dict(recall_at1=(1/tied if better==0 else 0.),mrr=float(np.mean(1/(better+np.arange(1,tied+1)))),tie_count=tied)

def stage_score(signal,times,kinds,shift,channel='mix'):
    t=np.asarray(times)+shift
    if len(t)==0 or t.min()<0 or t.max()>len(signal['onset'])-1:return None
    onset=np.interp(t,np.arange(len(signal['onset'])),signal['onset']);beat=np.interp(t,np.arange(len(signal['beat_strength'])),signal['beat_strength']);energy=np.interp(t,np.arange(len(signal['energy'])),signal['energy']);values=[]
    for k,o,b,e in zip(kinds,onset,beat,energy):
        quiet=k in ('deceleration','pose_hold');v=e if channel=='energy' else .7*o+.3*b if k=='speed_peak' else .9*o+.1*b if k=='motion_onset' else .4*o+.6*b
        if quiet:v=1-(e if channel=='energy' else o)
        values.append(v)
    return float(np.mean(values))

def phase_correspondence(A,B,RA,RB,offsetA,offsetB,band,margin):
    matches=[];available=0
    for kind in sorted({(p['type'],p['body_region']) for p in A}):
        aa=[p for p in A if (p['type'],p['body_region'])==kind];bb=[p for p in B if (p['type'],p['body_region'])==kind]
        if not bb:continue
        available+=len(aa);PA=RA[[p['frame'] for p in aa]];PB=RB[[p['frame'] for p in bb]];rel=PA[:,None].swapaxes(-1,-2)@PB[None];cost=Rotation.from_matrix(rel.reshape(-1,3,3)).magnitude().reshape(len(aa),len(bb),22).mean(-1)
        for i,p in enumerate(aa):
            order=np.argsort(cost[i],kind='stable');j=int(order[0]);reverse=int(np.argmin(cost[:,j]));gap=float(cost[i,order[1]]-cost[i,j]) if len(order)>1 else None;ta=float(p['frame']+offsetA);tb=float(bb[j]['frame']+offsetB);mutual=reverse==i;unique=gap is not None and gap>margin
            matches.append(dict(type=kind[0],region=kind[1],source_frame=p['frame'],target_frame=bb[j]['frame'],audio_time_error_frames=abs(ta-tb),pose_error_rad=float(cost[i,j]),nearest_margin_rad=gap,mutual=mutual,unique=unique,clock_consistent=abs(ta-tb)<=band,evidence_state='HIGH_CONFIDENCE_UNCALIBRATED_PROXY' if mutual and unique and abs(ta-tb)<=band else 'UNKNOWN',semantic_truth='UNKNOWN'))
    return dict(matches=matches,source_phases=len(A),target_phases=len(B),type_compatible_source_phases=available,proxy_supported=sum(m['evidence_state']=='HIGH_CONFIDENCE_UNCALIBRATED_PROXY' for m in matches),policy='UNBANDED_POSE_NEAREST_PLUS_RECIPROCAL_UNIQUENESS; CLOCK_USED_ONLY_FOR_DIAGNOSTIC_NOT_NEIGHBOR_SELECTION')

def run(out):
    out=Path(out);root=Path('outputs/manifold_bridge_r47_01')
    if not read(root/'receipts/main_browser_acceptance.json')['passed']:raise ValueError('Effect delivery precedes P2')
    if out.exists():raise ValueError('Fresh phase audit')
    out.mkdir(parents=True);cfg=read('src/dunhuang/research/manifold_bridge_r47/config.json');manifest_path=Path('outputs/paired_event_r41_01/data/manifest.json');manifest=read(manifest_path);events={r['event_uid']:r for r in manifest['rows']};sources={r['source_uid']:r for r in manifest['sources']};pairs=read('outputs/phase_r45_01/qualification/reference_target_audit.json')['pairs'];pairs=[p for p in pairs if p['same_choreography'] and p['shared_audio_support'][1]-p['shared_audio_support'][0]>=90];assert len(pairs)==476
    grouped={};cache={};signals={};audio_refs={};audio_cfg=read('src/dunhuang/research/phase_r45/config.json');metadata=[]
    for pair in pairs:
        a,b=events[pair['source_event_uid']],events[pair['target_event_uid']];assert a['role']==b['role']=='train' and a['paired_clock_verified'] and b['paired_clock_verified'];assert a['choreography_uid']==b['choreography_uid'] and a['audio_sha256']==b['audio_sha256'] and a['raw_motion_sha256']!=b['raw_motion_sha256'];ids=sorted((a['source_uid'],b['source_uid']));key=tuple(ids);grouped.setdefault(key,[]).append(pair['shared_audio_support']);audio_refs[a['audio_sha256']]=sources[a['source_uid']]['audio']
    for audio,ref in sorted(audio_refs.items()):
        p=checked(ref);s=extract(p,audio_cfg);y,sr=sf.read(p,always_2d=True);y=y.mean(1);energy=[]
        for t in range(len(s['onset'])):
            x=y[round(t*sr/30):round((t+1)*sr/30)];energy.append(float(np.sqrt(np.mean(x*x))) if len(x) else 0.)
        s['energy']=(np.array(energy)/max(np.quantile(energy,.95),1e-10)).clip(0,1).tolist();signals[audio]=s
    audios=sorted(signals);rows=[];motion_truth=[]
    for num,(key,intervals) in enumerate(sorted(grouped.items())):
        sa,sb=[sources[u] for u in key];support=union(intervals)
        for s in (sa,sb):
            if s['source_uid'] not in cache:
                p=checked(s['source_arrays']);cache[s['source_uid']]=np.load(p,allow_pickle=False)['R']
        def choose(s):
            return [p for p in s['phases'] if p['frame']+s['audio_offset_frames']-15>=0 and p['frame']+s['audio_offset_frames']+15<len(signals[s['audio_sha256']]['onset']) and any(a<=p['frame']+s['audio_offset_frames']<b and p['evidence_span'][0]+s['audio_offset_frames']>=a and p['evidence_span'][1]+s['audio_offset_frames']<=b for a,b in support)]
        A=choose(sa);B=choose(sb);c=phase_correspondence(A,B,cache[key[0]],cache[key[1]],sa['audio_offset_frames'],sb['audio_offset_frames'],cfg['stage_match_band_frames'],cfg['phase_pose_margin_rad']);motion_truth.append(dict(source_pair=key,source_raw_sha256=[sa['raw_motion_sha256'],sb['raw_motion_sha256']],audio_sha256=sa['audio_sha256'],source_group=sa['source_group'],choreography_uid=sa['choreography_uid'],actual_audio_support=support,phase_audit=c,exact_stage_truth='UNKNOWN'))
        for s,ps in ((sa,A),(sb,B)):
            times=[float(p['frame']+s['audio_offset_frames']) for p in ps];kinds=[p['type'] for p in ps];signal=signals[s['audio_sha256']];donors=[a for a in audios if a!=s['audio_sha256'] and times and min(times)-15>=0 and max(times)+15<len(signals[a]['onset'])];donor=donors[0] if donors else None;rng=np.random.default_rng(cfg['seed']+num);shuffle=list(rng.permutation(kinds));variants={}
            for method,ss,types,channel in [('Actual',signal,kinds,'mix'),('EnergyOnly',signal,kinds,'energy'),('TypeShuffled',signal,shuffle,'mix'),('MusicSwap',signals[donor] if donor else None,kinds,'mix')]:
                if ss is None:variants[method]=dict(evaluable=False,reason='NO_DISTINCT_REAL_AUDIO_SUPPORT');continue
                scores=[stage_score(ss,times,types,shift,channel) for shift in cfg['stage_offset_candidates']];variants[method]=dict(evaluable=bool(scores and all(v is not None for v in scores)),scores=scores)
                if variants[method]['evaluable']:variants[method].update(rank_zero(scores))
            variants['NoMusic']=dict(evaluable=bool(times),scores=[0.]*5,**rank_zero([0.]*5))
            rows.append(dict(source_uid=s['source_uid'],raw_motion_sha256=s['raw_motion_sha256'],audio_sha256=s['audio_sha256'],source_group=s['source_group'],phase_count=len(ps),audio_times=times,types=kinds,audio_donor=donor,variants=variants,label_scope='KNOWN_SOURCE_CLOCK_ZERO_OFFSET_NOT_SEMANTIC_EVENT_MATCH',clock_only_reference_accuracy=1.,clock_only_scope='PHYSICAL_CLOCK_METADATA_LEAKAGE_UPPER_REFERENCE_NOT_CONTENT_MODEL'))
        print('R47_PHASE_PAIR',num+1,len(grouped),len(A),len(B),c['proxy_supported'],flush=True)
    # Repeated shared support and recording reuse must not inflate music-level denominators.
    grouped_scores=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in rows:
        for method,v in row['variants'].items():
            if v['evaluable']:grouped_scores[method][row['audio_sha256']].append((row['raw_motion_sha256'],v['recall_at1']))
    means={}
    for method,groups in grouped_scores.items():
        means[method]={}
        for group,values in groups.items():
            raw=collections.defaultdict(list)
            for uid,v in values:raw[uid].append(v)
            means[method][group]=float(np.mean([np.mean(v) for v in raw.values()]))
    comparisons={}
    for control in ('EnergyOnly','TypeShuffled','MusicSwap'):
        common=sorted(set(means.get('Actual',{}))&set(means.get(control,{})));v=[means['Actual'][g]-means[control][g] for g in common];ci=group_ci(v,cfg['seed'],cfg['bootstrap_draws']);comparisons[control]=dict(music_groups=len(common),mean_improvement=float(np.mean(v)) if v else None,ci95=ci,passed=len(common)>=cfg['minimum_phase_groups'] and ci[0] is not None and ci[0]>0)
    values=list(means.get('Actual',{}).values());ci=group_ci(values,cfg['seed'],cfg['bootstrap_draws']);supported=sum(r['phase_audit']['proxy_supported'] for r in motion_truth);all_phases=sum(r['phase_audit']['source_phases'] for r in motion_truth);fraction=supported/max(all_phases,1);clock_gate=len(values)>=cfg['minimum_phase_groups'] and ci[0] is not None and ci[0]>.2 and all(v['passed'] for v in comparisons.values());write(out/'result.json',dict(source_manifest_sha256=sha(manifest_path),registered_pair_records=len(pairs),independent_pair_keys=len(grouped),music_groups=len(audios),phase_pairs=motion_truth,clock_tasks=rows,recording_balanced_group_scores=means,actual_accuracy_ci95=ci,control_comparisons=comparisons,cycle_proxy_supported=supported,cycle_proxy_phases=all_phases,cycle_proxy_fraction=fraction,necessary_clock_screen=clock_gate,necessary_cycle_screen=fraction>=cfg['minimum_cycle_stage_fraction'],exact_stage_correspondence_verified=False,training_admitted=False,training_stop_reason='STAGE_TRUTH_UNKNOWN; PROXY_CYCLE_AND_CLOCK_SCORES_DO_NOT_CREATE_SEMANTIC_TRUTH',oracle_labels_read=False,optimizer_steps=0,test_tensor_reads=0,human_reviewed=False,can_route=False,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()));print('R47_PHASE_FINISHED',len(grouped),len(audios),clock_gate,fraction,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
