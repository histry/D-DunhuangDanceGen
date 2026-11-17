"""Data V3 source seals -> continuous, region-aware paired event candidates."""
import argparse,collections,subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha
from dunhuang.contracts import matrices
from dunhuang.research.complete_r36.core import read,write,identity
from dunhuang.research.transition_r37.content import observations
from .events import profiles,phases,segment,validate_roles


def run(out,replay=None):
    out=Path(out)
    if out.exists():raise ValueError('Fresh event population required')
    out.mkdir(parents=True);(out/'units').mkdir();(out/'sources').mkdir();cfg=read('src/dunhuang/research/paired_event_r41/config.json');source=Path(cfg['source']);published=read('reports/transfer_audit_summary.json')['evidence']
    for name in ('manifest.json','source_seals.json'):
        if sha(source/name)!=published['a2/'+name]:raise ValueError('Frozen source lineage differs')
    byraw={}
    for r in read(source/'manifest.json')['rows']:byraw.setdefault(r['raw_motion']['sha256'],r)
    seals=read(source/'source_seals.json');validate_roles([dict(raw_motion_sha256=s['raw_motion']['sha256'],audio_sha256=s['audio']['sha256'],source_group=byraw[s['raw_motion']['sha256']]['source_group'],role=s['role']) for s in seals]);old=read(cfg['r37_manifest']);normal=Path(cfg['normalizer'])
    if sha(normal)!=old['normalizer_sha256']:raise ValueError('Train normalizer changed')
    skeleton=np.load('outputs/transition_r35_01/generation/arrays.npz');J=skeleton['skeleton_J22'];parents=skeleton['parents'];np.savez_compressed(out/'skeleton.npz',J=J,parents=parents)
    audio={};records=[];rows=[];exclusions=[];choreography=collections.defaultdict(set)
    for k,sealed in enumerate(seals):
        role=sealed['role']
        if role not in ('train','val'):raise ValueError('Test source access forbidden')
        raw=sealed['raw_motion']['sha256'];r=byraw[raw];ak=sealed['audio']['sha256'];cal=read(checked(sealed['source_clock']));c=cal['calibration']
        if not c['verified'] or not c['origin_anchor_used'] or c['maximum_residual_seconds']>1/30 or cal['raw_motion']['sha256']!=raw or cal['raw_audio']['sha256']!=ak or c['audio_offset_seconds']!=r['audio_offset_seconds']:raise ValueError('Clock identity or calibration differs')
        checked(sealed['raw_motion']);native=np.load(checked(sealed['native']));R=matrices(native[:,7:139].reshape(-1,22,6));v,vm=observations(R,J,parents);vm[0,198:272]=False;profile=profiles(R,native[:,4:7]);phase=phases(profile,cfg)
        if ak not in audio:
            af=source/'audio'/(ak+'.npz');af_replay=Path(cfg['source_replay'])/'audio'/af.name
            if sha(af)!=sha(af_replay):raise ValueError('Audio feature replay differs')
            z=np.load(af);a=checked(sealed['audio']);info=sf.info(a);audio[ak]=(z['values'].copy(),z['mask'].copy(),info,sha(af))
        av,am,info,feature_sha=audio[ak];offset=30*r['audio_offset_seconds'];lo=max(0,int(np.ceil(-offset)));hi=min(len(R),int(np.floor(len(av)-1-offset))+1,int(np.floor(info.frames/info.samplerate*30-offset)))
        if hi-lo<cfg['minimum_core_frames']:
            exclusions.append(dict(recording_uid=r['recording_uid'],reason='INSUFFICIENT_REAL_SHARED_SUPPORT',role=role,interval=[lo,hi]));continue
        cores,evidence=segment(profile,phase,cfg,[lo,hi]);uid=identity(dict(raw=raw,algorithm='whole_record_regions_r41_v1'));source_file=out/'sources'/(uid+'.npz');np.savez_compressed(source_file,R=R,root=native[:,4:7],**{name:value for name,value in profile.items() if name!='omega'})
        source_row=dict(source_uid=uid,recording_uid=r['recording_uid'],raw_motion_sha256=raw,audio_sha256=ak,role=role,source_group=r['source_group'],choreography_uid=r['choreography_uid'],source_frames=len(R),shared_source_interval=[lo,hi],audio_offset_frames=float(offset),audio=sealed['audio'],native=sealed['native'],source_clock=sealed['source_clock'],features_sha256=feature_sha,source_arrays=dict(path=str(source_file),sha256=sha(source_file)),phases=phase,boundary_evidence=evidence,events=[])
        for a,b in cores:
            L=max(lo,a-cfg['context_frames']);H=min(hi,b+cfg['context_frames']);ticks=L+offset+np.arange(H-L);i=np.floor(ticks).astype(int);j=np.ceil(ticks).astype(int);alpha=(ticks-i)[:,None];music=(av[i]*(1-alpha)+av[j]*alpha).astype(np.float32);mm=am[i]&am[j];motion=v[L:H].copy();mask=vm[L:H].copy();motion[:,-2]=np.linspace(0,1,H-L);motion[:,-1]=(b-a)/30
            eu=identity(dict(source=uid,interval=[a,b],algorithm='whole_record_regions_r41_v1'));file=out/'units'/(eu+'.npz');np.savez_compressed(file,music=music,music_mask=mm,motion=motion,motion_mask=mask)
            anchors=[dict(q,evidence_truncated_by_core=q['evidence_span'][0]<a or q['evidence_span'][1]>b) for q in phase if a<=q['frame']<b]
            row=dict(event_uid=eu,source_uid=uid,recording_uid=r['recording_uid'],source_dataset='AIST++',raw_motion_sha256=raw,audio_sha256=ak,source_group=r['source_group'],role=role,choreography_uid=r['choreography_uid'],source_interval=[a,b],context_interval=[L,H],core_slice=[a-L,b-L],frames=b-a,context_frames=H-L,audio_frame_interval=[float(a+offset),float(b+offset)],context_audio_frame_interval=[float(L+offset),float(H+offset)],audio_pairing='VERIFIED_RECORDING_CLOCK_ONLY',paired_clock_verified=True,source_clock=sealed['source_clock'],phase_anchors=anchors,motion_activity=float(profile['global'][a:b].mean()),audio_activity=float(np.sqrt(np.mean(music[a-L:b-L][mm[a-L:b-L]]**2))),low_activity_proxy=bool(np.mean(profile['global'][a:b])<.2),path=str(file),sha256=sha(file),observed_body22=True,root_scope='OBSERVED_FITTED_SOURCE_TRANSLATION_NOT_CONTACT_TRUTH',contact_truth='UNKNOWN',semantic_completeness='UNKNOWN',music_motion_phase_truth='UNKNOWN',human_reviewed=False,can_route=False,full_record_derivative=True,boundary_flags=dict(source_start=a==lo,source_end=b==hi,over_preferred_max=b-a>cfg['maximum_core_frames']),label_scope='PHYSICAL_CLOCK_PAIR_WITH_KINEMATIC_PROXY_PHASES')
            rows.append(row);source_row['events'].append(eu)
        records.append(source_row);choreography[r['choreography_uid']].add(role)
        if (k+1)%100==0:print('R41_P0',k+1,len(rows),flush=True)
    isolation=validate_roles(rows);groups={role:sorted({x['source_group'] for x in records if x['role']==role}) for role in ('train','val')};view=[]
    for role in ('train','val'):
        for group in groups[role][:cfg['viewer_groups_per_role']]:view.append(min((x for x in records if x['source_group']==group and x['role']==role),key=lambda x:x['raw_motion_sha256'])['source_uid'])
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,source_manifest_sha256=sha(source/'manifest.json'),source_seals_sha256=sha(source/'source_seals.json'),normalizer=dict(path=str(normal),sha256=sha(normal)),rows=rows,sources=records,exclusions=exclusions,role_counts=dict(collections.Counter(r['role'] for r in rows)),source_counts=dict(collections.Counter(r['role'] for r in records)),music_groups={k:len(v) for k,v in groups.items()},role_isolation=isolation,shared_choreography_across_roles=[k for k,v in choreography.items() if len(v)>1],viewer_source_uids=view,viewer_selection='METADATA_SORTED_FIRST_SOURCE_IN_FIRST_SIX_MUSIC_GROUPS_PER_ROLE_BEFORE_TRAINING',derivatives='COMPUTED_ON_FULL_RECORD_BEFORE_ALL_EVENT_CUTS',event_label_scope='KINEMATIC_CANDIDATE_NOT_SEMANTICALLY_COMPLETE_PRIMITIVE',test_tensor_reads=0,human_event_truth=False,optimizer_steps=0,can_route=False,operational_qualified=False,finedance_training=False)
    write(out/'manifest.json',result)
    if replay:
        old=read(Path(replay)/'manifest.json');assert result['viewer_source_uids']==old['viewer_source_uids'];assert [x['event_uid'] for x in rows]==[x['event_uid'] for x in old['rows']]
        for row,prior in zip(rows,old['rows']):
            a=np.load(row['path']);b=np.load(prior['path']);assert set(a.files)==set(b.files)
            for key in a.files:assert np.array_equal(a[key],b[key]),(row['event_uid'],key)
        assert result['role_counts']==old['role_counts'] and result['role_isolation']==old['role_isolation']
    print('R41_P0_COMPLETE',result['role_counts'],result['source_counts'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--replay-from');a=p.parse_args();run(a.out,a.replay_from)
