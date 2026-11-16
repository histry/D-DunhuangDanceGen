"""Frozen Data V3 source roles/clocks -> variable-length real pairs. No ChangE music pseudo-pairs."""
import argparse,collections,subprocess
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
from dunhuang.io import checked
from dunhuang.contracts import matrices
from .content import observations

CONFIG=dict(source='outputs/transfer_audit_01/a2',lengths=[120,240,360],positions_per_source_length=3,
            updates=80,batch_music_groups=4,units_per_group=2,seed=37041,width=96,learning_rate=.0002,reconstruction_weight=.1,
            validation_scope='HISTORICALLY_EXPOSED_DATA_V3_VAL_NOT_NEW_BLIND',can_route=False)

def run(out):
    out=Path(out)
    if out.exists():raise ValueError('Fresh paired population required')
    out.mkdir(parents=True);(out/'units').mkdir();source=Path(CONFIG['source']);published=read('reports/transfer_audit_summary.json')['evidence']
    for name in ('manifest.json','source_seals.json'):
        if sha(source/name)!=published['a2/'+name]:raise ValueError('Frozen Data V3 source lineage differs')
    original=read(source/'manifest.json')['rows'];byraw={}
    for row in original:byraw.setdefault(row['raw_motion']['sha256'],row)
    z=np.load('outputs/transition_r35_01/generation/arrays.npz');J=z['skeleton_J22'];parents=z['parents'];rows=[];cache={};roles={};native_roles={};seals=[]
    moments={k:dict(sum=np.zeros(n),square=np.zeros(n),count=np.zeros(n)) for k,n in [('music',50),('motion',274)]}
    for i,sealed in enumerate(read(source/'source_seals.json')):
        role=sealed['role']
        if role not in ('train','val'):raise ValueError('Test tensor access forbidden')
        raw=sealed['raw_motion']['sha256'];r=byraw[raw];audio_sha=sealed['audio']['sha256']
        if roles.setdefault(audio_sha,role)!=role or native_roles.setdefault(raw,role)!=role:raise ValueError('Source/audio crosses roles')
        cal=read(checked(sealed['source_clock']));c=cal['calibration']
        if not c['verified'] or not c['origin_anchor_used'] or c['maximum_residual_seconds']>1/30:raise ValueError('Unverified physical clock')
        if cal['raw_motion']['sha256']!=raw or cal['raw_audio']['sha256']!=audio_sha or c['audio_offset_seconds']!=r['audio_offset_seconds']:raise ValueError('Clock binding differs')
        native=np.load(checked(sealed['native']));R=matrices(native[:,7:139].reshape(-1,22,6));values,mask=observations(R,J,parents)
        if audio_sha not in cache:
            file=source/'audio'/(audio_sha+'.npz');other=Path(str(source)+'_replay')/'audio'/file.name
            if sha(file)!=sha(other):raise ValueError('Audio feature replay differs')
            a=np.load(file);cache[audio_sha]=(a['values'].copy(),a['mask'].copy());seals.append(dict(audio_sha256=audio_sha,feature_sha256=sha(file)))
        av,am=cache[audio_sha];offset=30*r['audio_offset_seconds']
        for n in CONFIG['lengths']:
            lo=max(0,int(np.ceil(-offset)));hi=min(len(R)-n,int(np.floor(len(av)-1-offset-(n-1))))
            if hi<lo:continue
            for start in sorted(set(np.rint(np.linspace(lo,hi,CONFIG['positions_per_source_length'])).astype(int))):
                ticks=start+offset+np.arange(n);a=np.floor(ticks).astype(int);b=np.ceil(ticks).astype(int);alpha=(ticks-a)[:,None]
                music=(av[a]*(1-alpha)+av[b]*alpha).astype(np.float32);mm=am[a]&am[b]
                x=values[start:start+n].copy();xm=mask[start:start+n].copy();x[:,-2]=np.linspace(0,1,n);x[:,-1]=n/30
                uid=raw+':'+str(start)+':'+str(n);file=out/'units'/(str(len(rows))+'.npz');np.savez_compressed(file,music=music,music_mask=mm,motion=x,motion_mask=xm)
                rows.append(dict(unit_uid=uid,path=str(file),sha256=sha(file),role=role,source_group=r['source_group'],recording_uid=r['recording_uid'],
                                 raw_motion_sha256=raw,audio_sha256=audio_sha,audio_frame_interval=[float(start+offset),float(start+offset+n)],
                                 source_interval=[int(start),int(start+n)],paired_clock_verified=True,source_clock=sealed['source_clock'],frames=n))
                if role=='train':
                    for name,value,valid in (('music',music,mm),('motion',x,xm)):
                        m=moments[name];v=np.where(valid,value,0).astype(float);m['sum']+=v.sum(0);m['square']+=(v*v).sum(0);m['count']+=valid.sum(0)
        if i%100==0:print('R37_CONTENT_PREPARE',i+1,len(rows),flush=True)
    normalization={}
    for k,m in moments.items():
        count=np.maximum(m['count'],1);mean=m['sum']/count;std=np.sqrt(np.maximum(m['square']/count-mean*mean,1e-4))
        normalization[k]=dict(mean=mean.tolist(),std=std.tolist(),observed_count=m['count'].tolist())
    train_groups={r['source_group'] for r in rows if r['role']=='train'};val_groups={r['source_group'] for r in rows if r['role']=='val'}
    if train_groups&val_groups:raise ValueError('Known music-work grouping crosses roles')
    write(out/'normalization.json',dict(scope='TRAIN_ONLY_SELECTED_REAL_VARIABLE_UNITS_WITH_OBSERVED_MASKS',statistics=normalization))
    result=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=CONFIG,rows=rows,
                source_manifest_sha256=sha(source/'manifest.json'),source_seals_sha256=sha(source/'source_seals.json'),audio_features=seals,
                counts=dict(collections.Counter(r['role'] for r in rows)),train_groups=len(train_groups),val_groups=len(val_groups),
                content_contract='BODY22_POSE_FK_DYNAMICS274; CONTIGUOUS_WINDOWS120_240_360_NOT_SEMANTICALLY_COMPLETE_EVENTS',human_event_truth=False,
                targets='VERIFIED_RECORDING_CLOCK_NOT_ARTISTIC_COMPATIBILITY',normalizer_sha256=sha(out/'normalization.json'),optimizer_steps=0,operational_qualified=False)
    write(out/'manifest.json',result);print('R37_CONTENT_POPULATION',result['counts'],len(train_groups),len(val_groups),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
