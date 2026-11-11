"""CPU representation audit, before expensive Jukebox or any optimization."""
import argparse,json,sys,subprocess,time
from pathlib import Path
import numpy as np
from dunhuang.io import checked,pinned,write,sha
from dunhuang.contracts import MotionAdapter
from .sources import read,ref,raw_path,validate_links
from .conversion import decode,contacts,normalized_window

def run(assets,raw_root,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);raw_root=Path(raw_root).resolve()
    population=read(raw_root/'population.json');links=validate_links(population['records'])
    edge=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(edge))
    import torch
    from vis import SMPLSkeleton
    torch.set_num_threads(4);skeleton=SMPLSkeleton('cpu')
    with np.load(checked(assets['scaler'])) as s:scale,bias=s['scale'],s['bias']
    if scale.shape!=(151,) or bias.shape!=(151,) or np.any(scale==0):raise ValueError('Scaler differs')
    ledger=[];supported=[];excess=[];sources=[];eligible={}
    begun=time.monotonic()
    for index,row in enumerate(population['records']):
        uid=row['pair_uid'];dataset=row['dataset_id']
        if row['purpose_role']=='excluded_official_ignore':
            sources.append(dict(pair_uid=uid,dataset=dataset,status='EXCLUDED_OFFICIAL_IGNORE'));continue
        path=raw_path(raw_root,row['motion_asset'])
        native,observed=decode(path,dataset)
        native[:,:4]=contacts(MotionAdapter.to_edge(native),skeleton)
        supervision=observed.copy();supervision[:4]=True # Derived, not observed Contact4.
        source=dict(pair_uid=uid,dataset=dataset,status='DECODED',motion_sha256=sha(path),frames_30hz=len(native),
                    observed_channels=int(observed.sum()),derived_channels=4,
                    missing_channels=np.flatnonzero(~supervision).tolist(),raw_audio_present=False)
        audio=raw_path(raw_root,row['audio_asset'],required=False)
        if audio.exists():checked(dict(path=str(audio),sha256=row['audio_asset']['sha256']));source['raw_audio_present']=True
        sources.append(source)
        for start in range(0,len(native)-149,75):
            x,n,v=normalized_window(native,start,scale,bias)
            _,official,official_valid=normalized_window(native,start,scale,bias,False)
            numerical=bool(v.all());full=numerical and bool(supervision.all())
            rowid=uid+':'+str(start)
            record=dict(window_uid=rowid,pair_uid=uid,dataset=dataset,start_frame=start,
                role=row['purpose_role'],source_group=row['group_uid'],numerical_full151_supported=numerical,
                full151_supervision=bool(supervision.all()),representation_eligible=full,
                official_global_root_supported=bool(official_valid.all()),
                pairing_status=row['offset_status'],pairing_verified=False,audio_offset_seconds=row['audio_offset_seconds'])
            ledger.append(record);supported.append(v.all(0));excess.append(np.maximum(abs(n)-1,0).max(0))
            if full and row['purpose_role']=='train' and uid not in eligible:
                eligible[uid]=dict(record,raw_motion=row['motion_asset'],raw_audio=row['audio_asset'])
        if index%50==0:
            write(out/'progress.json',dict(sources=index+1,windows=len(ledger),seconds=time.monotonic()-begun))
            print('AUDIT',index+1,len(ledger),flush=True)
    support=np.asarray(supported,bool);errors=np.asarray(excess,np.float32)
    np.savez_compressed(out/'support.npz',channel_supported=support,max_normalized_excess=errors)
    write(out/'windows.json',ledger);write(out/'sources.json',sources)
    summary={}
    for dataset in ('aistpp','finedance'):
        selected=[i for i,r in enumerate(ledger) if r['dataset']==dataset]
        rows=[ledger[i] for i in selected];subset=support[selected]
        summary[dataset]=dict(registered_sources=sum(r['dataset']==dataset for r in sources),
            decoded_sources=sum(r['dataset']==dataset and r['status']=='DECODED' for r in sources),windows=len(rows),
            numerical_full151=sum(r['numerical_full151_supported'] for r in rows),
            supervision_full151=sum(r['full151_supervision'] for r in rows),
            representation_eligible=sum(r['representation_eligible'] for r in rows),
            official_global_root_supported=sum(r['official_global_root_supported'] for r in rows),
            unsupported_window_counts=(~subset).sum(0).tolist(),
            worst_normalized_excess=errors[selected].max(0).tolist())
    # Only Train windows for extraction smoke, selected before feature results.
    chosen=[];chosen_groups=set()
    for candidate in sorted(eligible.values(),key=lambda r:r['window_uid']):
        if candidate['source_group'] in chosen_groups:continue
        chosen.append(candidate);chosen_groups.add(candidate['source_group'])
        if len(chosen)==2:break
    write(out/'feature_candidates.json',dict(rows=chosen,pairing_verified=False,
        scope='Train-only release-declared alignment candidate smoke; not a paired training manifest'))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    report=dict(schema='paired_representation_audit_v1',code_commit=commit,raw_root=str(raw_root),
        population=ref(raw_root/'population.json'),scaler=assets['scaler'],checkpoint=assets['checkpoint'],
        edge_commit=assets['edge_commit'],conversion_sha256=sha(Path(__file__).with_name('conversion.py')),
        motion_adapter_sha256=sha(Path(__file__).resolve().parents[1]/'contracts.py'),links=links,datasets=summary,windows=len(ledger),
        support=ref(out/'support.npz'),window_ledger=ref(out/'windows.json'),source_ledger=ref(out/'sources.json'),
        feature_candidates=ref(out/'feature_candidates.json'),clipped_targets=0,normalizer_refitted=False,
        motion_origin='training: first window XZ, preserve absolute height; reference: official uncentered',
        contacts='derived official EDGE 3D forward foot displacement <0.01m; terminal frame=1; not measured labels',
        missing_finedance='body22 only, joints22/23 identity placeholders with false supervision',
        paired_training_ready=False,music_qualified=False,full_motion_qualified=False,
        blockers=['Release-declared zero audio offsets not independently calibrated',
                  'Recording/choreography independence unverified; retain historical exposed roles',
                  'No verified paired Jukebox training manifest'],optimizer_steps=0)
    write(out/'report.json',report);write(out/'status.json',dict(status='COMPLETED_REPRESENTATION_AUDIT_NOT_TRAINING',exit_status=0))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--raw-root',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args()
    try:run(read(a.assets),a.raw_root,a.out)
    except Exception:
        import traceback
        p=Path(a.out)
        if p.exists():write(p/'error.json',dict(error=traceback.format_exc(),training=False))
        raise
