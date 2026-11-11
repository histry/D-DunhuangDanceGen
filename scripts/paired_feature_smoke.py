"""Two Train-only extraction candidates; never asserts real-alignment qualification."""
import argparse,sys,json
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import checked,pinned,write,sha
from dunhuang.data.sources import read,ref,raw_path
from dunhuang.data.conversion import decode,contacts,normalized_window
from dunhuang.contracts import MotionAdapter

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',required=True);ap.add_argument('--audit',required=True)
    ap.add_argument('--raw-root',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();assets=read(a.assets);audit=read(Path(a.audit)/'report.json')
    candidates=read(checked(audit['feature_candidates']))['rows']
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    edge=pinned(assets['edge_root'],assets['edge_commit']);sys.path.insert(0,str(edge))
    import torch
    from vis import SMPLSkeleton
    from data.audio_extraction.jukebox_features import extract
    torch.set_num_threads(4);skeleton=SMPLSkeleton('cpu')
    with np.load(checked(assets['scaler'])) as s:scale,bias=s['scale'],s['bias']
    rows=[]
    for index,c in enumerate(candidates):
        if c['role']!='train':raise ValueError('Feature smoke is Train-only')
        folder=out/str(index);folder.mkdir()
        raw=raw_path(a.raw_root,c['raw_motion']);audio=raw_path(a.raw_root,c['raw_audio'])
        native,observed=decode(raw,c['dataset']);native[:,:4]=contacts(MotionAdapter.to_edge(native),skeleton)
        target,_,valid=normalized_window(native,c['start_frame'],scale,bias)
        if not valid.all() or not observed[4:].all():raise ValueError('Representation gate changed')
        # Save absolute native height/root. Training applies the same fixed XZ origin.
        np.save(folder/'motion.npy',target)
        obs=np.broadcast_to(observed,(150,151)).copy();derived=np.zeros_like(obs);derived[:,:4]=True
        supervision=obs|derived
        for key,v in [('observed',obs),('derived',derived),('supervision',supervision)]:np.save(folder/(key+'.npy'),v)
        info=sf.info(audio);begin=round((c['start_frame']/30+c['audio_offset_seconds'])*info.samplerate);end=begin+5*info.samplerate
        with sf.SoundFile(audio) as f:
            if end>len(f):raise ValueError('Real audio crop cannot use padding')
            f.seek(begin);pcm=f.read(end-begin,dtype='float32',always_2d=True).mean(1)
        crop=folder/'audio.wav';sf.write(crop,pcm,info.samplerate,subtype='FLOAT')
        features,_=extract(str(crop),skip_completed=False,dest_dir=str(folder/'extract_cache'))
        features=np.asarray(features,np.float32)
        if features.shape!=(150,4800) or not np.isfinite(features).all() or not features.any():raise ValueError('Jukebox contract differs')
        np.save(folder/'jukebox.npy',features)
        write(folder/'jukebox_receipt.json',dict(window_uid=c['window_uid'],audio_sha256=c['raw_audio']['sha256'],
            crop=ref(crop),sample_start=begin,sample_end=end,sample_rate=info.samplerate,
            layer=66,fps=30,edge_commit=assets['edge_commit'],extractor_sha256=sha(edge/'data/audio_extraction/jukebox_features.py'),
            features_sha256=sha(folder/'jukebox.npy'),silence_padding=False))
        row=dict(window_uid=c['window_uid'],dataset=c['dataset'],source_group=c['source_group'],role='train',
            start_frame=c['start_frame'],fps=30,frames=150,audio_offset_seconds=c['audio_offset_seconds'],
            audio=ref(audio),raw_motion=ref(raw),motion=ref(folder/'motion.npy'),observed_body_joints=24,
            contact_policy='official_edge_forward_displacement_0.01m_last1',
            music_uid='audio:'+c['raw_audio']['sha256'],recording_uid=None,choreography_uid=None,
            alignment_attestation=None,pairing_evidence='release_declared_zero_offset_not_calibrated')
        for key in ('observed','derived','supervision','jukebox','jukebox_receipt'):
            row[key]=ref(folder/(key+('.json' if key=='jukebox_receipt' else '.npy')))
        rows.append(row);print('REAL_JUKEBOX_CANDIDATE',c['window_uid'],flush=True)
    manifest=dict(schema='paired_generation_manifest_v2',checkpoint_sha256=assets['checkpoint']['sha256'],
        scaler_sha256=assets['scaler']['sha256'],recording_independence_verified=False,blind_test_claimed=False,rows=rows)
    write(out/'manifest.json',manifest)
    prepared=dict(assets,paired_generation_manifest=ref(out/'manifest.json'));write(out/'assets.json',prepared)
    from dunhuang.data.paired_manifest import inspect
    _,_,report=inspect(prepared);write(out/'preflight.json',report)
    write(out/'status.json',dict(status='COMPLETED_CANDIDATE_FEATURES_NOT_TRAINING',exit_status=0,
         candidates=len(rows),training_steps=0,paired_ready=report['ready']))
if __name__=='__main__':main()
