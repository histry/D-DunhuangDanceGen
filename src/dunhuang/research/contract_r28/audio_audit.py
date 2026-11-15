"""Authenticate real FineDance waveforms and rebuild the pinned35D interface, without training."""
import argparse
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from dunhuang.research.state_r27.common import read,write,sha,fresh,commit,forbid_optimizers,runtime

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    d,n,s,f,extract,info,a=runtime(read("src/dunhuang/research/state_r27/config.json"));a.close()
    import librosa
    assets=Path(cfg["asset_root"])/"finedance";receipt=read(assets/"asset_receipt.json");records=[]
    for name in cfg["finedance_ids"]:
        wav=assets/"music_wav"/(name+".wav");expected=next(r for r in receipt["records"] if r["path"]=="music_wav/"+name+".wav")
        assert sha(wav)==expected["sha256"];wave_info=sf.info(wav)
        feat,_=extract(str(wav),skip_completed=False);assert feat.ndim==2 and feat.shape[1]==35 and np.isfinite(feat).all()
        original=np.load(assets/"music_npy"/(name+".npy"));length=min(len(feat),len(original))
        assert abs(len(feat)-len(original))<=1 and abs(len(feat)-wave_info.duration*30)<=2
        start=240 if name=="004" else 0;motion=np.load(root/"data/finedance/motion319"/(name+".npy"))
        new=feat[start:].astype(np.float32);count=min(len(motion),len(new));assert abs(len(motion)-len(new))<=2 and count>=256
        path=out/(name+".npy");np.save(path,new)
        records.append(dict(sequence_id=name,waveform_sha256=sha(wav),waveform_frames=wave_info.frames,sample_rate=wave_info.samplerate,channels=wave_info.channels,
            original_feature_sha256=sha(assets/"music_npy"/(name+".npy")),recomputed_feature_sha256=sha(path),
            raw_feature_frames=len(feat),official_trim_frames=start,trimmed_feature_frames=len(new),motion_frames=len(motion),common_prefix_frames=count,
            paired_identity="OFFICIAL_ARCHIVE_SEQUENCE_ID_AND_COMMON_TIMELINE",fps=30,feature_dim=35,
            producer="PINNED_LODGE_extract_CURRENT_LIBROSA_"+librosa.__version__,
            old_vs_recomputed_rms={key:float(np.sqrt(np.mean((feat[:length,lo:hi]-original[:length,lo:hi])**2))) for key,lo,hi in (("onset",0,1),("MFCC",1,21),("chroma",21,33),("onset_binary",33,34),("beat_binary",34,35))},
            physically_measured_contact=False,pretraining_exposure_unknown=True,admitted_for_this_round_training=False))
    result=dict(records=records,official_extractor_sha256=sha("inputs/lodge_r22/official/dld/data/utils/audio.py"),
        authenticated_waveforms=True,recomputed_35D_available=True,old_feature_producer_version_unknown=True,optimizer_steps=0,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,files={p.name:sha(p) for p in out.glob("*.npy")}))
    if replay:
        assert result==read(Path(replay)/"result.json")
        for p in out.glob("*.npy"):assert np.array_equal(np.load(p),np.load(Path(replay)/p.name))
    print("R28_FINE_AUDIO_READY",len(records),librosa.__version__,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");args=p.parse_args()
    run(read(args.config),args.root,args.out,args.replay_from)
