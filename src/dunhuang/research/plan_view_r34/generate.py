"""Actual plan-bound research windows; zero context is disclosed, never a claim of learned bridges."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import numpy as np
import soundfile as sf
import torch
from .common import *
from .compiler import compile_window,packets,choose
from dunhuang.research.coverage_r34.common import Runtime,restore,sample
from dunhuang.research.coverage_r34.common import read as readcfg
from dunhuang.research.context_r21.data import same_audio_pcm

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers()
    songs=read(root/"planning/songs.json");rt=Runtime(cfg);initial={k:v.detach().clone() for k,v in rt.c.change.state_dict().items()}
    arrays={};records=[];calls=0;noise=[];seeds=[cfg["seed"],cfg["seed"]+1]
    for song in songs:
        plan=read(checked(song["plan"]));index=choose(plan);compiled,binding=compile_window(plan,index,song["audio"]["sha256"])
        start=binding["window_start"];wave,sr=sf.read(checked(song["audio"]),dtype="float32",always_2d=True)
        lo=round(start/30*sr);hi=lo+round(256/30*sr);assert 0<=lo<hi<=len(wave)
        file=out/f'{song["id"]}_window.wav';sf.write(file,wave[lo:hi],sr,subtype="FLOAT")
        music,_=rt.e(str(file),skip_completed=False);music=np.asarray(music[:256],np.float32)
        assert music.shape==(256,35) and np.isfinite(music).all()
        arrays[song["id"]+"_music35"]=music
        arrays[song["id"]+"_source_body"]=compiled["body"];arrays[song["id"]+"_context_body"]=compiled["context_body"]
        reference=torch.zeros(1,len(compiled["body"]),139,device="cuda");reference[0,:,7:]=torch.tensor(compiled["body"],device="cuda")
        with torch.no_grad():reference22=rt.f(reference,rt.s)[0,:,:22].cpu().numpy()
        arrays[song["id"]+"_source22"]=reference22-reference22[:,:1]
        arrays["skeleton_J22"]=rt.s.J[0,:22].cpu().numpy();arrays["parents"]=np.array(rt.s.parents[:22])
        ref,ctx=packets(compiled,binding)
        for arm in ("Frozen","Coverage"):
            if arm=="Frozen":rt.c.change.load_state_dict(initial)
            else:restore(rt,root/"models/Coverage/adapter.pt")
            pred,nr,_=sample(rt,torch.tensor(np.repeat(music[None],2,axis=0),device="cuda"),source_packet=ref,context_packet=ctx,seeds=seeds)
            calls+=1;noise.append(nr);raw=rt.n.unnormalize(pred.clone())
            with torch.no_grad():p=rt.f(raw,rt.s)[:,:,:22]
            arrays[song["id"]+"_"+arm+"_normalized"]=pred.cpu().numpy();arrays[song["id"]+"_"+arm+"_139"]=raw.cpu().numpy();arrays[song["id"]+"_"+arm+"_22"]=p.cpu().numpy()
            print("R34_PLAN_GENERATION",song["id"],index,arm,flush=True)
        binding.update(song_id=song["id"],selected_slot=index,seeds=seeds,music_window=ref_file(file),music_feature_sha256=__import__("hashlib").sha256(music.tobytes()).hexdigest(),
            audio_sample_start=lo,audio_sample_end=hi,sample_rate=sr,source_audio_sha256=song["audio"]["sha256"],sample_start_error_seconds=lo/sr-start/30,
            frozen_model_sha256=cfg["dyn_checkpoint_sha256"],candidate_model_sha256=sha(root/"models/Coverage/adapter.pt"),
            source_control_verified=True,context_parameters_zero=True)
        records.append(binding);rt.check()
        if replay:assert same_audio_pcm(file,Path(replay)/file.name)
    rt.close()
    assert calls==6
    np.savez_compressed(out/"arrays.npz",**arrays)
    # File paths differ between fresh runs; identities, clocks, features and motion must match.
    result=dict(records=records,noise=noise,sampler_calls=6,generated_trajectories=12,optimizer_steps=0,
        context_trained=False,full_song_generations=0,adjacent_event_connection_qualified=False,full_native_status="NOT_EVALUABLE_22_OF24",operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=cfg,planning=ref_file(root/"planning/songs.json"),arrays_sha256=sha(out/"arrays.npz")))
    if replay:
        left=json.dumps(result,sort_keys=True).replace(str(out.resolve()),"<RUN>");right=json.dumps(read(Path(replay)/"result.json"),sort_keys=True).replace(str(Path(replay).resolve()),"<RUN>");assert left==right
        same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")

def ref_file(p):
    from dunhuang.data.sources import ref
    return ref(p)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(readcfg(a.config),a.root,a.out,a.replay_from)
