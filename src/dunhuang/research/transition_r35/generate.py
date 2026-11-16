"""Fixed positive and real plan-bound two-boundary previews. No extra counterfactual sweep."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse,json
import numpy as np
import soundfile as sf
import torch
from dunhuang.research.coverage_r34.common import Path,Runtime,forbid_optimizers,fresh,read,sample,population_data,commit,write,same_arrays
from dunhuang.io import checked
from dunhuang.research.coverage_r34.context_contract import context
from dunhuang.research.coverage_r34.context_train import restore_context
from dunhuang.research.coverage_r34.context_eval import paired_metrics
from dunhuang.research.plan_view_r34.common import ref
from dunhuang.research.context_r21.data import same_audio_pcm
from .compiler import compile,make_packets
from .train import targets,cg
from .planning import Sources

def run(cfg,root,out,replay=None):
    root=Path(root);out=fresh(out);forbid_optimizers();rt=Runtime(cfg,context_training=True)
    zero={k:v.detach().clone() for k,v in rt.c.context_branch.state_dict().items()}
    arrays={};records=[];noise=[];calls=0;trajectories=0;sources=Sources()
    arrays["skeleton_J22"]=rt.s.J[0,:22].cpu().numpy();arrays["parents"]=np.array(rt.s.parents[:22])
    def execute(key,music,src,ctx,seeds):
        nonlocal calls,trajectories
        for arm in ("Frozen","Context"):
            if arm=="Frozen":rt.c.context_branch.load_state_dict(zero);condition=None
            else:restore_context(rt,root/"context/context.pt");condition=ctx
            pred,nr,_=sample(rt,music,source_packet=src,context_packet=condition,seeds=seeds)
            noise.append(nr);calls+=1;trajectories+=len(music)
            raw=rt.n.unnormalize(pred.clone())
            with torch.no_grad():p=rt.f(raw,rt.s)[:,:,:22]
            arrays[key+"_"+arm+"_normalized"]=pred.cpu().numpy();arrays[key+"_"+arm+"_139"]=raw.cpu().numpy();arrays[key+"_"+arm+"_22"]=p.cpu().numpy()
            np.savez_compressed(out/"arrays.npz",**arrays)
            write(out/"progress.json",dict(status="IN_PROGRESS",code_commit=commit(),sampler_calls=calls,generated_trajectories=trajectories,noise=noise,last_key=key+"_"+arm))
            print("R35_MAIN_PREVIEW",key,arm,flush=True)
    _,_,data,z=population_data();cfg32=read("src/dunhuang/research/content_r32/config.json")
    paired=[]
    for role in ("train","holdout"):
        cases=cfg32["train_cases"] if role=="train" else cfg32["holdout_cases"];rows,y,music=targets(rt,data,cases)
        key="aist_"+role;execute(key,music,None,context(y,[64,160]),[cfg["seed"]+int(c) for c in cases])
        arrays[key+"_truth139"]=np.stack([r["model_raw"] for r in rows]);arrays[key+"_truth22"]=np.stack([r["model_points"] for r in rows]);arrays[key+"_music35"]=music.cpu().numpy()
        for arm in ("Frozen","Context"):
            for i,c in enumerate(cases):
                paired.append(dict(role=role,arm=arm,case=c,item=i,**paired_metrics(arrays[key+"_"+arm+"_139"][i],arrays[key+"_"+arm+"_22"][i],
                    data[c]["model_raw"],data[c]["model_points"])))
    # The exact R33 two-source probe protects against a new Context damaging the learned Core.
    a=dict(np.load(Path(cfg["r34"])/"data/arrays.npz"));body,points,music,src,ctx=cg(rt,z,a)
    execute("change",music,src,ctx,[cfg["seed"]+i for i in range(4)])
    from dunhuang.research.state_r27.common import metrics
    change=[]
    for arm in ("Frozen","Context"):
        for i in range(4):
            t=np.zeros((256,139));t[:,7:]=a["body"][i%2]
            p=a["local_points"][i%2]
            mask=np.zeros(256,bool);mask[64:160]=True
            change.append(dict(arm=arm,item=i,event=i%2,music=i//2,geometry=metrics(arrays["change_"+arm+"_139"][i],arrays["change_"+arm+"_22"][i],t,p,mask)))
    arrays["change_truth22"]=a["local_points"][[0,1,0,1]];arrays["change_music35"]=music.cpu().numpy()
    # Fix challenging slots by OLD boundary cost, not by candidate video or generation score.
    for song in read(root/"planning/songs.json"):
        oldplan=read(checked(song["old_plan"]));newplan=read(checked(song["plan"]))
        possible=[]
        for i in range(1,len(oldplan["events"])-1):
            from .graph import zones
            legal=True
            for candidate_plan in (oldplan,newplan):
                e=candidate_plan["events"][i];a,b=e["music_interval"];start=a-64;zone=zones(e)
                right=b-zone["exit_edit"][0]
                if start<0 or start+256>candidate_plan["frames"] or b+right+8>start+256:legal=False
            if not legal:continue
            score=sum(song["boundaries"][j]["old"]["cost"] for j in (i-1,i))
            possible.append((score,i))
        if not possible:raise ValueError("No complete fixed two-boundary window")
        _,index=max(possible,key=lambda x:(x[0],-x[1]))
        for plan_arm,plan in (("Old",oldplan),("Graph",newplan)):
            compiled,binding=compile(plan,index,song["audio"]["sha256"]);start=binding["window_start"];key=song["id"]+"_"+plan_arm
            wave,sr=sf.read(checked(song["audio"]),dtype="float32",always_2d=True);lo=round(start/30*sr);hi=lo+round(256/30*sr)
            assert hi<=len(wave);wav=out/(key+".wav");sf.write(wav,wave[lo:hi],sr,subtype="FLOAT")
            music,_=rt.e(str(wav),skip_completed=False);music=np.asarray(music[:256],np.float32);assert music.shape==(256,35)
            src,ctx=make_packets(compiled,binding)
            execute(key,torch.tensor(np.repeat(music[None],2,axis=0),device="cuda"),src,ctx,[cfg["seed"],cfg["seed"]+1])
            reference=[]
            for t in range(start,start+256):
                e=next(e for e in plan["events"] if e["music_interval"][0]<=t<e["music_interval"][1])
                r,p=sources.body(e,[t]);reference.append(p[0])
            arrays[key+"_reference22"]=np.stack(reference);arrays[key+"_music35"]=music
            binding.update(song_id=song["id"],plan_arm=plan_arm,selected_slot=index,key=key,audio_window=ref(wav),audio_start_sample=lo,audio_end_sample=hi,sample_rate=sr,
                context_model=ref(root/"context/context.pt"),frozen_source_sha256=cfg["dyn_checkpoint_sha256"],seeds=[cfg["seed"],cfg["seed"]+1],
                source_domain_status="ADAPTED_ONLY_FEITIAN_PIPA; OTHER_CATALOG_SOURCES_EXPOSED_UNQUALIFIED_TRANSFER")
            records.append(binding)
            if replay:assert same_audio_pcm(wav,Path(replay)/wav.name)
    rt.check();rt.close()
    result=dict(code_commit=commit(),records=records,paired=paired,source_protection=change,noise=noise,sampler_calls=calls,generated_trajectories=trajectories,
        optimizer_steps=0,context_trained=True,additional_ablation_calls=0,hard_observation_override=False,
        complete_three_event_sequences=0,whole_song_generations=0,operational_qualified=False,full_native_status="NOT_EVALUABLE_22_OF24")
    np.savez_compressed(out/"arrays.npz",**arrays);write(out/"result.json",result)
    if replay:
        same_arrays(out/"arrays.npz",Path(replay)/"arrays.npz")
        def canonical(x):
            x=json.loads(json.dumps(x))
            for row in x["records"]:row.pop("audio_window")
            return x
        assert canonical(result)==canonical(read(Path(replay)/"result.json"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args()
    run(read("src/dunhuang/research/transition_r35/config.json"),a.root,a.out,a.replay_from)
