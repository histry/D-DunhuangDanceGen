"""Display-only bundle with true music, fixed plan bindings and unmodified model output."""
import argparse,base64,gzip,json,hashlib,shutil
import numpy as np
import soundfile as sf
from dunhuang.research.plan_view_r34.common import *
from .planning import Sources
from dunhuang.planning.event_plan import semantic_plan_id

def run(cfg,root,out):
    root=Path(root);out=fresh(out);(out/"audio").mkdir();src=Sources()
    z=np.load(root/"generation/arrays.npz");meta=read(root/"generation/result.json");tracks={};sounds={};cases=[]
    def track(p):
        p=np.ascontiguousarray(p,dtype="<f4");key=hashlib.sha256(p.tobytes()).hexdigest()
        tracks.setdefault(key,dict(shape=list(p.shape),data=base64.b64encode(p.tobytes()).decode()))
        return key
    def sound(p,name):
        h=sha(p)
        if h not in sounds:
            target=out/"audio"/name;shutil.copyfile(p,target);sounds[h]="audio/"+name
        return sounds[h]
    def wave(p):
        x,_=sf.read(p,dtype="float32",always_2d=True);x=abs(x.mean(1));b=np.linspace(0,len(x),801,dtype=int)
        return [float(x[a:c].max()) if a<c else 0. for a,c in zip(b[:-1],b[1:])]
    def full(plan):
        return np.concatenate([src.body(e,np.arange(*e["music_interval"]))[1] for e in plan["events"]])
    def diagnosis(p,q):
        rel=p-p[:,:1];truth=q-q[:,:1]
        return dict(source_error=np.sqrt(np.mean((rel-truth)**2,axis=(1,2))).tolist(),
            body_jerk=[None]*3+np.linalg.norm(np.diff(rel,n=3,axis=0)*30**3,axis=-1).mean(-1).tolist(),
            body_D2=[None]*2+np.linalg.norm(np.diff(rel,n=2,axis=0)*900,axis=-1).mean(-1).tolist(),
            root_speed=[None]+np.linalg.norm(np.diff(p[:,0],axis=0)*30,axis=-1).tolist(),
            feet_speed=[None]+np.linalg.norm(np.diff(p[:,[7,8,10,11]],axis=0)*30,axis=-1).mean(-1).tolist())
    for song in read(root/"planning/songs.json"):
        old=read(checked(song["old_plan"]));new=read(checked(song["plan"]));oldp=full(old);newp=full(new);generations={}
        bindings=[r for r in meta["records"] if r["song_id"]==song["id"]]
        for r in bindings:
            key=r["key"];p=z[key+"_Context_22"][0];q=z[key+"_reference22"]
            assert np.max(abs(q-(oldp if r["plan_arm"]=="Old" else newp)[r["window_start"]:r["window_start"]+256]))<2e-5
            generations[r["plan_arm"]]=dict(binding=r,start=r["window_start"],frames=256,Frozen=track(z[key+"_Frozen_22"][0]),Context=track(p),
                diagnostics=diagnosis(p,q),selected_slot=r["selected_slot"],source_status="SELECTED_SOURCE_NOT_AUTOMATICALLY_ADAPTED")
        alternatives=[]
        for slot in song["slots"]:
            values=[]
            for e in slot["candidates"]:
                p=src.body(e,np.arange(*e["music_interval"]))[1]
                values.append(dict(event_uid=e["event_uid"],source_uid=e["source_uid"],track=track(p),start=e["music_interval"][0],
                    candidate_uid=e["candidate_uid"],time_warp=e["time_warp"],music_score=e["display_scores"]["heuristic"]))
            alternatives.append(values)
        cases.append(dict(id=song["id"],name=song["name"]+" · 新旧完整计划",type="plan",frames=song["frames"],fps=30,
            audio=sound(checked(song["audio"]),song["name"]),audio_provenance=song["audio"],beats=song["beats"],waveform=song["waveform"],
            Old=track(oldp),Graph=track(newp),old_events=[{k:e[k] for k in ("source_uid","time_warp","music_interval","primitive")} for e in old["events"]],boundaries=song["boundaries"],slots=song["slots"],alternatives=alternatives,generations=generations,
            initial_slot=bindings[0]["selected_slot"],details=dict(old_plan=song["old_plan"],new_plan=song["plan"],semantic_id=semantic_plan_id(new),Root="UNKNOWN",contact="UNKNOWN")))
    population=read("outputs/reference_r26_01/data/result.json")["records"]
    configs=read("src/dunhuang/research/content_r32/config.json")
    for role in ("train","holdout"):
        for i,c in enumerate(configs["train_cases"] if role=="train" else configs["holdout_cases"]):
            key="aist_"+role;q=z[key+"_truth22"][i];r=next(x for x in population if x["case"]==c);audio=Path("outputs/reference_r26_01/data")/r["crop_audio"]
            cases.append(dict(id=key+"_"+str(c),name=f"AIST {role} · case{c}",type="paired",frames=256,fps=30,audio=sound(audio,f"aist_{c}.wav"),audio_provenance=dict(source=r["audio"],clock=r["clock"],source_start=r["source_start"],crop_sha256=sha(audio)),
                waveform=wave(audio),beats=[],Old=track(q),Graph=track(q),boundaries=[],slots=[],generations=dict(study=dict(start=0,frames=256,Frozen=track(z[key+"_Frozen_22"][i]),Context=track(z[key+"_Context_22"][i]),diagnostics=diagnosis(z[key+"_Context_22"][i],q),selected_slot=0,binding=dict(case=c,gap=[64,160],observations=[[56,64],[160,168]],truth="CONTINUOUS_AIST",role="HISTORICAL_EXPOSED_DEVELOPMENT_HOLDOUT" if role=="holdout" else "TRAIN_POSITIVE"))),
                initial_slot=0,details=dict(case=c,paired_truth=True,contact="HEIGHT_PROXY")))
    for i in range(4):
        m=i//2;e=i%2;audio=Path(f"outputs/lodge_reference_r24_01/main/music{m}.wav");q=z["change_truth22"][i]
        cases.append(dict(id=f"change{i}",name=f"来源保持 · {'飞天' if e==0 else '琵琶'} × 音乐{m}",type="source",frames=256,fps=30,audio=sound(audio,f"change_music{m}.wav"),
            waveform=wave(audio),beats=[],Old=track(q),Graph=track(q),boundaries=[],slots=[],generations=dict(study=dict(start=0,frames=256,Frozen=track(z["change_Frozen_22"][i]),Context=track(z["change_Context_22"][i]),diagnostics=diagnosis(z["change_Context_22"][i],q),selected_slot=0,
                binding=dict(current_source="female_FeiTian" if e==0 else "male_pipa_1",protected_core=[64,160],Root="UNKNOWN",context="SAME_RECORDING_BODY; NOT_CROSS_EVENT_TRUTH"))),
            initial_slot=0,details=dict(source_body_only=True,Root="UNKNOWN",contact="UNKNOWN",not_independent_transfer=True)))
    payload=dict(schema="r35_plan_context_view_v1",cases=cases,tracks=tracks,parents=z["parents"].tolist(),selection=read(root/"planning/result.json"),
        context_model=ref(root/"context/context.pt"),generation=ref(root/"generation/result.json"),human_reviewed=False,operational_qualified=False,whole_song_generations=0,smoothing=False)
    raw=json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    (out/"data.js").write_text('window.DANCE_BUNDLE_B64="'+base64.b64encode(gzip.compress(raw,mtime=0)).decode()+'";\n')
    for n in ("index.html","viewer.js","style.css"):shutil.copyfile(Path("src/dunhuang/research/transition_r35/web")/n,out/n)
    files=[dict(name=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob("*")) if p.is_file()]
    write(out/"protocol.json",dict(code_commit=commit(),cases=len(cases),display_only=True,trained_context=True,common_music_frame=True,
        files=files,audio_count=len(sounds),generated_sequences="REGISTERED_SHORT_WINDOWS_ONLY; NO_COMPLETE_THREE_EVENT_OR_WHOLE_SONG",live_browser_verified=False))
    print("R35_VIEWER",len(cases),len(tracks),len(raw),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    run(read("src/dunhuang/research/transition_r35/config.json"),a.root,a.out)
