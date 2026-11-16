"""Standalone display-only XYZ, true audio and identity-bound diagnostics; no model/rotation tensors."""
import argparse,base64,gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
from dunhuang.contracts import matrices
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions
from dunhuang.planning.event_plan import candidate_identity
from .common import *
from .compiler import native_body

def run(root,out):
    root=Path(root);out=fresh(out);audio_dir=out/"audio";audio_dir.mkdir()
    gen=np.load(root/"plan_generation/arrays.npz");bindings=read(root/"plan_generation/result.json")["records"]
    parents=gen["parents"].tolist();J=gen["skeleton_J22"];tracks={};sounds={};case_rows=[];originals={};provenance=[]
    def track(x):
        x=np.ascontiguousarray(x,dtype="<f4");key=hashlib.sha256(x.tobytes()).hexdigest()
        if key not in tracks:tracks[key]=dict(shape=list(x.shape),dtype="float32_le",data=base64.b64encode(x.tobytes()).decode())
        return key
    def fk(body):
        R=matrices(body.reshape(-1,22,6),row=True);return relative_fk(R,J,parents)
    def sound(path,name):
        key=sha(path)
        if key not in sounds:
            target=audio_dir/name;shutil.copyfile(path,target);assert sha(target)==key;sounds[key]="audio/"+name
        return sounds[key]
    def original(event,interval=None):
        lo,hi=event["source_interval"] if interval is None else interval;lo=int(lo);hi=int(hi)
        key=f'{event["source_sha256"]}:{lo}:{hi}'
        if key not in originals:
            native=np.load(checked(event["reference"]));assert 0<=lo<hi<=len(native)
            R=matrices(native[lo:hi,7:].reshape(-1,24,6))[:,:22]
            originals[key]=dict(track=track(relative_fk(R,J,parents)),source_start=lo,source_end=hi,fps=30,Root_truth="UNKNOWN",source_uid=event["source_uid"])
        return key
    def candidate(e):
        body,pos=native_body(e,np.arange(*e["music_interval"]))
        p=fk(body)
        return dict(event_uid=e["event_uid"],source_uid=e["source_uid"],candidate_uid=e["candidate_uid"],link_id=e["link_id"],
            source_interval=e["source_interval"],music_interval=e["music_interval"],time_warp=e["time_warp"],source_positions=pos.tolist(),
            original=original(e),aligned_track=track(p),scores=e["display_scores"],score_scope=e["score_scope"],certificate=e["timing_certificate"],
            aligned_global_start=e["music_interval"][0],selected=False)
    def diagnostics(points,reference,valid=None):
        p=np.asarray(points,np.float64);r=np.asarray(reference,np.float64);rel=p-p[:,:1]
        valid=np.ones(len(p),bool) if valid is None else valid
        wrist=np.stack((p[:,20]-p[:,16],p[:,21]-p[:,17]),1);truth=np.stack((r[:,20]-r[:,16],r[:,21]-r[:,17]),1)
        err=np.sqrt(np.mean((wrist-truth)**2,axis=(1,2)));error=[float(v) if ok else None for v,ok in zip(err,valid)]
        values=dict(source_error=error)
        for k,name in ((2,"body_D2"),(3,"body_jerk")):
            x=np.linalg.norm(np.diff(rel,n=k,axis=0)*30**k,axis=-1).mean(-1)
            values[name]=[None]*k+x.tolist()
        speed=np.linalg.norm(np.diff(p[:,0],axis=0)*30,axis=-1);values["root_speed"]=[None]+speed.tolist()
        feet=p[:,[7,8,10,11]];v=np.linalg.norm(np.diff(feet[:,:,[0,2]],axis=0)*30,axis=-1)
        near=feet[:-1,:,1]<=.05;values["foot_proxy"]=[None]+[float(x[m].mean()) if m.any() else None for x,m in zip(v,near)]
        return values
    # Three independent complete-music planning cases; actual generation exists for one bound window each.
    for song in read(root/"planning/songs.json"):
        plan=read(checked(song["plan"]));body=[];slots=[]
        for event,slot in zip(plan["events"],song["slots"]):
            b,_=native_body(event,np.arange(*event["music_interval"]));body.append(b)
            cs=[candidate(c) for c in slot["candidates"]]
            for c in cs:c["selected"]=c["candidate_uid"]==slot["selected_candidate_uid"]
            slots.append(dict(slot,candidates=cs))
        full=fk(np.concatenate(body));binding=next(x for x in bindings if x["song_id"]==song["id"])
        si=binding["selected_slot"];start=binding["window_start"];a,b=binding["music_interval"]
        expected=next(c for c in slots[si]["candidates"] if c["selected"])
        assert expected["link_id"]==binding["link_id"]
        # Official GPU source FK independently confirms this display's CPU transform.
        assert abs(fk(gen[song["id"]+"_source_body"])-gen[song["id"]+"_source22"]).max()<2e-5
        valid=(np.arange(256)+start>=a)&(np.arange(256)+start<b)
        result=dict(link_id=binding["link_id"],window_start=start,frames=256,seeds=binding["seeds"],
            baseline=track(gen[song["id"]+"_Frozen_22"][0]),candidate=track(gen[song["id"]+"_Coverage_22"][0]),baseline_variants={},
            observation=[a,b],source_control=[a,b],contexts=[[a-8,a],[b,b+8]],context_status="计划边缘已读取；Context零输出未训练",
            diagnostics=diagnostics(gen[song["id"]+"_Coverage_22"][0],full[start:start+256],valid),
            details=binding,qualification="研究短窗；未证明跨事件衔接；Native24不可评价")
        slots[si]["generation"]=result
        case_rows.append(dict(id=song["id"],name=song["name"]+" · 完整计划",fps=30,frames=song["frames"],audio=sound(checked(song["audio"]),song["name"]),
            beats=song["beats"],waveform=song["waveform"],slots=slots,reference_track=track(full),type="full_song_plan",
            reference_scope="编排计划预演；来源排列存在切换，连接尚待生成；来源世界Root未知",details=dict(plan=song["plan"],semantic_plan_id=song["semantic_plan_id"],scope=song["scope"])))
        provenance.append(dict(song_id=song["id"],plan=song["plan"],generation_binding=binding["generation_identity"]))
    # Existing actual R34 continuous-coverage experiments: no invented link to the new full-song slots.
    a=np.load(root/"data/arrays.npz");meta=read(root/"data/result.json")["records"];ev=np.load(root/"preview/arrays.npz")
    old=read("outputs/lodge_reference_r24_01/main/protocol.json")
    import soundfile as sf
    for m in range(2):
        wav=Path(f"outputs/lodge_reference_r24_01/main/music{m}.wav");wave,sr=sf.read(wav,dtype="float32",always_2d=True)
        mono=wave.mean(1);bins=np.linspace(0,len(mono),513,dtype=int);peaks=[float(abs(mono[x:y]).max()) for x,y in zip(bins[:-1],bins[1:])]
        for e in range(2):
            event=read(checked(old["plans"]["A" if e==0 else "B"]))["events"][0]
            event=dict(event,music_interval=[64,160],time_warp=[[x,y+64] for x,y in event["time_warp"]]);event["candidate_uid"]=candidate_identity(event)
            link=identity(sha(wav),event,0);p=a["local_points"][e]
            assert abs(fk(a["body"][e])-p).max()<2e-5
            lo=int(np.floor(min(meta[e]["source_positions"])));hi=int(np.ceil(max(meta[e]["source_positions"])))+1
            c=dict(event_uid=event["event_uid"],source_uid=event["source_uid"],candidate_uid=event["candidate_uid"],link_id=link,
                source_interval=[lo,hi],core_source_interval=event["source_interval"],music_interval=[0,256],
                time_warp=[[float(x),int(y)] for y,x in enumerate(meta[e]["source_positions"])],source_positions=meta[e]["source_positions"],
                original=original(event,[lo,hi]),aligned_track=track(p),aligned_global_start=0,scores=dict(heuristic=None,global_score=None,sync_score=None),
                score_scope="固定研究来源；未进行检索排序",certificate=dict(status="OBSERVED_CONTINUOUS_EXTENSION; CORE_MAP_UNCHANGED"),selected=True)
            i=3*m+e+1;candidate_p=ev["Coverage_correct_22"][i]
            result=dict(link_id=link,window_start=0,frames=256,seeds=[24041],baseline=track(ev["Frozen_correct_22"][i]),candidate=track(candidate_p),
                baseline_variants={arm:track(ev[arm+"_correct_22"][i]) for arm in ("FrozenFull","Extra")},observation=[0,256],source_control=[0,256],contexts=[],
                context_status="未使用上下文；外围来自同一录制，不是相邻事件",diagnostics=diagnostics(candidate_p,p),
                details=dict(source=meta[e],audio_sha256=sha(wav),noise_seed=24041,candidate_model_sha256=sha(root/"models/Coverage/adapter.pt")),
                qualification="连续覆盖主效果；Core肩腕门槛未通过；非整曲或邻接学习")
            slot=dict(slot=0,start=64,end=160,selected_candidate_uid=c["candidate_uid"],selected_link_id=link,candidates=[c],generation=result,
                legal_candidates=1,rejection_counts={},rejected_examples=[])
            case_rows.append(dict(id=f"study_m{m}_e{e}",name=f"R34研究片段m{m} · {event['source_uid']}",fps=30,frames=256,
                audio=sound(wav,f"study_music{m}.wav"),beats=np.flatnonzero(__import__("dunhuang.research.control_r33.common",fromlist=["population_data"]).population_data()[3]["music"][m,:,-1]>.5).tolist(),
                waveform=peaks,slots=[slot],reference_track=track(p),type="continuous_study",reference_scope="真实同录制连续身体参考；世界Root未知；非相邻事件链",details=result["details"]))
    payload=dict(schema="two_module_view_r34_v1",fps=30,parents=parents,cases=case_rows,tracks=tracks,originals=originals,
        initial_case="study_m0_e1",initial_tab="generation",common_clock="ONE_AUDIO_TIME_TO_INTEGER_FLOOR_FRAME; ALL_CANVASES_IDENTICAL_FRAME",
        contact_truth="UNKNOWN",Native24="NOT_EVALUABLE_22_OF24",context_trained=False,smoothing=False)
    data=json.dumps(payload,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
    packed=base64.b64encode(gzip.compress(data,mtime=0)).decode()
    (out/"data.js").write_text('window.DANCE_BUNDLE_B64="'+packed+'";\n',encoding="utf8")
    templates=Path("src/dunhuang/research/plan_view_r34/web")
    for name in ("index.html","viewer.js","style.css"):shutil.copyfile(templates/name,out/name)
    write(out/"protocol.json",dict(code_commit=commit(),display_only=True,rotation_training_tensors_included=False,weight_files_included=False,
        source=ref(root/"planning/songs.json"),generation=ref(root/"plan_generation/result.json"),study_generation=ref(root/"preview/result.json"),
        cases=[dict(id=x["id"],frames=x["frames"],slots=len(x["slots"]),generated_slots=sum("generation" in s for s in x["slots"])) for x in case_rows],
        audio_files={str(p.relative_to(out)):ref(p) for p in audio_dir.iterdir()},payload_uncompressed_bytes=len(data),payload_json_sha256=hashlib.sha256(data).hexdigest(),
        live_browser_interaction_verified=False,common_integer_clock=True,source_original_clock="INDEPENDENT30FPS; MASTER_PAUSED",
        source_world_Root_known=False,generated_world_motion_unchanged=True))
    print("R34_VIEW_BUNDLE",len(case_rows),len(tracks),len(originals),len(data),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);a=p.parse_args();run(a.root,a.out)
