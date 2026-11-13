"""Fresh-process checks for full-song lineage, saved arrays, plans and media."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha,write
from dunhuang.music import clock_plan
from dunhuang.research.source_capacity.common import read,same_arrays
from dunhuang.planning.time_map import source_positions
from dunhuang.planning.event_plan import validate,anchors

def check(root):
    root=Path(root);music=root/"music";mp=read(music/"protocol.json")
    inventory=read(checked(mp["inventory"]));assert len(inventory)==3
    arrays=0;plans=0
    same_arrays(music/"arrays.npz",root/"music_replay/arrays.npz")
    assert read(music/"result.json")==read(root/"music_replay/result.json")
    for item in inventory:
        x,sr=sf.read(checked(item["audio"]),dtype="float32",always_2d=True)
        assert clock_plan(len(x),sr)==item["clock"] and len(x)==item["samples"]
        assert np.isfinite(x).all() and np.max(np.abs(x))>0
    for binding in mp["bindings"]:
        file=checked(binding["plan"]);plan=read(file);T=inventory[binding["case"]]["clock"]["frames"]
        assert plan["frames"]==T
        validate(plan,inventory[binding["case"]]["audio"]["sha256"],T)
        previous=0
        for e in plan["events"]:
            a,b=e["music_interval"]
            assert a==previous and b>a
            pos=source_positions(e,np.arange(a,b))
            assert np.isfinite(pos).all() and np.all(np.diff(pos)>=0)
            previous=b
        assert previous==T
        other=root/"music_replay"/file.parent.name/file.name
        assert plan==read(other)
        same_arrays(checked(binding["condition"]),other.parent/"condition.npz");plans+=1
    fp=read(root/"features/protocol.json");gp=read(root/"generation/protocol.json");gr=read(root/"generation/result.json")
    assert len(fp["songs"])==3 and len(gr["records"])==10
    for asset in fp["producer_runtime"]["weights"]+fp["producer_runtime"]["package_sources"]+[fp["producer_runtime"]["official_extractor"]]:checked(asset)
    plans_index=read(checked(mp["generation_plans"]));replayed_index=read(root/"music_replay/generation_plans.json");sideplans={}
    for item,other in zip(plans_index,replayed_index):
        assert item["status"]==other["status"]=="FULL_EVENT_PLANS_BOUND"
        for label in ("A","B"):
            plan=read(checked(item["plans"][label]));assert plan==read(checked(other["plans"][label]))
            case=item["case"];validate(plan,inventory[case]["audio"]["sha256"],inventory[case]["clock"]["frames"]);sideplans[(case,label)]=plan
        assert all(a["reference"]["sha256"]!=b["reference"]["sha256"] for a,b in zip(sideplans[(item["case"],"A")]["events"],sideplans[(item["case"],"B")]["events"]))
    assert gr==read(root/"generation_replay/result.json")
    for song,item in zip(fp["songs"],inventory):
        assert song["audio"]["sha256"]==item["audio"]["sha256"] and song["clock"]==item["clock"]
        v=np.load(checked(song["features"]));assert v.shape==(item["clock"]["windows"],150,4800) and np.isfinite(v).all()
        if song.get("feature_receipt"):
            old=read(checked(song["feature_receipt"]));assert old.get("original_audio",old.get("audio"))["sha256"]==item["audio"]["sha256"] and old["clock"]==item["clock"] and old["layer"]==66
        wave,sr=sf.read(checked(item["audio"]),dtype="float32",always_2d=True);mono=wave.mean(1)
        for i,sl in enumerate(song["slices"]):
            clip,rate=sf.read(checked(sl["audio"]),dtype="float32",always_2d=True);start=round(i*2.5*sr)
            expected=np.pad(mono,(0,item["clock"]["silence_samples"]))[start:start+5*sr,None]
            assert rate==sr and np.array_equal(clip,expected) and sl["source_start_sample"]==start
            assert hashlib.sha256(v[i].tobytes()).hexdigest()==sl["features_sha256"]
    for row,b in zip(gr["records"],gp["bindings"]):
        folder=Path(checked(b["native"])).parent;T=inventory[row["case"]]["clock"]["frames"]
        checked(b["physical"]);checked(b["fk"]);request=read(checked(b["request"]));arm=row["arm"]
        expected=anchors(sideplans[(row["case"],"A" if arm=="temporal_A" else "B")]) if arm in ("temporal_A","temporal_B") else []
        assert request["anchors"]==expected and request["frames"]==T and request["audio"]["sha256"]==inventory[row["case"]]["audio"]["sha256"]
        gen=read(folder/"generation.json");assert gen["weights_before"]==gen["weights_after"] and gen["adapter"] is None and gen["training"] is False
        assert not row["sampler"]["post_generation_pose_paste"]
        if arm=="music_order_control":
            original=np.load(checked(fp["songs"][row["case"]]["features"]));order=read(folder.parent/"permuted_music_order.json")
            assert np.array_equal(np.load(checked(request["features"])),original[order])
        else:assert request["features"]["sha256"]==fp["songs"][row["case"]]["features"]["sha256"]
        assert np.load(folder/"native.npy").shape==(T,151) and np.load(folder/"native_fk.npy").shape==(T,24,3)
        assert read(folder/"physical.json")==read(root/"generation_replay"/folder.name/"physical.json")
        for name in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy"):
            assert np.array_equal(np.load(folder/name),np.load(root/"generation_replay"/folder.name/name));arrays+=1
    vp=read(root/"videos/protocol.json");video_counts=[]
    assert len(vp["records"])==10
    for row in vp["records"]:
        file=checked(row["video"])
        probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_streams","-of","json",str(file)],text=True))
        videos=[s for s in probe["streams"] if s["codec_type"]=="video"];audio=[s for s in probe["streams"] if s["codec_type"]=="audio"]
        assert len(videos)==1 and len(audio)==1 and int(videos[0]["nb_read_frames"])==row["frames"]
        assert videos[0]["r_frame_rate"]=="30/1"
        video_counts.append(dict(case=row["case"],arm=row["arm"],frames=int(videos[0]["nb_read_frames"]),video_sha256=sha(file)))
    changed=subprocess.check_output(["git","diff","--name-only",mp["config"]["baseline"],"--","src/dunhuang","configs"],text=True).splitlines()
    formal=[p for p in changed if not p.startswith("src/dunhuang/research/") and p not in ("configs/exposed_full_song_r7_v1.json","configs/source_contrast_r7_v1.json")]
    assert formal==[]
    write(root/"public_checks.json",dict(formal_baseline_changed_paths=formal,code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        verifier_sha256=sha(__file__),full_songs=len(inventory),plans_verified=plans,exact_generation_replay_arrays=arrays,
        videos=video_counts,fresh_generation_replay=read(root/"generation_replay/replay.json"),
        optimizer_calls=0,operational_qualified=False,human_reviewed=False))
    print("INDEPENDENT_R7_FULL_SONG_CHECKS",plans,arrays,len(video_counts),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();check(a.out)
