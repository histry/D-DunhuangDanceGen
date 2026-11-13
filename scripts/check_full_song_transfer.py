"""Fresh-process checks for full-song lineage, saved arrays, plans and media."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha,write
from dunhuang.music import clock_plan
from dunhuang.research.source_capacity.common import read,same_arrays
from dunhuang.planning.time_map import source_positions
from dunhuang.planning.event_plan import validate

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
    assert gr==read(root/"generation_replay/result.json")
    for song,item in zip(fp["songs"],inventory):
        assert song["audio"]["sha256"]==item["audio"]["sha256"] and song["clock"]==item["clock"]
        v=np.load(checked(song["features"]));assert v.shape==(item["clock"]["windows"],150,4800) and np.isfinite(v).all()
        if song.get("feature_receipt"):checked(song["feature_receipt"])
        for sl in song["slices"]:checked(sl["audio"])
    for row,b in zip(gr["records"],gp["bindings"]):
        folder=Path(checked(b["native"])).parent;T=inventory[row["case"]]["clock"]["frames"]
        checked(b["physical"]);checked(b["fk"]);checked(b["request"])
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
    write(root/"public_checks.json",dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        verifier_sha256=sha(__file__),full_songs=len(inventory),plans_verified=plans,exact_generation_replay_arrays=arrays,
        videos=video_counts,fresh_generation_replay=read(root/"generation_replay/replay.json"),
        optimizer_calls=0,operational_qualified=False,human_reviewed=False))
    print("INDEPENDENT_R7_FULL_SONG_CHECKS",plans,arrays,len(video_counts),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();check(a.out)
