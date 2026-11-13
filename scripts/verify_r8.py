"""Independent saved-output checks; never performs optimization or generation."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.io import checked,sha,write
from dunhuang.research.source_capacity.common import read

def reference_error(native,plan):
    values=[]
    for e in plan["events"]:
        src=np.load(checked(e["reference"]));lo,hi=e["source_interval"];a,b=e["music_interval"];warp=np.array(e["time_warp"])
        t=np.minimum(np.interp(np.arange(a,b),warp[:,1],warp[:,0]),hi-1);left=np.floor(t).astype(int);right=np.minimum(left+1,len(src)-1)
        from dunhuang.contracts import matrices
        l=Rotation.from_matrix(matrices(src[left,7:139].reshape(-1,22,6)).reshape(-1,3,3));r=Rotation.from_matrix(matrices(src[right,7:139].reshape(-1,22,6)).reshape(-1,3,3))
        ref=l*Rotation.from_rotvec((l.inv()*r).as_rotvec()*np.repeat(t-left,22)[:,None])
        actual=Rotation.from_matrix(matrices(native[a:b,7:139].reshape(-1,22,6)).reshape(-1,3,3))
        values.extend((actual.inv()*ref).magnitude().reshape(-1,22)[:,1:].ravel().tolist())
    return float(np.mean(values))

def run(root):
    root=Path(root);single=read(root/"single/result.json");p=read(root/"single/protocol.json");cfg=p["config"];count=0;errors=[]
    assert single==read(root/"single_replay/result.json")
    for row,b in zip(single["records"],p["bindings"]):
        folder=Path(checked(b["native"])).parent;previous=root/"single_replay"/folder.name
        checked(b["fk"]);checked(b["physical"]);q=read(checked(b["request"]));plan=read(checked(b["plan"]))
        assert q["frames"]==row["frames"] and q["windows"]==1
        for name in ("normalized.npy","windows.npy","edge.npy","native.npy","native_fk.npy","official_fk.npy"):
            assert np.array_equal(np.load(folder/name),np.load(previous/name));count+=1
        assert read(folder/"physical.json")==read(previous/"physical.json")
        x=np.load(folder/"native.npy");recomputed=reference_error(x,plan)
        label=row["source_label"] or "A";errors.append(abs(recomputed-row["sources"][label]["mean_source_rad"]))
        if row["arm"] in ("S1","S2","S3"):
            audit=read(folder/"reperformance_guide.json")
            assert audit["reference"]["controlled_joints"]==cfg["guide"]["groups"][row["arm"]]
            assert audit["reference"]["source_root_translation"] is False and audit["reference"]["source_contact"] is False
            assert audit["neural_weight_updates"]==0 and len(audit["steps"])==50 and not audit["post_generation_pose_paste"]
        g=read(folder/"generation.json");assert g["weights_before"]==g["weights_after"] and g["adapter"] is None
    assert len(single["records"])==cfg["full_generation_budget"] and max(errors)<1e-6
    prep=read(root/"prepared/protocol.json");pairs=read(checked(prep["pairs"]))
    assert len({u for pair in pairs for u in pair["source_uids"]})==8
    assert len({u for pair in pairs for u in pair["source_sha256"]})==8
    bridge=read(root/"bridge/result.json");assert bridge==read(root/"bridge_replay/result.json")
    bp=read(root/"bridge/protocol.json");candidates=read(checked(read(root/"boundary_audit/protocol.json")["candidates"]));byname={c["name"]:c for c in candidates}
    for b in bp["bindings"]:
        x=np.load(checked(b["motion"]));old=np.load(checked(byname[b["name"]]["native"]));a,z=bp["config"]["edit"];mask=np.ones(len(x),bool);mask[a:z]=False
        assert np.array_equal(x[mask],old[mask]);previous=root/"bridge_replay"/(b["name"]+"_"+b["arm"])
        assert np.array_equal(x,np.load(previous/"edited_native.npy"))
    for phase in ("whole","joint"):
        r=read(root/phase/"result.json");assert r==read(root/(phase+"_replay")/"result.json")
    if not single["single_event_gate_passed"] or not bridge["bridge_gate_passed"]:
        assert read(root/"whole/result.json")["new_whole_song_generations"]==0
        assert read(root/"joint/result.json")["new_local_generations"]==0
    media=read(root/"videos/protocol.json");videos=[]
    for v in media["videos"]:
        f=checked(v["video"]);probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_streams","-of","json",str(f)],text=True))
        stream=next(s for s in probe["streams"] if s["codec_type"]=="video");assert int(stream["nb_read_frames"])==v["frames"] and stream["r_frame_rate"]=="30/1"
        videos.append(dict(name=v["name"],frames=v["frames"],sha256=sha(f)))
    changed=subprocess.check_output(["git","diff","--name-only",cfg["baseline"],"--","src/dunhuang","configs"],text=True).splitlines()
    formal=[p for p in changed if not p.startswith("src/dunhuang/research/")]
    assert not formal
    write(root/"independent_checks.json",dict(code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),verifier_sha256=sha(__file__),
        exact_single_event_arrays=count,source_readout_max_error_rad=max(errors),distinct_sources=8,formal_changed_paths=formal,
        bridge_protected_motion_exact=True,single_gate=single["single_event_gate_passed"],bridge_gate=bridge["bridge_gate_passed"],videos=videos,
        new_neural_weight_updates=0,new_generation_calls=0,operational_qualified=False))
    print("R8_INDEPENDENT_CHECKS",count,len(videos),single["single_event_gate_passed"],bridge["bridge_gate_passed"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args();run(a.out)
