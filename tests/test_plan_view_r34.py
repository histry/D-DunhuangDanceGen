import copy
import numpy as np
import pytest
from dunhuang.research.plan_view_r34.common import identity,relative_fk
from dunhuang.research.plan_view_r34.compiler import compile_window,choose
from dunhuang.data.sources import ref
from dunhuang.planning.event_plan import candidate_identity

def event(path,uid,source,start):
    a,b=start,start+96
    e=dict(event_uid=uid,source_uid=uid,source_sha256=uid*64,reference=ref(path),source_interval=[source,source+96],music_interval=[a,b],
        time_warp=[[source,a],[source+36,a+36],[source+60,a+60],[source+96,b]],primitive=dict(source_frame=source+48,music_frame=a+48,joints=list(range(12,20)),radius=12))
    e["candidate_uid"]=candidate_identity(e);return e

def test_binding_is_sensitive_to_music_event_map_and_window(tmp_path):
    p=tmp_path/"native.npy";np.save(p,np.zeros((400,151),np.float32));e=event(p,"a",100,96)
    a=identity("one",e);assert a!=identity("two",e) and a!=identity("one",e,32)
    q=copy.deepcopy(e);q["time_warp"][1][1]+=1;assert identity("one",q)!=a

def test_compiler_uses_actual_adjacent_plan_events_and_scoped_source(tmp_path):
    from dunhuang.contracts import sixd
    native=np.zeros((400,151),np.float32);native[:,7:]=sixd(np.broadcast_to(np.eye(3),(400,24,3,3))).reshape(400,144)
    p=tmp_path/"native.npy";np.save(p,native)
    events=[event(p,uid,source,start) for uid,source,start in (("a",0,0),("b",100,96),("c",200,192))]
    plan=dict(events=events,frames=288,fps=30,audio_sha256="music")
    i=choose(plan);a,b=compile_window(plan,i,"music")
    assert i==1 and b["window_start"]==32 and b["source_control_local_interval"]==[64,160]
    assert b["neighbor_event_uids"]==["a"]*8+["c"]*8 and a["control"].sum()==96
    assert a["body"].shape==(96,132) and a["context_indices"].tolist()==list(range(56,64))+list(range(160,168))
    assert np.array_equal(a["source_positions"],np.arange(100,196))

def test_relative_fk_uses_real_parent_hierarchy():
    R=np.broadcast_to(np.eye(3),(3,22,3,3));J=np.arange(66).reshape(22,3)/10
    p=relative_fk(R,J,[-1]+list(range(21)))
    assert np.allclose(p,J[None]-J[:1])

@pytest.mark.parametrize("name",["planning","compiler","generate","bundle","verify"])
def test_imports(name):
    import importlib
    importlib.import_module("dunhuang.research.plan_view_r34."+name)

def test_replay_accepts_container_metadata_only_but_rejects_motion_change(tmp_path):
    import wave,struct,json
    from dunhuang.research.plan_view_r34.verify import verify_generation
    original=tmp_path/"original";replay=tmp_path/"replay";original.mkdir();replay.mkdir()
    p=original/"audio.wav"
    with wave.open(str(p),"wb") as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(np.arange(100,dtype="<i2").tobytes())
    raw=p.read_bytes();junk=b"JUNK"+struct.pack("<I",4)+b"TIME"
    changed=raw[:4]+struct.pack("<I",len(raw)-8+len(junk))+raw[8:]+junk
    (replay/"audio.wav").write_bytes(changed)
    for folder in (original,replay):
        np.savez(folder/"arrays.npz",motion=np.arange(8,dtype=np.float32),music35=np.ones((3,35),np.float32))
        (folder/"result.json").write_text(json.dumps(dict(records=[dict(music_window=ref(folder/"audio.wav"),seed=1)],optimizer_steps=0)))
    r=verify_generation(original,replay)
    assert r["audio_PCM_exact"] and r["music35_and_motion_arrays_exact"]
    np.savez(replay/"arrays.npz",motion=np.arange(8,dtype=np.float32)+1,music35=np.ones((3,35),np.float32))
    with pytest.raises(ValueError,match="Frozen array replay differs"):verify_generation(original,replay)
