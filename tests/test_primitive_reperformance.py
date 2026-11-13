import copy,json
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from dunhuang.contracts import sixd,matrices
from dunhuang.io import sha
from dunhuang.research.primitive_reperformance.reference import build
from dunhuang.research.primitive_reperformance.study import whole
CFG=dict(base_pose_strength=.55,phase_pose_bonus=.15,phase_sigma_frames=3.,boundary_fade_frames=6,groups={"S1":list(range(12,20)),"S2":list(range(12,22)),"S3":list(range(22))})
def fixture(tmp):
    source=np.zeros((140,151),np.float32);source[:,7:]=np.tile([1,0,0,0,1,0],24)
    for j in range(22):source[:,7+6*j:13+6*j]=sixd(Rotation.from_rotvec(np.arange(140)[:,None]*np.array([.002,0,0])).as_matrix())
    source[:,4:7]=100 # unknown Root must never become a label
    file=tmp/"motion.npy";np.save(file,source)
    e=dict(source_interval=[0,120],music_interval=[0,104],time_warp=[[0,0],[48,40],[72,64],[120,104]],event_uid="source_event",
        reference=dict(path=str(file),sha256=sha(file)),observed_joints=list(range(22)),root_translation_observed=False,
        primitive=dict(source_frame=60,music_frame=52,joints=list(range(12,20)),radius=12),feasibility={"native":"UNKNOWN"},
        full_phases=[dict(frame=t,type="upper_speed_peak",observed_joints=list(range(12,20))) for t in (10,60,100)])
    return dict(schema="event_plan_v1",fps=30,frames=104,audio_sha256="audio",time_scale_limits=[.75,1.33],events=[e])
def test_complete_reference_preserves_nonlinear_map_and_masks_unknowns(tmp_path):
    p=fixture(tmp_path);x,m,w,ph=build(p,CFG)
    tau=np.interp(np.arange(104),[0,40,64,104],[0,48,72,120]);expected=Rotation.from_rotvec(tau[:,None]*[.002,0,0]).as_matrix()
    assert np.allclose(matrices(x[:,127:133]),expected,atol=1e-6) # actual wrist20
    assert m[:,:22].all() and not m[:,22:].any() and not x[:,:7].any()
    assert (w[:,:22]>0).all() and not w[:,22:].any()
    assert np.all(np.diff([a["music_frame"] for a in ph])>0)
def test_complete_reference_rejects_unsupported_wrist_and_phase(tmp_path):
    p=fixture(tmp_path);p["events"][0]["observed_joints"]=list(range(20))
    with pytest.raises(ValueError):build(p,CFG)
    p=fixture(tmp_path);p["events"][0]["full_phases"][0]["frame"]=121
    with pytest.raises(ValueError):build(p,CFG)
def test_full_song_never_runs_when_either_gate_fails(tmp_path):
    single=tmp_path/"single";single.mkdir();bridge=tmp_path/"bridge";bridge.mkdir()
    (single/"result.json").write_text(json.dumps(dict(single_event_gate_passed=True,qualification=dict(source_passes=16,native_passes=16,denominator=16))))
    (bridge/"result.json").write_text(json.dumps(dict(bridge_gate_passed=False)))
    out=tmp_path/"whole";whole({},tmp_path/"missing_assets",single,out,bridge=bridge)
    assert json.loads((out/"result.json").read_text())["new_whole_song_generations"]==0

def test_dense_joint_groups_and_no_source_root_paste(tmp_path):
    torch=pytest.importorskip("torch");pytest.importorskip("pytorch3d")
    from dunhuang.research.primitive_reperformance.guide import prepare
    p=fixture(tmp_path);file=tmp_path/"plan.json";file.write_text(json.dumps(p))
    for group in ("S1","S2","S3"):
        folder=tmp_path/group;folder.mkdir()
        q=dict(event_plan=dict(path=str(file),sha256=sha(file)),frames=104,windows=1,audio={"sha256":"audio"},condition_group=group,reperformance_config=CFG)
        norm=SimpleNamespace(scaler=SimpleNamespace(scale_=torch.ones(151),min_=torch.zeros(151)))
        guide=prepare(q,norm,folder,"cpu");x=torch.zeros(1,150,151);x[:,:,7:]=torch.tensor([1,0,0,0,1,0]*24)
        x[:,:,4:7]=.2
        y=guide(x,.9)
        assert torch.equal(y[:,:,4:7],x[:,:,4:7]) and torch.equal(y[:,:,139:],x[:,:,139:])
        expected=group!="S1"
        assert bool((abs(y[0,:104,127:139]-x[0,:104,127:139])>1e-6).any())==expected


def test_worker_environment_survives_legacy_native_policy_namespace(monkeypatch):
    from pathlib import Path
    from dunhuang.research.primitive_reperformance.environment import worker_environment
    monkeypatch.setenv("PYTHONPATH","unrelated_legacy_package")
    monkeypatch.setenv("NATIVE_POLICY_TEST","preserved")
    env=worker_environment()
    assert Path(env["PYTHONPATH"]).name=="src" and (Path(env["PYTHONPATH"])/"dunhuang").is_dir()
    assert env["NATIVE_POLICY_TEST"]=="preserved" and "unrelated" not in env["PYTHONPATH"]


def test_phase_proxy_partial_path_preserves_order_and_allows_null():
    from dunhuang.research.primitive_reperformance.phase_analysis import partial
    e=[dict(frame=10,type="onset"),dict(frame=20,type="hold"),dict(frame=30,type="turn")]
    a=[dict(frame=11,type="onset"),dict(frame=18,type="unrelated"),dict(frame=33,type="turn")]
    p=partial(e,a)
    assert len(p)==2 and [v["generated_frame"] for v in p]==[11,33]
    assert partial(e,[])==[]
