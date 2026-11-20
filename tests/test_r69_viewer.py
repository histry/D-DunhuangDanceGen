"""R69 projection, replay and availability tests; no synthetic human ratings."""
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _module():
    here=Path(__file__).resolve().parent;path=here/"viewer.py"
    if not path.is_file():
        choices=list((here.parent/"src/dunhuang/research").glob("*r69/viewer.py"))
        if len(choices)!=1:raise RuntimeError("Exactly one R69 viewer module required")
        path=choices[0]
    spec=importlib.util.spec_from_file_location("r69_viewer_contract_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


v=_module();SEED=bytes(range(32))


def fixture(tmp_path,monkeypatch):
    raw=tmp_path/"source_audio.wav";raw.write_bytes(b"IDENTIFIED_RAW_AUDIO_FIXTURE")
    import hashlib
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    commands=[]
    def extract(command,check):
        commands.append(command)
        Path(command[-1]).write_bytes(b"IDENTITY_ONLY_PCM_FIXTURE_NOT_AUDIO_DECODE_TEST")
    monkeypatch.setattr(v.subprocess,"run",extract)
    J=np.c_[np.arange(22)*.01,np.arange(22)*.03,np.arange(22)*.005];parents=np.r_[-1,np.arange(21)]
    native=dict(R=np.broadcast_to(np.eye(3),(300,22,3,3)).copy(),fps=30)
    q=np.linspace(10,309,240);panels=[]
    for i,method in enumerate(v.METHODS):
        R=np.broadcast_to(np.eye(3),(240,22,3,3)).copy()
        R[:,0]=Rotation.from_rotvec([.02*i,0,0]).as_matrix()
        full=np.broadcast_to(np.eye(3),(528,22,3,3)).copy();full[:240]=R
        panels.append(dict(method=method,R=R,q=q.copy(),full_R=full,row=dict(status="LEGAL",
            secret_score=100+i,connection=dict(constructible=True,passes_original_screen=True,
            pair_music_interval=[1000,1528],boundary_jets=[[123,456]]))))
    case=dict(key="private_case_key",source_uid="private_unchange_source_name",source_interval=[10,310],
        target_interval=[1000,1240],pair_music_interval=[1000,1528],parent_side="left",songidx=0,
        music_id="private_music_name",audio_sha256=digest)
    clocks={"0":dict(frames=3000,audio_path=str(raw),audio_sha256=digest)}
    return (case,panels,native),J,parents,clocks,commands


def payload(path,tag):
    text=path.read_text(encoding="utf8")
    match=re.search(r'<script id="'+tag+r'" type="application/json">(.*?)</script>',text,re.S)
    assert match
    return json.loads(match.group(1)),text


def build(tmp_path,monkeypatch,name="result",seed=SEED,missing=False):
    row,J,parents,clocks,commands=fixture(tmp_path,monkeypatch)
    if missing:row[1][2]["full_R"]=None
    out=tmp_path/name;out.mkdir()
    v.build(out,[row],J,parents,clocks,blind_seed=seed)
    return out,row,commands


def test_anonymous_page_and_protocol_contain_no_named_method_or_source_metadata(tmp_path,monkeypatch):
    out,row,commands=build(tmp_path,monkeypatch)
    public,text=payload(out/"reviewer/index.html","r69-review-data")
    assert public["frames"]==528 and len(public["cases"][0]["variants"])==6
    for word in list(v.METHODS)+[row[0]["source_uid"],row[0]["key"],row[0]["music_id"],"secret_score","boundary_jets"]:
        assert word not in text
        assert word not in (out/"reviewer/protocol.json").read_text(encoding="utf8")
    assert set(public["cases"][0])=={"blind_item_id","audio","variants"}
    for variant in public["cases"][0]["variants"]:
        assert set(variant)=={"blind_variant_id","available","P"}
    admin=json.loads((out/"admin/map.json").read_text())
    assert {x["method"] for x in admin["cases"][0]["variants"]}==set(v.METHODS)
    assert admin["schema"]=="R69_ADMIN_V1" and admin["form_id"]=="F01"
    assert admin["human_reviewed"] is False
    assert all(part not in str(public) for part in ("source_uid","masks","score","seed_sha256"))
    assert commands[0][commands[0].index("-t")+1]=="17.6"
    assert commands[0][commands[0].index("-map_metadata")+1]=="-1"


def test_same_private_seed_replays_all_page_and_map_bytes_and_manifest_excludes_admin(tmp_path,monkeypatch):
    row,J,parents,clocks,_=fixture(tmp_path,monkeypatch)
    a=tmp_path/"one";b=tmp_path/"two";a.mkdir();b.mkdir()
    for root in (a,b):v.build(root,[row],J,parents,clocks,blind_seed=SEED)
    files={str(p.relative_to(a)) for p in a.rglob("*") if p.is_file()}
    assert all((a/name).read_bytes()==(b/name).read_bytes() for name in files)
    manifest=json.loads((a/"viewer_manifest.json").read_text())
    assert manifest["private_admin_included"] is False
    assert all(not x["path"].startswith("admin/") for x in manifest["files"])
    assert not list((a/"reviewer").glob("*review*.json"))


def test_private_seed_change_changes_blind_identity_without_changing_motion_population(tmp_path,monkeypatch):
    row,J,parents,clocks,_=fixture(tmp_path,monkeypatch)
    a=tmp_path/"seed1";b=tmp_path/"seed2";a.mkdir();b.mkdir()
    v.build(a,[row],J,parents,clocks,blind_seed=SEED)
    v.build(b,[row],J,parents,clocks,blind_seed=bytes(reversed(SEED)))
    x,_=payload(a/"reviewer/index.html","r69-review-data");y,_=payload(b/"reviewer/index.html","r69-review-data")
    assert x["pack_id"]!=y["pack_id"]
    assert x["cases"][0]["blind_item_id"]!=y["cases"][0]["blind_item_id"]
    coordinates=lambda p:sorted(json.dumps(v["P"],sort_keys=True) for v in p["cases"][0]["variants"])
    assert coordinates(x)==coordinates(y)
    with pytest.raises(ValueError,match="private32-byte"):
        v.build(tmp_path/"no-seed",[row],J,parents,clocks,blind_seed=123)


def test_missing_full_pair_is_not_replaced_by_valid_core_in_anonymous_playback(tmp_path,monkeypatch):
    out,_,_=build(tmp_path,monkeypatch,missing=True)
    named,_=payload(out/"viewer/index.html","r69-data")
    panel=next(p for p in named["cases"][0]["panels"] if p["method"]=="BeatDP")
    assert panel["P"] is None and panel["full_pair_available"] is False
    assert len(panel["core_P"])==240 and panel["core_only_available"] is True
    public,text=payload(out/"reviewer/index.html","r69-review-data")
    unavailable=[p for p in public["cases"][0]["variants"] if not p["available"]]
    assert len(unavailable)==1 and unavailable[0]["P"] is None
    assert "if(!v.available&&value!==''&&value!=='UNSURE')o.disabled=true" in text


def test_mismatched_full_pair_core_or_music_clock_is_rejected(tmp_path,monkeypatch):
    row,J,parents,clocks,_=fixture(tmp_path,monkeypatch)
    row[1][0]["full_R"][10,0]=Rotation.from_rotvec([.1,0,0]).as_matrix()
    out=tmp_path/"bad_pair";out.mkdir()
    with pytest.raises(ValueError,match="actual selected Core bytes"):
        v.build(out,[row],J,parents,clocks,blind_seed=SEED)
    row[1][0]["full_R"][:240]=row[1][0]["R"]
    row[0]["target_interval"]=[1001,1241]
    out=tmp_path/"bad_clock";out.mkdir()
    with pytest.raises(ValueError,match="actual full-pair music clock differ"):
        v.build(out,[row],J,parents,clocks,blind_seed=SEED)


def test_reviewer_projection_rejects_added_metadata_and_remains_blank_until_user_action(tmp_path,monkeypatch):
    out,_,_=build(tmp_path,monkeypatch)
    public,text=payload(out/"reviewer/index.html","r69-review-data")
    public["cases"][0]["variants"][0]["method"]="leak"
    with pytest.raises(ValueError,match="variant whitelist"):
        v._review_projection(public)
    assert "if(!items.length)throw Error" in text
    assert "R69_QUALITY_REVIEW_V1" in text and "R69_QUALITY_DRAFT_V1" in text
    assert "UNSURE" in text and "UNRATED" in text
    assert "method_identity_seen:''" in text and "own_judgment:false" in text
    assert list(v.DIMENSIONS)==["music_coordination","transition_timing","continuity","richness","naturalness"]

def test_no_event_with_explicit_null_connection_keeps_placeholder(tmp_path,monkeypatch):
    row,J,parents,clocks,_=fixture(tmp_path,monkeypatch)
    panel=next(p for p in row[1] if p['method']=='BeatDP')
    panel.update(R=None,q=None,full_R=None,row=dict(status='NO_EVENT',reason='NO_SOURCE_EVENT',connection=None))
    out=tmp_path/'null_connection';out.mkdir()
    v.build(out,[row],J,parents,clocks,blind_seed=SEED)
    public,_=payload(out/'reviewer/index.html','r69-review-data')
    assert sum(not x['available'] for x in public['cases'][0]['variants'])==1
