"""Presentation contracts: actual clocks, retained failures, no invented quality."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pytest


def _module():
    here = Path(__file__).resolve().parent
    path = here/"viewer.py"
    if not path.is_file():
        choices = list((here.parent/"src/dunhuang/research").glob("*r68/viewer.py"))
        if len(choices) != 1:
            raise RuntimeError("Exactly one isolated R68 viewer expected")
        path = choices[0]
    spec = importlib.util.spec_from_file_location("r68_viewer_contract_under_test", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


v = _module()


def _fixture(tmp_path, monkeypatch, alternate=False):
    monkeypatch.chdir(tmp_path)
    sha = "a"*64; other_sha = "b"*64
    existing = tmp_path/"outputs/rhythm_r66_01/main/viewer"
    existing.mkdir(parents=True)
    hashes = {}
    # Byte-copy fixtures test identity handling, not audio decoding or music quality.
    for identity, A in [(sha, 1000), (other_sha, 1500)]:
        name = identity[:12]+"_"+str(A)+"_"+str(A+240)+".wav"
        body = b"PCM_COPY_IDENTITY_FIXTURE_"+identity.encode()
        (existing/name).write_bytes(body); hashes[name] = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(v, "_old_audio_hashes", lambda: hashes)
    def no_extract(*args, **kwargs):
        raise AssertionError("Verified existing PCM must be reused")
    monkeypatch.setattr(v.subprocess, "run", no_extract)
    parents = np.r_[-1, np.arange(21)]
    J = np.c_[np.zeros(22), np.arange(22)*.03, np.zeros(22)]
    R = np.broadcast_to(np.eye(3), (240,22,3,3)).copy()
    native = dict(R=np.broadcast_to(np.eye(3), (300,22,3,3)).copy(), fps=30)
    q = np.linspace(100,399,240)
    event = np.zeros(240, bool); event[115:125] = True
    support = np.zeros(240, bool); support[60:180] = True
    case = dict(key="registered_case", source_uid="real_source_uid", source_interval=[100,400],
                target_interval=[1000,1240], songidx=0, music_id="song0", audio_sha256=sha,
                selected_events={"A":dict(type="upper_speed_peak", body_region="upper", frame=250,
                                           evidence_span=[249,252], semantic_truth="UNKNOWN"), "B":None})
    clocks = {"0":dict(audio_sha256=sha, audio_path="not-read-after-verified-reuse.wav",frames=3000,
                        beat_frames=[1000,1020,1200,1300],accent_frames=[1010,1210]),
              "1":dict(audio_sha256=other_sha,audio_path="not-read-other.wav",frames=3000,
                        beat_frames=[1500,1520,1700,1800],accent_frames=[1510,1710])}
    if alternate:
        case["alternate_music"] = dict(songidx=1, music_id="song1", audio_sha256=other_sha,
                                       target_interval=[1500,1740])
    row = dict(status="LEGAL", viewer=dict(event_mask=event.tolist(),support_mask=support.tolist(),
        compensation_mask=(support&~event).tolist(),event_target_before_local=120.,
        event_target_after_local=114.,actual_FK_D3=np.arange(237,dtype=float).tolist(),
        baseline_FK_D3=np.ones(237).tolist()),
        warp=dict(legal=True, rate_min=.8, boundary_jets=[[1,2,3]], token_matrix=[[4,5,6]]),
        connectivity=dict(constructible=True,passes_original_screen=True,boundary_jets=[[99,98,97]],
                          bridge_R_sha256="c"*64))
    panels = [dict(method=m,R=R.copy(),q=q.copy(),row=dict(row)) for m in v.METHODS]
    panels[3] = dict(method="A_late",R=None,q=None,row=dict(status="NOT_FEASIBLE",
        reason="RATE_LIMIT",viewer=dict(event_mask=event.tolist(),support_mask=support.tolist())))
    return case, panels, native, J, parents, clocks, existing


def _payload(path):
    text = path.read_text(encoding="utf8")
    match = re.search(r'<script id="r68-data" type="application/json">(.*?)</script>',text,re.S)
    assert match is not None
    return json.loads(match.group(1)), text


def test_primary_keeps_failure_and_native_original_duration_on_independent_clock(tmp_path,monkeypatch):
    case, panels, native, J, parents, clocks, _ = _fixture(tmp_path,monkeypatch)
    out = tmp_path/"primary"; out.mkdir()
    page = v.build(out,[(case,panels,native)],J,parents,clocks)
    data, text = _payload(page); item = data["items"][0]
    assert [p["method"] for p in item["panels"]] == ["E1W","A_early","A_late"]
    assert item["panels"][2]["P"] is None
    assert item["panels"][2]["status"] == "NOT_FEASIBLE"
    assert item["panels"][2]["reason"] == "RATE_LIMIT"
    assert len(item["panels"][0]["P"]) == 240
    assert item["native"]["frames"] == len(item["native"]["P"]) == 300
    assert item["native"]["fps"] == 30 and item["native"]["retimed"] is False
    assert item["native"]["target_music_synchronized"] is False
    assert "alternate_music" not in item
    assert "+1e-4" in text
    protocol = json.loads((page.parent/"protocol.json").read_text(encoding="utf8"))
    assert protocol["Native24"] == "NOT_EVALUABLE" and protocol["can_route"] is False


def test_six_way_audio_control_does_not_duplicate_or_change_motion_payload(tmp_path,monkeypatch):
    case,panels,native,J,parents,clocks,_ = _fixture(tmp_path,monkeypatch,alternate=True)
    root = tmp_path/"controls"; root.mkdir()
    page = v.build(root,[(case,panels,native)],J,parents,clocks,methods=v.METHODS)
    data,_ = _payload(page); item = data["items"][0]
    assert len(item["panels"]) == 6
    alt = item["alternate_music"]
    assert alt["body_arrays_changed"] is False and alt["diagnostic_values_changed"] is False
    assert alt["audio"] != item["audio"] and alt["beats"] == [0.,20.,200.]
    assert "P" not in alt and "q" not in alt
    assert item["panels"][0]["q"] == item["panels"][2]["q"]
    assert item["panels"][0]["actual_D3"] == list(np.arange(237,dtype=float))
    assert item["panels"][0]["diagnostics"]["warp"] == {"legal":True,"rate_min":.8}
    assert "boundary_jets" not in item["panels"][0]["diagnostics"]["connection"]
    assert len(list(page.parent.glob("*.wav"))) == 2


def test_invalid_mask_and_misaligned_derivative_curve_are_rejected(tmp_path,monkeypatch):
    case,panels,native,J,parents,clocks,_ = _fixture(tmp_path,monkeypatch)
    panels[0]["row"]["viewer"]["event_mask"] = [False]*239
    out=tmp_path/"bad_mask";out.mkdir()
    with pytest.raises(ValueError,match="boolean 240"):
        v.build(out,[(case,panels,native)],J,parents,clocks)
    panels[0]["row"]["viewer"]["event_mask"] = [False]*240
    panels[0]["row"]["viewer"]["actual_FK_D3"] = [1.]*240
    out=tmp_path/"bad_d3";out.mkdir()
    with pytest.raises(ValueError,match="237-point"):
        v.build(out,[(case,panels,native)],J,parents,clocks)


def test_changed_previously_sealed_music_is_not_silently_reextracted(tmp_path,monkeypatch):
    case,panels,native,J,parents,clocks,existing = _fixture(tmp_path,monkeypatch)
    (existing/("a"*12+"_1000_1240.wav")).write_bytes(b"CHANGED")
    out=tmp_path/"corrupt_audio";out.mkdir()
    with pytest.raises(ValueError,match="Existing authorized R66 PCM changed"):
        v.build(out,[(case,panels,native)],J,parents,clocks)


def test_untrusted_identity_cannot_end_the_embedded_json_script(tmp_path,monkeypatch):
    case,panels,native,J,parents,clocks,_ = _fixture(tmp_path,monkeypatch)
    case["source_uid"]="</script><img src=x>"
    out=tmp_path/"escaped";out.mkdir()
    data,text=_payload(v.build(out,[(case,panels,native)],J,parents,clocks))
    assert data["items"][0]["source"] == case["source_uid"]
    assert "</script><img src=x>" not in text
