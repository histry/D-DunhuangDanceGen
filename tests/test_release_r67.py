"""TEST_FIXTURE publication contracts; no real experiment or human labels.

Load the script by file location so the tests also work in a checkout whose
scripts/ directory is not a Python package. Source arrays/media are never read.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def _script():
    candidates = (Path(__file__).with_name("release_r67.py"),
                  Path(__file__).resolve().parents[1]/"scripts/release_r67.py",
                  Path("scripts/release_r67.py"))
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeError("R67 publisher script not found")


SPEC = importlib.util.spec_from_file_location("r67_publication_test_fixture", _script())
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
SHA = "a"*64


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data)+"\n").encode("utf8"))


def _reward_source():
    relative = Path("src/dunhuang/research/timing_r64/music.py")
    candidates = (Path(__file__).resolve().parents[1]/relative,
                  Path.cwd()/relative, Path("D:/DunhuangDanceGen")/relative)
    for path in candidates:
        if path.is_file():
            return path.read_bytes()
    raise RuntimeError("The actual original R64 reward source is required; no mirrored test implementation")


def test_common_roi_boundaries_whole_evidence_and_audio_offset_once(monkeypatch):
    """A large nonzero A makes a second offset application observable."""
    reward_source = _reward_source()
    class VirtualPath:
        def __init__(self, value): self.value = value
        def __truediv__(self, value): return VirtualPath(self.value+"/"+str(value))
        def __str__(self): return self.value
        def is_file(self): return True
        def stat(self): return SimpleNamespace(st_size=123)
        def read_bytes(self): return reward_source
    case = dict(key="TEST_FIXTURE", source_uid="TEST_FIXTURE_SOURCE", source_interval=[0,240],
                target_interval=[1000,1240], songidx=0, audio_sha256=SHA)
    record = dict(case_key=case["key"], method="E1W", source_uid=case["source_uid"],
                  source_interval=[0,240], target_interval=[1000,1240], songidx=0, status="LEGAL",
                  metrics={"source_phase_diagnostics":[
                      dict(type="speed_peak", target_frame=t, source_evidence_whole_inside_core=whole)
                      for t,whole in ((2,True),(3,True),(100,False),(236,True),(237,True))],
                      "actual_output_peaks":{"peaks":[{"frame":t} for t in (3,20,236,237)]}})
    clock = dict(audio_sha256=SHA, frames=2000, beat_frames=[1003,1236],
                 accent_frames=[1003,1236], accent_strength=[1.]*2000)
    def fake_read(path):
        name = str(path)
        if name.endswith("artifacts_manifest.json"):
            return {"files":{"result.json":SHA,"population.json":SHA}}
        if name.endswith("population.json"): return {"cases":[case]}
        if name.endswith("result.json"): return {"records":[record]}
        if name.endswith("scientific_files.json"):
            return {"files":[dict(path="main/music_clocks.json",sha256=SHA,bytes=123)]}
        if name.endswith("music_clocks.json"): return {"0":clock}
        raise AssertionError(name)
    monkeypatch.setattr(release,"read",fake_read)
    monkeypatch.setattr(release,"sha",lambda path:SHA)
    out = release.r66_common_roi(VirtualPath("TEST_FIXTURE_REPO"),
                                {"r64_music_clocks_sha256":SHA},[],SHA)
    row = out["records"][0]
    assert (row["mapped_peak_count"],row["redetected_peak_count"]) == (2,3)
    assert row["mapped_source_proxy"]["primary"] == 1.
    assert .32 < row["mapped_minus_output_primary"] < .34
    assert out["methods"]["E1W"]["evaluable_source_intervals"] == 1
    assert out["methods"]["E1W"]["ci95"] is None
    assert "per_event_distance_frames" not in json.dumps(out)
    # Same private inputs with a wrong frozen-clock identity must fail closed.
    with pytest.raises(ValueError,match="clock bytes/hash"):
        release.r66_common_roi(VirtualPath("TEST_FIXTURE_REPO"),
                               {"r64_music_clocks_sha256":"b"*64},[],SHA)


def test_phase_and_main_overlay_bind_their_own_scientific_html(tmp_path):
    inputs=[]; paths=[]; science={}
    for mode, old, new in (("phase_audit","1"*64,"2"*64),("main","3"*64,"4"*64)):
        science[mode] = {"files":[{"path":"viewer/index.html","sha256":old}]}
        path=tmp_path/(mode+"_presentation_fix.json")
        _write(path,{"files":[{"path":f"outputs/sdmr_warp_rank_01/{mode}/viewer/index.html",
                             "old_sha256":old,"new_sha256":new,"scope":"TEST_FIXTURE_PRESENTATION_ONLY"}]})
        paths.append(path)
    out=release.presentation_receipts(tmp_path,tmp_path/"root",tmp_path/"execution",paths,science,inputs)
    assert len(out["phase_html_overlays"]) == len(out["main_html_overlays"]) == 1
    assert all(row["old_hash_matches_scientific_manifest"] for row in out["html_overlays"])
    assert out["main_html_overlays"][0]["sealed_scientific_html_sha256"] == "3"*64
    assert out["main_html_overlays"][0]["declared_local_overlay_sha256"] == "4"*64
    assert out["main_html_overlays"][0]["scope"] == "TEST_FIXTURE_PRESENTATION_ONLY"
    assert not out["main_html_overlays"][0]["local_file_rehashed_by_this_exporter"]
    science["main"]["files"][0]["sha256"]="5"*64
    mismatch=release.presentation_receipts(tmp_path,tmp_path/"root",tmp_path/"execution",paths,science,[])
    assert mismatch["phase_html_overlays"][0]["old_hash_matches_scientific_manifest"]
    assert not mismatch["main_html_overlays"][0]["old_hash_matches_scientific_manifest"]


def test_all_twelve_ui_case_records_and_scope_survive_projection(tmp_path):
    path=tmp_path/"main_browser_acceptance.json"
    checks=[dict(selected=i,frame="121",panels=4,clock="TEST_FIXTURE_FRAME_CLOCK") for i in range(12)]
    _write(path,dict(scope="TEST_FIXTURE_UI_NOT_ARTISTIC_ASSESSMENT",registeredCasesChecked=checks,
                     fourPanels=True,fullExperimentalDetailsRetained=True))
    out=release.presentation_receipts(tmp_path,tmp_path/"root",tmp_path/"execution",[path],{},[])
    record=out["records"][0]["projection"]
    assert record["registeredCasesChecked"] == checks
    assert record["scope"] == "TEST_FIXTURE_UI_NOT_ARTISTIC_ASSESSMENT"
    assert record["registeredCaseCheckSummary"]["checked_record_count"] == 12
    assert record["registeredCaseCheckSummary"]["distinct_selected_indices"] == 12
    assert record["registeredCaseCheckSummary"]["all_records_have_four_panels"]


def test_scores_errors_intervals_survive_but_raw_q_tokens_and_matrices_do_not():
    raw={"queries":[{"uid":"TEST_FIXTURE","candidate_errors":[1.,0.],
                     "prefixes":{"1":[0],"16":[0,1]},"source_interval":[10,250],
                     "conditions":{"actual":{"Multi64":{"16":{"selected_tied_indices":[0,1],
                                   "expected_clock_error":.5,"ci95":[0.,1.]}}}}}],
         "q":[0.,1.],"tokens":[[1.,2.]],"B_matrix":[[1.,2.]],"boundary_jets":[[3.,4.]],
         "curve":[5.],"rotations":[[[1.]]],"audio_samples":[0.,1.],
         "scores_by_control":{"actual":{"Single64":[.1,None]}},
         "body_reconstruction_mean_angles_rad":[.02,.03]}
    out=release.project(raw,scalar_lists=True)
    assert out["queries"][0]["candidate_errors"] == [1.,0.]
    assert out["queries"][0]["prefixes"]["16"] == [0,1]
    assert out["queries"][0]["source_interval"] == [10,250]
    assert out["scores_by_control"]["actual"]["Single64"] == [.1,None]
    assert out["body_reconstruction_mean_angles_rad"] == [.02,.03]
    assert not set(("q","tokens","B_matrix","boundary_jets","curve","rotations","audio_samples")) & set(out)
    assert out["queries"][0]["conditions"]["actual"]["Multi64"]["16"]["ci95"] == [0.,1.]


def test_failed_partial_bank_is_retained_without_becoming_completed_main(tmp_path):
    root=tmp_path/"root"; archive=root/"failures/main_706"; partial=archive/"partial_main"
    _write(archive/"execution_main.json",dict(commit="TEST_FIXTURE_706",exit_status=1,status="FAILED",
                                             password="TEST_FIXTURE_SECRET",command="TEST_FIXTURE_SECRET"))
    (archive/"main.log").write_bytes(b"TEST_FIXTURE_FAILURE_LOG_NOT_PUBLIC_CONTENT\n")
    bank={"actual_unique_bank_q_count":1,"legal_path_count":10,"qspread":{"max_abs":0.,
          "pairwise_mean_abs":[[0.]],"pairwise_max_abs":[[0.]]},"candidates":[{
          "uid":"TEST_FIXTURE_CANDIDATE","q":[0.,1.],"time_warp":[[0,0],[1,1]],"cost":0.,
          "source_interval":[0,240],"target_interval":[0,240]}],"prefixes":{"1":["TEST_FIXTURE_CANDIDATE"]},
          "graph":{"source":[0,239],"options":[[0],[239]]}}
    _write(partial/"banks.json",[{"case_key":"TEST_FIXTURE","bank":bank}])
    out=release.archived_failures(root,[])
    assert out["completed_scientific_stages"] == 0
    assert release.stage_ready(partial,("records.json","banks.json","registration.json")) == "INCOMPLETE_NOT_PUBLISHED_AS_FINISHED"
    item=out["records"][0]
    assert release.failed_receipt(item["execution_receipts"][0]["projection"])
    assert item["log_hashes"][0]["sha256"] == release.sha(archive/"main.log")
    text=json.dumps(out)
    assert "TEST_FIXTURE_SECRET" not in text and "TEST_FIXTURE_FAILURE_LOG_NOT_PUBLIC_CONTENT" not in text
    public_candidate=item["partial_banks"]["banks"][0]["bank"]["candidates"][0]
    assert "q" not in public_candidate and "time_warp" not in public_candidate
    assert item["partial_banks"]["scope"] == "PREPARED_SHARED_BANK_METADATA_NOT_EXECUTED_CANDIDATE_RESULTS"
