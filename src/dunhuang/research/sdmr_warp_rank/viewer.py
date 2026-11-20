"""Four actual source-resampling panels, one audio clock, no quality winner."""
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import relative_fk
from .common import write, sha

METHODS = ("E1W", "Diverse", "Single64", "Multi64")
LABELS = {"E1W": "冻结 E1-W · 重定时参考（非原速）",
          "Diverse": "时钟差异最大候选 · 非最优标签",
          "Single64": "冻结 R41 · 单尺度64读出所选",
          "Multi64": "冻结 R41 · 多尺度64读出所选"}


def _clock(case, music):
    candidates = [music.get(str(case.get("songidx"))), music.get(case.get("music_id"))]
    candidates.extend(music.values())
    expected = case.get("audio_sha256")
    for clock in candidates:
        if isinstance(clock, Mapping) and clock.get("audio_path") and (
                expected is None or clock.get("audio_sha256") == expected):
            return clock
    raise ValueError("Actual source audio clock/path is unavailable for this registered case")


def _old_audio_hashes():
    source = Path("outputs/rhythm_r66_01/main/viewer_manifest.json")
    if not source.exists():
        return {}
    doc = json.loads(source.read_text(encoding="utf-8-sig"))
    return {Path(row["path"]).name: row["sha256"] for row in doc["files"]
            if Path(row["path"]).suffix.lower() == ".wav"}


def _result(value):
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 4:
        method, rotations, positions, row = value
        return dict(method=method, R=rotations, q=positions, row=row)
    raise ValueError("Panel result must be {method,R,q,row} or its four-element tuple")


def _diagnostics(row):
    measurement = row.get("metrics") or {}
    actual = measurement.get("actual", {})
    return dict(score=row.get("score"), candidate_uid=row.get("candidate_uid", row.get("candidateUID", row.get("candidate_id"))),
                connectivity=row.get("connectivity", "NOT_EVALUABLE"),
                FK=actual.get("FK", {}), endpoint_speed=actual.get("endpoint_mean_joint_speed_radps"),
                range_ratios=measurement.get("versus_uniform"),
                reconstruction=measurement.get("reconstruction_check"),
                artistic_quality="NOT_EVALUABLE", Native24="NOT_EVALUABLE",
                world_root_contact="UNKNOWN")


def build(target, display, J, parents, music):
    target = Path(target); watch = target / "viewer"; watch.mkdir()
    parents = list(map(int, parents))
    if len(parents) != 22:
        raise ValueError("The viewer requires the exact body22 parent contract")
    old_hashes = _old_audio_hashes()
    items, audio_files = [], {}
    for case, raw_results in display:
        A, B = map(int, case["target_interval"])
        if B - A != 240:
            raise ValueError("This viewer is the fixed 240-frame/8-second Core comparison")
        clock = _clock(case, music)
        audio_sha = case.get("audio_sha256", clock.get("audio_sha256"))
        if not isinstance(audio_sha, str) or len(audio_sha) != 64:
            raise ValueError("Verified raw audio SHA256 is required")
        name = audio_sha[:12] + "_" + str(A) + "_" + str(B) + ".wav"
        if name not in audio_files:
            # Identical to the sealed R66 PCM extraction: same frames/name/codec.
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", clock["audio_path"],
                            "-ss", str(A / 30), "-t", "8", "-vn", "-c:a", "pcm_s16le",
                            "-ac", "1", "-ar", "22050", str(watch / name)], check=True)
            digest = sha(watch / name)
            expected = old_hashes.get(name)
            if expected is not None and digest != expected:
                raise ValueError("R67 shared audio differs from the sealed R66 PCM excerpt: " + name)
            audio_files[name] = dict(sha256=digest, sealed_R66_sha256=expected,
                                    sealed_R66_exact=expected == digest if expected is not None else None)
        results = {}
        for value in raw_results:
            result = _result(value)
            if result["method"] in results:
                raise ValueError("Duplicate method panel; equal candidate IDs are allowed, duplicate method slots are not")
            results[result["method"]] = result
        panels = []
        for method in METHODS:
            result = results.get(method)
            if result is None:
                panels.append(dict(method=method, status="MISSING_OUTPUT", reason="REGISTERED_PANEL_NOT_AVAILABLE"))
                continue
            row = result.get("row") or {}
            if result.get("R") is None:
                panels.append(dict(method=method, status=row.get("status", "NOT_EVALUABLE"),
                                   reason=row.get("reason", (row.get("warp") or {}).get("reason", "NO_ACTUAL_OUTPUT")),
                                   diagnostics=_diagnostics(row)))
                continue
            R, q = np.asarray(result["R"]), np.asarray(result["q"], dtype=float)
            if R.shape != (240, 22, 3, 3) or not np.isfinite(R).all() or q.shape != (240,) or not np.isfinite(q).all() or not (np.diff(q) > 0).all():
                raise ValueError("Actual finite body22 Core and its genuine increasing source positions are required")
            P = relative_fk(R, J, parents)
            if P.shape != (240, 22, 3) or not np.isfinite(P).all():
                raise ValueError("Invalid actual body22 FK playback coordinates")
            measurement = row.get("metrics") or {}
            panels.append(dict(method=method, status=row.get("status", "AVAILABLE"),
                P=np.round(P, 5).tolist(), q=np.round(q, 6).tolist(),
                phase_target_frames=[p["target_frame"] for p in measurement.get("source_phase_diagnostics", [])
                                     if p.get("type", "").endswith("speed_peak")],
                diagnostics=_diagnostics(row)))
        items.append(dict(key=case.get("key", case.get("case_key", case.get("uid"))),
            source=case["source_uid"], source_interval=case["source_interval"], target_interval=[A, B],
            music=case.get("music_id", case.get("songidx", "UNKNOWN")), audio=name, panels=panels,
            beats=[float(t - A) for t in clock.get("beat_frames", []) if A <= t < B],
            accents=[float(t - A) for t in clock.get("accent_frames", []) if A <= t < B]))
    if not items:
        raise ValueError("Keep a registered nonempty media population; do not publish an empty success viewer")
    payload = dict(items=items, parents=parents, labels=LABELS, fps=30, frames=240,
                   p0_audit_url="../../phase_audit/viewer/index.html")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    if template.count("__DATA__") != 1:
        raise ValueError("Viewer payload placeholder must occur exactly once")
    (watch / "index.html").write_bytes(template.replace("__DATA__", encoded).encode("utf-8"))
    write(watch / "protocol.json", dict(scope="FOUR_ACTUAL_SOURCE_RESAMPLED_BODY22_CORES_NOT_NEURAL_GENERATION",
        fps=30, frames=240, panel_order=list(METHODS), labels=LABELS,
        selection="KEEP_ALL_REGISTERED_CASES_AND_FAILURE_PLACEHOLDERS; CANDIDATE_ID_REPEATS_ALLOWED",
        Diverse="SCORE_INDEPENDENT_MAXIMUM_CLOCK_DIFFERENCE_CANDIDATE_NOT_OPTIMALITY",
        camera="SHARED_FIXED_ROOT_RELATIVE_Y_UP; SCALE190PX_PER_M; NO_PER_PANEL_RESCALE",
        display_positions=len(items), unique_source_intervals=len({(item["source"], tuple(item["source_interval"])) for item in items}),
        audio="SHARED_ACTUAL_MUSIC_PCM_MONO22050; AUDIO_ELEMENT_IS_ONLY_MASTER_CLOCK", audio_files=audio_files,
        frame_controls="ONE_PLAY_PAUSE_SEEK_RATE_PREVIOUS_NEXT_FRAME; NO_INDEPENDENT_PANEL_CLOCKS",
        derivative_curve_clock="FK_Dn_INDEX_J_READS_OUTPUT_FRAMES_J_TO_J+n; DISPLAY_AT_RIGHT_ENDPOINT_FRAME",
        transition_frames_reserved=48, bridge_result="SEPARATE_DIAGNOSTIC_NOT_INHERITED_FROM_CORE_WARP",
        world_root_contact="UNKNOWN", Native24="NOT_EVALUABLE", artistic_quality="NOT_EVALUABLE",
        human_reviewed=False, semantic_stage_truth="UNKNOWN", can_route=False,
        p0_audit="OPTIONAL_SEPARATE_JSON_AT_PARENT/p0_event_audit.json; NOT_AUTOMATIC_EVENT_TRUTH"))
    write(target / "viewer_manifest.json", dict(files=[dict(path=p.relative_to(target).as_posix(),
        bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(watch.iterdir()) if p.is_file()],
        raw_rotations=False, full_audio=False, weights=False, trajectories_in_html=True,
        media_selection_independent_of_scores=True, quality_winner=None, can_route=False))
    return watch / "index.html"
