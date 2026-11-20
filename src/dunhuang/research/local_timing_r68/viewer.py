"""R68 actual resampling playback and an independent original-speed source view.

Only presentation is assembled here.  No time map is chosen, no failed output
is replaced, and no body22 diagnostic is promoted to artistic/Native quality.
"""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from dunhuang.research.complete_r36.core import relative_fk


METHODS = ("E1W", "Uniform", "A_early", "A_late", "B_early", "B_late")
PRIMARY = ("E1W", "A_early", "A_late")
LABELS = {"E1W": "冻结 E1-W · 同预算基线",
          "Uniform": "匀速 · 独立对照",
          "A_early": "A 上身速度峰代理 · 提前6帧",
          "A_late": "A 上身速度峰代理 · 延后6帧",
          "B_early": "B 上身减速代理 · 提前6帧",
          "B_late": "B 上身减速代理 · 延后6帧"}
MASKS = ("event_mask", "support_mask", "edit_mask", "compensation_mask", "guard_mask")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def _write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n").encode("utf8"))


def _clock(case, clocks):
    choices = [clocks.get(str(case.get("songidx"))), clocks.get(case.get("music_id"))]
    choices.extend(clocks.values())
    for clock in choices:
        if (isinstance(clock, Mapping) and clock.get("audio_path")
                and clock.get("audio_sha256") == case.get("audio_sha256")):
            return clock
    raise ValueError("Actual identified full-music clock is required")


def _old_audio_hashes():
    path = Path("outputs/rhythm_r66_01/main/viewer_manifest.json")
    if not path.is_file():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8-sig"))
    return {Path(row["path"]).name: row["sha256"] for row in doc["files"]
            if Path(row["path"]).suffix.lower() == ".wav"}


def _audio_file(watch, clock, A, B, old_hashes, audio_files, raw_hashes):
    if not 0 <= A < B <= clock["frames"] or B-A != 240:
        raise ValueError("A genuine 240-frame audio interval is required")
    audio_sha = clock["audio_sha256"]
    name = audio_sha[:12]+"_"+str(A)+"_"+str(B)+".wav"
    if name not in audio_files:
        expected = old_hashes.get(name)
        old_path = Path("outputs/rhythm_r66_01/main/viewer")/name
        if expected is not None and old_path.is_file():
            if _sha(old_path) != expected:
                raise ValueError("Existing authorized R66 PCM changed")
            shutil.copy2(old_path, watch/name)
            provenance = "EXACT_EXISTING_R66_PCM_COPY_NO_NEW_EXTRACTION"
        else:
            raw = Path(clock["audio_path"])
            if raw not in raw_hashes:
                raw_hashes[raw] = _sha(raw)
            if raw_hashes[raw] != audio_sha:
                raise ValueError("Actual raw audio identity differs from source clock")
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(raw),
                            "-ss", str(A/30), "-t", "8", "-vn", "-c:a", "pcm_s16le",
                            "-ac", "1", "-ar", "22050", str(watch/name)], check=True)
            provenance = "IDENTICAL_R66_PCM_EXTRACTION_CONTRACT"
        digest = _sha(watch/name)
        if expected is not None and digest != expected:
            raise ValueError("R68 shared music is not byte-identical to frozen R66 PCM")
        audio_files[name] = dict(sha256=digest, sealed_R66_sha256=expected,
            sealed_R66_exact=expected == digest if expected is not None else None,
            provenance=provenance, audio_interval=[A, B])
    return name


def _rotations(value, frames=None):
    r = np.asarray(value, dtype=float)
    if (r.ndim != 4 or r.shape[1:] != (22, 3, 3) or len(r) < 2
            or frames is not None and len(r) != frames or not np.isfinite(r).all()
            or not np.allclose(r.swapaxes(-1, -2) @ r, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(r), 1, atol=1e-5, rtol=0)):
        raise ValueError("Actual proper body22 SO(3) playback rotations required")
    return r


def _mask(value, name, default=False):
    if value is None:
        return [bool(default)] * 240
    x = np.asarray(value)
    if x.shape != (240,) or x.dtype.kind != "b":
        raise ValueError(name + " must be a genuine boolean 240-frame mask")
    return x.tolist()


def _d3(row, name):
    view = row.get("viewer") or {}
    value = view.get(name + "_FK_D3")
    metrics = row.get("metrics") or {}
    if value is None:
        value = metrics.get(name, {}).get("FK", {}).get("D3", {}).get("curve")
    if value is None and name == "baseline":
        value = metrics.get("reference", {}).get("FK", {}).get("D3", {}).get("curve")
    if value is None:
        return None
    curve = np.asarray(value, dtype=float)
    if curve.shape != (237,) or not np.isfinite(curve).all() or (curve < 0).any():
        raise ValueError("D3 must be the existing finite 237-point full-Core curve")
    return curve.tolist()


def _scalar_summary(row):
    """Keep readable evidence, not solver internals or arbitrary signal arrays."""
    result = {}
    for name in ("status", "reason", "method", "event_type", "delta_frames"):
        value = row.get(name)
        if value is None or isinstance(value, (str, bool, int, float)):
            result[name] = value
    warp = row.get("warp") or {}
    result["warp"] = {k: v for k, v in warp.items()
                      if isinstance(v, (str, bool, int, float)) or v is None}
    result["boundary"] = {k: v for k, v in (row.get("boundary") or row.get("bridge_invariance") or {}).items()
                          if isinstance(v, (str, bool, int, float)) or v is None}
    connection = row.get("connection") or row.get("connectivity") or {}
    if isinstance(connection, Mapping):
        result["connection"] = {k: connection.get(k) for k in
            ("constructible", "passes_original_screen", "bridge_R_sha256", "reason")}
    result.update(artistic_quality="NOT_EVALUABLE", Native24="NOT_EVALUABLE",
                  world_root_contact="UNKNOWN", stage_semantics="KINEMATIC_PROXY_NOT_HUMAN_ROLE")
    return result


def _panel(result, case, q0, J, parents):
    method = result["method"]
    row = result.get("row") or {}
    view = row.get("viewer") or {}
    value = dict(method=method, status=row.get("status", "UNKNOWN"),
                 reason=row.get("reason", (row.get("warp") or {}).get("reason")),
                 diagnostics=_scalar_summary(row), actual_D3=_d3(row, "actual"),
                 baseline_D3=_d3(row, "baseline"))
    for name in MASKS:
        default = name == "guard_mask"
        if default and view.get(name) is None:
            value[name] = [i < 4 or i >= 236 for i in range(240)]
        else:
            value[name] = _mask(view.get(name), name)
    for name in ("event_target_before_local", "event_target_after_local"):
        t = view.get(name)
        if t is not None and (isinstance(t, bool) or not isinstance(t, (int, float, np.number))
                              or not np.isfinite(t) or not 0 <= t <= 239):
            raise ValueError("Target event markers use genuine local frame centers")
        value[name] = None if t is None else float(t)
    if result.get("R") is None:
        value.update(P=None, q=None, dq=None)
        return value
    R = _rotations(result["R"], 240)
    q = np.asarray(result.get("q"), dtype=float)
    a, b = case["source_interval"]
    if (q.shape != (240,) or not np.isfinite(q).all() or not (np.diff(q) > 0).all()
            or q[0] != a or q[-1] != b-1):
        raise ValueError("Complete actual monotone source clock required")
    P = relative_fk(R, J, parents)
    if P.shape != (240, 22, 3) or not np.isfinite(P).all():
        raise ValueError("Finite body22 FK required")
    # The actual q footprint is computed before rounding for presentation.
    if view.get("edit_mask") is None:
        value["edit_mask"] = (q != q0).tolist()
    value.update(P=np.round(P, 5).tolist(), q=np.round(q, 6).tolist(),
                 dq=np.round(q-q0, 8).tolist())
    return value


def build(out, display, J, parents, clocks, methods=None):
    """Build one immutable viewer: primary triplet first, six-way controls later.

    display entries are (case, panels, native). native contains the untouched
    source interval rotations R at fps=30. Its silent clock is independent.
    Each panel has method/R/q/row; row.viewer supplies actual local masks.
    """
    methods = list(PRIMARY if methods is None else methods)
    if (not methods or len(set(methods)) != len(methods)
            or any(m not in METHODS for m in methods) or "E1W" not in methods):
        raise ValueError("Registered unique methods including E1W required")
    out = Path(out); watch = out/"viewer"; watch.mkdir()
    parents = np.asarray(parents)
    J = np.asarray(J, dtype=float)
    if (parents.shape != (22,) or parents.dtype.kind not in "iu" or parents[0] != -1
            or any(not 0 <= parents[j] < j for j in range(1, 22))
            or J.shape != (22, 3) or not np.isfinite(J).all()):
        raise ValueError("Same real common22 skeleton required for every panel")
    old_hashes = _old_audio_hashes(); audio_files = {}; raw_hashes = {}; items = []
    for case, raw_panels, native in display:
        A, B = map(int, case["target_interval"])
        a, b = map(int, case["source_interval"])
        if B-A != 240 or not 0 <= a < b:
            raise ValueError("Registered fixed240 source/music support required")
        results = {}
        for result in raw_panels:
            if not isinstance(result, Mapping) or result.get("method") in results:
                raise ValueError("One actual result per method; no duplicate replacement")
            results[result["method"]] = result
        base = results.get("E1W")
        if base is None or base.get("R") is None:
            raise ValueError("Actual E1W baseline is required to display intervention dose")
        q0 = np.asarray(base["q"], dtype=float)
        if q0.shape != (240,) or not np.isfinite(q0).all():
            raise ValueError("Actual E1W q0 must address all 240 frames")
        clock = _clock(case, clocks)
        if B > clock["frames"] or A < 0:
            raise ValueError("Music clip may not leave original audio clock")
        name = _audio_file(watch, clock, A, B, old_hashes, audio_files, raw_hashes)
        panels = []
        for method in methods:
            result = results.get(method, dict(method=method, R=None, q=None,
                row=dict(status="MISSING_OUTPUT", reason="REGISTERED_OUTPUT_NOT_AVAILABLE")))
            panels.append(_panel(result, case, q0, J, parents))
        if not isinstance(native, Mapping) or native.get("fps") != 30:
            raise ValueError("Original-speed source must declare its real fps=30")
        if native.get("source_start_frame", a) != a:
            raise ValueError("Original-speed source starts at a different registered source frame")
        rawR = _rotations(native["R"], b-a)
        rawP = relative_fk(rawR, J, parents)
        native_payload = dict(P=np.round(rawP, 5).tolist(), frames=b-a, fps=30,
            source_interval=[a, b], speed="ORIGINAL_30FPS_INDEPENDENT_SILENT_CLOCK",
            target_music_synchronized=False, retimed=False)
        events = case.get("selected_events") or {}
        event_metadata = {str(k): ({n: v for n, v in value.items() if n in
            ("frame", "type", "body_region", "evidence_span", "semantic_truth", "selection_rule")}
            if isinstance(value, Mapping) else None) for k, value in events.items()}
        item = dict(key=case["key"], source=case["source_uid"],
            source_interval=[a, b], target_interval=[A, B], q0=np.round(q0, 6).tolist(),
            music=case.get("music_id", case.get("songidx")), audio=name, panels=panels,
            source_sha256=case.get("raw_source_sha256"), audio_sha256=case["audio_sha256"],
            native=native_payload, selected_events=event_metadata,
            beats=[float(t-A) for t in clock.get("beat_frames", []) if A <= t < B],
            accents=[float(t-A) for t in clock.get("accent_frames", []) if A <= t < B])
        alternate = case.get("alternate_music")
        if alternate is not None:
            other = _clock(alternate, clocks)
            lo, hi = map(int, alternate["target_interval"])
            item["alternate_music"] = dict(audio=_audio_file(watch, other, lo, hi,
                old_hashes, audio_files, raw_hashes), music=alternate.get("music_id",alternate.get("songidx")),
                target_interval=[lo, hi], audio_sha256=other["audio_sha256"],
                beats=[float(t-lo) for t in other.get("beat_frames", []) if lo <= t < hi],
                accents=[float(t-lo) for t in other.get("accent_frames", []) if lo <= t < hi],
                scope="AUDIO_ONLY_TECHNICAL_CONTROL_NOT_ENERGY_MATCHED_OR_ARTISTIC_TRUTH",
                body_arrays_changed=False, diagnostic_values_changed=False)
        items.append(item)
    if not items:
        raise ValueError("An empty viewer is not a completed delivery")
    payload = dict(items=items, parents=parents.tolist(), labels=LABELS, methods=methods,
                   fps=30, frames=240, version="R68_LOCAL_TIME_INTERVENTION_ACTUAL_BODY22_V1")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = Path(__file__).with_name("viewer.html").read_text(encoding="utf8")
    if template.count("__DATA__") != 1:
        raise ValueError("One unambiguous actual payload placeholder required")
    (watch/"index.html").write_bytes(template.replace("__DATA__", encoded).encode("utf8"))
    _write(watch/"protocol.json", dict(scope="ACTUAL_SOURCE_ROTATION_RESAMPLING_NO_NEURAL_GENERATION",
        frames=240, fps=30, methods=methods, display_positions=len(items),
        unique_source_intervals=len({(i["source"], tuple(i["source_interval"])) for i in items}),
        roster="ALL_REGISTERED_POSITIONS_AND_FAILURES_NO_OUTCOME_SELECTION",
        native="UNTOUCHED_SOURCE_INTERVAL_AT_ORIGINAL30FPS_INDEPENDENT_SILENT_CLOCK_NO_TARGET_RETIMING",
        audio="ONE_TRUE_TARGET_MUSIC_CLOCK_FOR_ALL_RETIMED_PANELS", audio_files=audio_files,
        camera="SHARED_FIXED_ROOT_RELATIVE_BODY22_Y_UP_190PX_PER_METRE_NO_PER_PANEL_RESCALE",
        curves="FULL_CORE_D3_INDEX_J_USES_FRAMES_J_THROUGH_J+3; DISPLAY_AT_RIGHT_ENDPOINT",
        masks="LOCAL240_EVENT_COMPENSATION_SUPPORT_EDIT_AND_FOUR_FRAME_GUARD_FROM_ACTUAL_LEDGER",
        palette=dict(target_event="#f0ba68", compensation="#b69ae7", protected_edge="#7997ae",
                     output="#6fdac4", baseline="#b8c5d5"),
        diagnostic_scope="D3_AND_CLOCK_CHANGE_ARE_NOT_ARTISTIC_QUALITY_LABELS",
        human_reviewed=False, semantic_stage_truth="UNKNOWN", artistic_quality="NOT_EVALUABLE",
        Native24="NOT_EVALUABLE", world_root_contact="UNKNOWN", can_route=False,
        alternate_audio="OPTIONAL_CONTROLS_ONLY; SWITCH_AUDIO_AND_AUDIO_EVENTS_ONLY; BODY_AND_DYNAMIC_ARRAYS_IDENTICAL"))
    _write(out/"viewer_manifest.json", dict(files=[dict(path=p.relative_to(out).as_posix(),
        bytes=p.stat().st_size, sha256=_sha(p)) for p in sorted(watch.iterdir()) if p.is_file()],
        raw_rotations=False, full_audio=False, weights=False, trajectories_in_html=True,
        media_selection_independent_of_quality=True, original_speed_source_embedded=True,
        quality_winner=None, can_route=False))
    return watch/"index.html"
