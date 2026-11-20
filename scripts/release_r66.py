"""Publish R66 scalar diagnostics only; never copy media or scientific arrays.

Run from the repository after a completed stage, for example:
    python scripts/release_r66.py --out reports/r66_release
Every output directory must be fresh. Missing optional stages remain NOT_AVAILABLE.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean


HEX = re.compile(r"^[0-9a-f]{64}$")
DROP_KEYS = {
    "q", "R", "J", "P", "coordinates", "coords", "rotations", "rotation",
    "waveform", "samples", "audio_samples", "raw_audio", "audio_bytes",
    "source_positions", "source_grid", "target_grid", "curve", "curves",
    "source_to_target_anchors", "time_warp", "anchors", "target_peak_frames",
    "source_phase_anchors", "peak_frames", "event_observations", "events",
    "source_frame", "target_frame", "audio_frame", "source_evidence_span",
    "interpolation_read_support", "source_to_target_rate", "source_to_target_curvature",
    "inverse_source_rate_per_target_frame", "reviewer_id", "reviewer_roster_ref",
    "person_uid", "blind_item_id", "blind_variant_id", "password", "credential",
    "private_key", "token", "authentication_evidence_path", "seed_assignment",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False) + "\n").encode("utf-8"))


def scalar(value):
    return value is None or isinstance(value, (str, bool, int)) or (
        isinstance(value, float) and math.isfinite(value))


def scalar_tree(value):
    """Positive shape projection: dictionaries/scalars only, never sequences.

    Removing every sequence also strips per-frame curves, phase positions, FK
    coordinates, jointwise vectors, warp knots and raw events, regardless of key.
    """
    if scalar(value):
        return value
    if isinstance(value, dict):
        return {k: scalar_tree(v) for k, v in value.items()
                if k not in DROP_KEYS and (scalar(v) or isinstance(v, dict))}
    raise ValueError("Scalar projection does not accept a sequence or nonfinite scalar")


def interval(value):
    if not (isinstance(value, list) and len(value) == 2
            and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in value) and value[0] < value[1]):
        raise ValueError("Invalid metadata interval")
    return value


def safe_relative(base, name):
    name = str(name).replace("\\", "/")
    if ":" in name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Unsafe manifest-relative filename: " + name)
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError("Manifest escaped its scientific stage")
    return path, name


def binding(value):
    """Publish code/dependency hashes, never an unrestricted source binding."""
    output = {}
    for key in ("commit", "code", "dependencies"):
        item = value.get(key) if isinstance(value, dict) else None
        if isinstance(item, str):
            output[key] = item
        elif isinstance(item, dict):
            output[key] = {str(k): v for k, v in item.items()
                           if isinstance(v, str) and HEX.fullmatch(v)}
    return output


def phase_counts(rows):
    counts = Counter()
    used = Counter()
    for row in rows or []:
        if not isinstance(row, dict):
            raise ValueError("Invalid phase diagnostic row")
        kind = str(row.get("type", "UNKNOWN"))
        counts[kind] += 1
        if row.get("used_by_warp") is True:
            used[kind] += 1
    return {"count": sum(counts.values()), "by_type": dict(counts),
            "used_by_warp_by_type": dict(used), "semantic_truth": "UNKNOWN",
            "phase_coordinates_published": False}


def b_record(row):
    result = {key: row.get(key) for key in (
        "case_key", "source_uid", "songidx", "method", "status", "rotation_sha256")}
    result["source_interval"] = interval(row["source_interval"])
    result["target_interval"] = interval(row["target_interval"])
    result["warp_scalar_diagnostics"] = scalar_tree(row.get("warp", {}))
    result["actual_music_reward"] = scalar_tree(row.get("actual_music_reward", {}))
    metrics = row.get("metrics")
    result["metrics"] = scalar_tree(metrics) if metrics is not None else None
    if metrics:
        result["source_phase_counts"] = phase_counts(metrics.get("source_phase_diagnostics", []))
        result["hold_turn_counts"] = phase_counts(metrics.get("hold_turn_diagnostics", []))
        peaks = metrics.get("actual_output_peaks", {})
        result["actual_output_peak_counts"] = {
            "actual": len(peaks.get("peaks", [])), "uniform": len(peaks.get("uniform_peaks", [])),
            "status": peaks.get("status"), "scope": "REDETECTED_KINEMATIC_PROXY_NOT_EVENT_TRUTH"}
    result.update(quality="NOT_EVALUABLE", semantic_event_truth="UNKNOWN",
                  adjacent_bridge_revalidated=False, transition_frames_reserved=48,
                  Native24="NOT_EVALUABLE", can_route=False)
    return result


def flatten_numbers(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = prefix + "." + key if prefix else key
            if isinstance(item, dict):
                output.update(flatten_numbers(item, name))
            elif isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item):
                output[name] = float(item)
    return output


def b_summary(rows):
    methods = sorted({r["method"] for r in rows})
    result = {"registered_records": len(rows), "registered_cases": len({r["case_key"] for r in rows}),
              "unique_source_intervals": len({(r["source_uid"], tuple(r["source_interval"])) for r in rows}),
              "inference": "DESCRIPTIVE_ONLY_SHARED_SOURCE_INTERVALS_ACROSS_MUSIC_NOT_12_INDEPENDENT_SAMPLES",
              "confidence_intervals": None, "quality_winner": None, "methods": {}}
    for method in methods:
        own = [r for r in rows if r["method"] == method]
        groups = defaultdict(list)
        for row in own:
            groups[(row["source_uid"], tuple(row["source_interval"]))].append(row)
        by_source = []
        for (uid, span), items in sorted(groups.items()):
            values = defaultdict(list)
            for row in items:
                if row["status"] == "LEGAL":
                    for key, value in flatten_numbers({"metrics": row["metrics"],
                                                       "music": row["actual_music_reward"]}).items():
                        values[key].append(value)
            by_source.append({"source_uid": uid, "source_interval": list(span),
                              "registered_cases": len(items), "status_counts": dict(Counter(r["status"] for r in items)),
                              "available_legal_descriptive_means": {k: mean(v) for k, v in values.items()},
                              "metric_evaluable_case_counts": {k: len(v) for k, v in values.items()}})
        balanced = defaultdict(list)
        for source in by_source:
            for key, value in source["available_legal_descriptive_means"].items():
                balanced[key].append(value)
        result["methods"][method] = {
            "registered_cases": len(own), "status_counts": dict(Counter(r["status"] for r in own)),
            "source_intervals": by_source,
            "source_interval_balanced_available_legal_means": {k: mean(v) for k, v in balanced.items()},
            "metric_evaluable_source_intervals": {k: len(v) for k, v in balanced.items()},
            "missing_or_rejected_cases": [{"case_key": r["case_key"], "source_uid": r["source_uid"],
                                           "source_interval": r["source_interval"], "target_interval": r["target_interval"],
                                           "status": r["status"], "reason": r["warp_scalar_diagnostics"].get("reason")}
                                          for r in own if r["status"] != "LEGAL"]}
    return result


def validate_a_summary(value):
    """A summary already contains scalar scores/offsets, not event/audio arrays.

    Preserve all stratification, registered denominators, ties, query metrics and
    music-group bootstrap results. Fail rather than silently dropping evidence.
    """
    def check(item, key=""):
        if key in DROP_KEYS or key.lower() in {"energy_curve", "accent_strength", "beat_frames", "accent_frames"}:
            raise ValueError("Unexpected private array/identity field in A summary: " + key)
        if isinstance(item, dict):
            for k, v in item.items():
                check(v, k)
        elif isinstance(item, list):
            for v in item:
                check(v)
        elif not scalar(item):
            raise ValueError("Nonfinite or unsupported A summary scalar")
    check(value)
    return value


def a_synthetic(value):
    result = scalar_tree(value)
    result["scores_by_audio_offset_frames"] = {}
    for method, scores in value.get("scores", {}).items():
        if not isinstance(scores, list) or len(scores) != 5 or not all(scalar(v) for v in scores):
            raise ValueError("Synthetic control score vector differs from the five registered offsets")
        result["scores_by_audio_offset_frames"][method] = {
            str(offset): score for offset, score in zip((-15, -8, 0, 8, 15), scores)}
    result["synthetic_event_coordinates_published"] = False
    return result


def manifest_rows(stage, label, observed):
    path = stage / "artifacts_manifest.json"
    if not path.exists():
        return {"status": "NOT_AVAILABLE", "files": []}
    doc = read(path)
    files = []
    for name, expected in sorted(doc["files"].items()):
        target, relative = safe_relative(stage, name)
        if not HEX.fullmatch(expected) or not target.is_file() or sha(target) != expected:
            raise ValueError("Scientific manifest mismatch: " + label + "/" + relative)
        files.append({"stage": label, "path": relative, "sha256": expected, "bytes": target.stat().st_size})
    observed.append(path)
    return {"status": "REHASHED_EXACT", "manifest_sha256": sha(path), "binding": binding(doc.get("binding", {})),
            "files": files, "raw_artifact_bytes_copied": False}


def replay(path, observed, source_manifest):
    if not path.exists():
        return {"status": "NOT_AVAILABLE", "scientific_outputs_exact": None}
    doc = read(path)
    observed.append(path)
    result = scalar_tree({k: v for k, v in doc.items() if k not in ("binding", "differences")})
    result["binding"] = binding(doc.get("binding", {}))
    result["differences"] = [str(v) for v in doc.get("differences", [])]
    result["source_manifest_binding_exact"] = (
        source_manifest.is_file() and doc.get("source_manifest_sha256") == sha(source_manifest))
    result["status"] = "EXACT" if (doc.get("scientific_outputs_exact") is True
        and not result["differences"] and result["source_manifest_binding_exact"]) else "NOT_EXACT_OR_UNBOUND"
    result["receipt_sha256"] = sha(path)
    return result


RECEIPT_FIELDS = {
    "schema", "status", "mode", "phase", "stage", "commit", "code_commit", "executed_commit",
    "github_commit", "server_commit", "exit_status", "exit_code", "passed", "pid", "tmux_session",
    "started_at", "finished_at", "completed_at", "timestamp", "created_at", "current_time",
    "scientific_outputs_exact", "neural_training_steps", "training_steps", "new_planner_searches",
    "geometry_bridge_solves", "new_planner_runs", "optimizer_calls", "tests_passed", "tests_skipped",
    "sha256", "result_sha256", "log_sha256", "manifest_sha256", "receipt_sha256",
    "source_manifest_sha256", "success", "failed", "termination", "reason", "cleanup_status",
}


def receipt_projection(value):
    if not isinstance(value, dict):
        return {}
    output = {}
    for key, item in value.items():
        if key in RECEIPT_FIELDS and scalar(item):
            output[key] = item
        elif isinstance(item, dict) and key not in DROP_KEYS:
            nested = receipt_projection(item)
            if nested:
                output[key] = nested
    return output


def export(repo, out, b_root=None, a_root=None, a_replay=None, receipt_roots=()):
    repo = Path(repo).resolve()
    resolve = lambda p: Path(p).resolve() if Path(p).is_absolute() else (repo / p).resolve()
    b_root = resolve(b_root or "outputs/rhythm_r66_01")
    a_root = resolve(a_root or "outputs/timing_validity_r66_01")
    a_replay = resolve(a_replay or "outputs/timing_validity_r66_replay_01/result.json")
    out = resolve(out)
    if out.exists() or any(out.is_relative_to(p) for p in (b_root, a_root, a_replay.parent)):
        raise ValueError("Publication directory must be fresh and outside scientific input roots")
    if not (b_root / "main/result.json").is_file():
        raise ValueError("R66-B main result is required; no publication from an unfinished stage")
    observed, outputs = [], {}
    scientific = {}
    b_rows = []
    stage_states = {}
    for mode in ("main", "controls"):
        stage = b_root / mode
        if not (stage / "result.json").exists():
            stage_states["B_" + mode] = "NOT_AVAILABLE_NOT_CLAIMED_RUN"
            continue
        scientific["B_" + mode] = manifest_rows(stage, "B_" + mode, observed)
        if scientific["B_" + mode]["status"] != "REHASHED_EXACT":
            raise ValueError("Completed B stage lacks its scientific manifest")
        doc = read(stage / "result.json")
        own = [b_record(row) for row in doc["records"]]
        outputs["B_" + mode + ".json"] = {
            "population": doc["population"], "methods": doc["methods"], "records": own,
            "summary": b_summary(own), "binding": binding(doc.get("binding", {})),
            "quality": "NOT_EVALUABLE", "Native24": "NOT_EVALUABLE", "can_route": False}
        observed.append(stage / "result.json")
        viewer = stage / "viewer_manifest.json"
        if viewer.exists():
            observed.append(viewer)
            outputs["B_" + mode + "_viewer_manifest.json"] = {
                "source_manifest_sha256": sha(viewer), "files": [
                    {"path": v["path"], "bytes": v["bytes"], "sha256": v["sha256"]}
                    for v in read(viewer)["files"]], "media_copied": False,
                "delivery_authorization_or_browser_acceptance_inferred": False}
        b_rows.extend(own)
        stage_states["B_" + mode] = "SCIENTIFIC_FILES_REHASHED"
    identities = {}
    keys = set()
    for row in b_rows:
        key = (row["case_key"], row["method"])
        identity = (row["source_uid"], tuple(row["source_interval"]),
                    tuple(row["target_interval"]), row["songidx"])
        if key in keys or (row["case_key"] in identities and identities[row["case_key"]] != identity):
            raise ValueError("Duplicated or mismatched B case/method across stages")
        keys.add(key)
        identities[row["case_key"]] = identity
    outputs["B_all_available_summary.json"] = b_summary(b_rows)
    outputs["independent_B_main.json"] = replay(b_root / "replay_main/independent.json", observed,
                                               b_root / "main/artifacts_manifest.json")
    outputs["independent_B_controls.json"] = replay(b_root / "replay_controls/independent.json", observed,
                                                   b_root / "controls/artifacts_manifest.json")
    a_files = ("summary.json", "synthetic_control.json", "qualification.json")
    complete_a = all((a_root / name).exists() for name in a_files) and (a_root / "artifacts_manifest.json").exists()
    if complete_a:
        scientific["A_main"] = manifest_rows(a_root, "A_main", observed)
        outputs["A_summary.json"] = validate_a_summary(read(a_root / "summary.json"))
        outputs["A_synthetic_control.json"] = a_synthetic(read(a_root / "synthetic_control.json"))
        outputs["A_qualification.json"] = scalar_tree(read(a_root / "qualification.json"))
        observed.extend(a_root / name for name in a_files)
        stage_states["A_main"] = "SCIENTIFIC_FILES_REHASHED"
    else:
        stage_states["A_main"] = "NOT_AVAILABLE_OR_INCOMPLETE_NOT_CLAIMED_RUN"
    outputs["independent_A.json"] = replay(a_replay, observed, a_root / "artifacts_manifest.json")
    outputs["scientific_files.json"] = {"stages": scientific, "artifact_bytes_copied": False}
    roots = list(receipt_roots) or ["outputs/rhythm_r66_execution_01", "outputs/timing_validity_r66_execution_01"]
    receipts = []
    for root in roots:
        root = resolve(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.json")):
            public = receipt_projection(read(path))
            receipts.append({"receipt": root.name + "/" + path.relative_to(root).as_posix(),
                             "source_sha256": sha(path), "projection": public,
                             "scope": "WHITELISTED_EXECUTION_RECEIPT_NOT_SCIENTIFIC_RESULT"})
            observed.append(path)
    outputs["execution_receipts.json"] = {"records": receipts, "missing_receipts_do_not_imply_success": True}
    outputs["qualification.json"] = {
        "stage_states": stage_states, "human_reviewed": False, "independent_human_labels": 0,
        "semantic_event_truth": "UNKNOWN", "artistic_quality": "NOT_EVALUABLE",
        "source_style_quality": "NOT_EVALUABLE", "Native24": "NOT_EVALUABLE",
        "world_root_contact": "UNKNOWN", "adjacent_bridge_revalidated": False,
        "quality_winner": None, "operational_qualified": False, "can_route": False,
        "neural_training_steps": 0, "new_planner_searches": 0, "geometry_bridge_solves": 0,
        "B_scope": "FIXED_ENDPOINT_SOURCE_ROTATION_RESAMPLING; PROXY_REACHABILITY_AND_DYNAMIC_DIAGNOSTICS",
        "B_sample_scope": "12_DISPLAY_CASES_SHARE_8_SOURCE_INTERVALS; DESCRIPTIVE_NOT_12_INDEPENDENT_CI",
        "B_transition_scope": "48_FRAMES_RESERVED_NOT_REGENERATED; CHANGED_CORE_ENDPOINT_JETS_REQUIRE_NEW_BRIDGE_CHECK",
        "A_scope": "SOURCE_CLOCK_DISCRIMINATION_NOT_ARTISTIC_MATCHING; DATASET_ROLE_STRATA_SEPARATE",
        "new_independent_generalization": False, "historically_exposed_sources": True,
        "review_data_used_to_generate_labels": False,
        "no_signal_arrays_media_phase_coordinates_or_rater_identity_published": True}
    # Read the original protocol bytes, never reconstruct them from result fields.
    protocols = {}
    for letter, package in (("A", "timing_validity_r66"), ("B", "rhythm_r66")):
        path = repo / "src/dunhuang/research" / package / "protocol.json"
        if not path.is_file():
            raise ValueError("Original checked-out protocol missing: " + str(path))
        protocols[letter + "_protocol.json"] = path.read_bytes()
        observed.append(path)
    # Every input read above is recorded only by a repository-relative name/hash.
    outputs["publication_inputs.json"] = {"files": [
        {"path": p.relative_to(repo).as_posix() if p.is_relative_to(repo) else p.name,
         "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(set(observed))],
        "raw_inputs_or_media_copied": False}
    out.mkdir(parents=True, exist_ok=False)
    for name, value in outputs.items():
        write(out / name, value)
    for name, payload in protocols.items():
        (out / name).write_bytes(payload)
    main = outputs["B_main.json"]["summary"]
    counts = "；".join(f"{method}: {values['status_counts']}" for method, values in main["methods"].items())
    report = (
        "# R66 公开诊断摘要\n\n"
        f"B 主候选共有 {main['registered_cases']} 个展示案例、{main['unique_source_intervals']} 个来源区间。{counts}。\n\n"
        "所有拒绝记录保留其来源区间、目标区间、状态及原因；不可把仅成功案例的均值解释为完整人口结果。"
        "B 以来源区间先平均跨音乐案例，再作等区间描述，未计算12案例独立置信区间。\n\n"
        f"阶段状态：{json.dumps(stage_states, ensure_ascii=False)}。独立复验状态应分别查 independent_A、independent_B_main、independent_B_controls；缺失不是通过。\n\n"
        "A 评价固定原始来源时钟的正确/错位区分，各数据域与角色分开；错歌只是诊断，并非艺术错误真值。"
        "B 的优化拍点得分只是已用目标的技术可达性。实际输出重检峰值、D1/D2/D3、幅度及端点读数均为诊断，不能据单一 Jerk 或目标得分宣告质量获胜。\n\n"
        "本轮是来源旋转的合法时间重采样，不是神经动作重新演绎。固定端点姿态仍可能改变端点速度；48帧过渡仅保留预算，未重算或重新验收。"
        "艺术质量、文化语义及 Native24 均未取得独立证据，can_route=false。来源已有历史开发曝光。\n\n"
        "本发布仅含标量统计、区间身份、代码协议、文件哈希与回执投影；不含动作坐标/旋转、逐帧时间图、音频、媒体或审核身份。"
        "观看交付和浏览验收不由文件存在推断。\n")
    (out / "报告.md").write_bytes(report.encode("utf-8"))
    files = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)}
             for p in sorted(out.iterdir()) if p.is_file()]
    write(out / "public_manifest.json", {"files": files, "self_excluded": True,
                                          "qualification": "DIAGNOSTIC_ONLY_NO_QUALITY_WINNER"})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--out", default="reports/r66_release")
    parser.add_argument("--b-root")
    parser.add_argument("--a-root")
    parser.add_argument("--a-replay")
    parser.add_argument("--receipt-root", action="append", default=[])
    args = parser.parse_args()
    print(export(args.repo, args.out, args.b_root, args.a_root, args.a_replay, args.receipt_root))


if __name__ == "__main__":
    main()
