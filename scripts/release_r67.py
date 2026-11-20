"""Publish R67 scalar evidence; scientific/media arrays are rehashed, never copied.

Run from the repository: python scripts/release_r67.py --out reports/r67_release
Use a fresh --out for each later release. Incomplete optional stages are explicit.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

HEX = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN = {
    "q", "q_injected", "truth_q", "truth_tau", "h", "time_warp", "graph", "B_matrix",
    "R", "P", "J", "Q", "rotations", "coordinates", "coords", "tokens", "mt", "dt",
    "music_tokens", "motion_tokens", "source_positions", "target_positions", "token_frame_centers",
    "source_grid", "target_grid", "source_to_target_anchors", "source_phase_anchors",
    "interpolation_read_support", "boundary_jets", "boundary_proof", "boundary_matrix", "jets",
    "curve", "curves", "jerk_curve", "source_phase_diagnostics", "hold_turn_diagnostics",
    "peaks", "uniform_peaks", "audio_query_frames", "source_evidence_span", "source_frame",
    "target_frame", "audio_frame", "phase_target_frames", "source_phase_spans", "source_evidence_frames",
    "source_to_target_rate", "source_to_target_curvature", "inverse_source_rate_per_target_frame",
    "waveform", "samples", "audio_samples", "audio_bytes", "energy_curve", "accent_strength",
    "beat_frames", "accent_frames", "raw_audio", "reviewer_id", "reviewer_roster_ref", "person_uid",
    "blind_item_id", "blind_variant_id", "password", "credential", "token", "private_key",
    "source", "donor", "source_proofs", "raw_path", "audio_path", "arrays", "path",
}
SCALAR_LISTS = {
    "ci95", "ks", "methods", "readouts", "feature_controls", "offsets", "lags",
    "candidate_errors", "deformation_costs", "costs", "body_reconstruction_mean_angles_rad",
    "rotation_hashes", "original_cost_ranks", "candidate_indices", "selected_tied_indices",
    "source_interval", "target_interval", "audio_interval", "frame_scope", "support_component",
    "gap_interval_local", "pair_music_interval", "differences", "candidate_uids", "source_uids",
}
SCORE_METHODS = {"Single64", "Multi64", "UniformReadout64", "Energy", "RawDiagonal", "Deformation", "Random"}


def read(path):
    return json.loads(Path(path).read_bytes().decode("utf-8-sig"))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2,
                                     sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def scalar(value):
    return value is None or isinstance(value, (str, bool, int)) or isinstance(value, float) and math.isfinite(value)


def project(value, key="", context="", scalar_lists=False):
    """Fail-closed sequence projection. Scalars/dicts survive, signal lists do not.

    Structured row lists are metadata; scalar lists require an explicit metric,
    interval, candidate-index, prefix or score-vector contract.
    """
    if key in FORBIDDEN or key.lower().endswith(("_curve", "_curves")):
        return None
    if scalar(value):
        return value
    if isinstance(value, dict):
        result = {}
        for name, item in value.items():
            if name in FORBIDDEN or name.lower().endswith(("_curve", "_curves")):
                continue
            if isinstance(item, (list, tuple)):
                if not scalar_lists:
                    continue
                allowed = (name in SCALAR_LISTS or context in ("prefixes", "raw_score_ties", "G1_screened_score_ties")
                           or name in SCORE_METHODS and context in ("scores", "scores_by_control", "actual", "zero", "swap"))
                if all(isinstance(v, dict) for v in item):
                    result[name] = [project(v, context=name, scalar_lists=True) for v in item]
                elif allowed and all(scalar(v) for v in item):
                    result[name] = list(item)
                # Other sequences (vectors, matrices, positions) are omitted.
                continue
            if isinstance(item, dict):
                result[name] = project(item, context=name, scalar_lists=scalar_lists)
            elif scalar(item):
                result[name] = item
            else:
                raise ValueError("Unsupported/nonfinite scalar at " + str(name))
        return result
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [project(item, context=context, scalar_lists=scalar_lists) for item in value]
    raise ValueError("Uncontracted top-level vector/matrix or nonfinite data")


def binding(doc):
    result = {}
    for key in ("commit", "code", "dependencies"):
        value = doc.get(key) if isinstance(doc, dict) else None
        if isinstance(value, str):
            result[key] = value
        elif isinstance(value, dict):
            result[key] = {str(k): v for k, v in value.items() if isinstance(v, str) and HEX.fullmatch(v)}
    return result


def interval(value):
    if not isinstance(value, list) or len(value) != 2 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value) or value[0] >= value[1]:
        raise ValueError("Malformed metadata interval")
    return list(value)


def hash_manifest(stage, label):
    path = stage / "manifest.json"; doc = read(path); files = []
    if not isinstance(doc.get("files"), dict):
        raise ValueError("R67 scientific manifest must map relative paths to SHA256")
    seen = set()
    for name, expected in sorted(doc["files"].items()):
        name = name.replace("\\", "/")
        if ":" in name or Path(name).is_absolute() or ".." in Path(name).parts or name in seen:
            raise ValueError("Unsafe or duplicated scientific manifest member")
        seen.add(name); member = (stage / name).resolve()
        if not member.is_relative_to(stage.resolve()) or not member.is_file() or not HEX.fullmatch(expected) or sha(member) != expected:
            raise ValueError("Frozen scientific file changed: " + label + "/" + name)
        files.append({"stage": label, "path": name, "bytes": member.stat().st_size, "sha256": expected})
    return {"status": "ALL_MANIFEST_BYTES_REHASHED_EXACT", "manifest_sha256": sha(path),
            "binding": binding(doc.get("binding", {})), "files": files, "artifact_bytes_copied": False}


def stage_ready(stage, required):
    if not stage.exists():
        return "NOT_AVAILABLE_NOT_CLAIMED_RUN"
    if not (stage / "manifest.json").is_file() or not all((stage / p).is_file() for p in required):
        return "INCOMPLETE_NOT_PUBLISHED_AS_FINISHED"
    return "SEALED"


def registration(doc):
    result = project({k: v for k, v in doc.items() if k not in ("binding", "protocol")})
    result["binding"] = binding(doc.get("binding", {}))
    before, after = doc.get("weights_before"), doc.get("weights_after")
    if before is not None or after is not None:
        if not isinstance(before, str) or not HEX.fullmatch(before) or before != after:
            raise ValueError("Frozen R41 network fingerprint changed or is unverified")
        result["frozen_network_digest_unchanged"] = True
    else:
        result["frozen_network_digest_unchanged"] = None
    return result


def _spread(doc):
    result = project(doc)
    for name in ("pairwise_max_abs", "pairwise_mean_abs"):
        matrix = doc.get(name, [])
        if matrix:
            n = len(matrix)
            if any(not isinstance(row, list) or len(row) != n for row in matrix):
                raise ValueError("qspread must address a square common candidate bank")
            values = [matrix[i][j] for i in range(n) for j in range(i+1, n)]
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in values):
                raise ValueError("Invalid qspread scalar diagnostics")
            result[name + "_off_diagonal_summary"] = {
                "pairs": len(values), "mean": sum(values)/len(values) if values else None,
                "minimum": min(values) if values else None, "maximum": max(values) if values else None,
                "independent_sample_count": None}
    return result


def public_banks(doc):
    result = []
    for record in doc:
        bank = record["bank"]
        public = project({k: v for k, v in bank.items() if k not in ("candidates", "qspread", "duplicate_paths")}, scalar_lists=True)
        public["qspread"] = _spread(bank.get("qspread", {}))
        public["candidates"] = [project(candidate, scalar_lists=True) for candidate in bank["candidates"]]
        public["duplicate_paths"] = project(bank.get("duplicate_paths", []), scalar_lists=True)
        if public["actual_unique_bank_q_count"] != len(public["candidates"]):
            raise ValueError("Public candidate count differs from the genuine unpadded bank")
        item = {"case_key": record.get("case_key", record.get("key")), "bank": public}
        if "injection" in record:
            item["injection_scalar_proof"] = project(record["injection"])
        result.append(item)
    return result


def population(doc):
    keys = ("key", "case_key", "source_uid", "event_uid", "parent_case_key", "parent_record_key", "parent_side",
            "music_id", "songidx", "audio_sha256", "source_R_sha256", "baseline_core_R_sha256", "eligible",
            "dataset", "role", "group", "recording_uid", "raw_motion_sha256", "exposure", "clock_scope")
    output = project({k: v for k, v in doc.items() if k not in ("cases", "sources")})
    output["cases"] = []
    for case in doc.get("cases", []):
        item = {k: case[k] for k in keys if k in case and scalar(case[k])}
        for name in ("source_interval", "target_interval", "audio_interval"):
            if case.get(name) is not None:
                item[name] = interval(case[name])
        output["cases"].append(item)
    return output


def detector_consistency(rows):
    output = []
    for row in rows:
        diagnostic = row["result"]
        item = {"case_key": row["case_key"], "method": row["method"], "source_uid": row["source_uid"],
                "diagnostic": project(diagnostic, scalar_lists=True)}
        mapped = diagnostic.get("mapped_alignment") or {}; redetected = diagnostic.get("output_alignment") or {}
        old = mapped.get("primary_half_beat_half_accent"); new = redetected.get("primary_half_beat_half_accent")
        item.update(mapped_source_music_score_as_R66_recorded=old, redetected_output_music_score_as_R66_recorded=new,
                    original_ROIs_differ=True,
                    score_difference_scope="ORIGINAL_R66_ALIGNMENT_ROIS_DIFFER; NO_SYSTEMATIC_OVERESTIMATION_INFERENCE_FROM_THESE_FIELDS",
                    detector_counts_scope="ONE_TO_ONE_REGION_MATCHES_WITHIN3_REPORTED_SEPARATELY_FROM_PROXY_SCORE")
        output.append(item)
    return output


def r66_common_roi(repo, cfg, inputs, expected_reward_sha=None):
    """New declared publication diagnostic, not a modification of R67 science.

    Both peak sets use local [3,237), genuine full audio evidence, and the exact
    original R64 weighted-accent function. Only the source set additionally
    requires its genuine original phase evidence to be wholly inside the Core.
    """
    import numpy as np
    old = repo / "outputs/rhythm_r66_01/main"
    manifest_path = old / "artifacts_manifest.json"
    manifest = read(manifest_path)
    def parent_file(name):
        path = old/name
        expected = manifest["files"].get(name)
        if not isinstance(expected,str) or not HEX.fullmatch(expected) or not path.is_file() or sha(path) != expected:
            raise ValueError("Frozen R66 common-ROI input changed: " + name)
        inputs.append(path)
        return path
    result_path = parent_file("result.json")
    population_path = parent_file("population.json")
    records = read(result_path)["records"]
    cases = {row["key"]:row for row in read(population_path)["cases"]}
    directory = repo / "reports/r64_release/scientific_files.json"
    seal = read(directory)
    evidence = [row for row in seal["files"] if row["path"] == "main/music_clocks.json"]
    if len(evidence) != 1:
        raise ValueError("Frozen R64 music clock seal must have one unambiguous member")
    clock_path = repo / "outputs/timing_r64_01/main/music_clocks.json"
    clock_sha = sha(clock_path)
    if (clock_sha != evidence[0]["sha256"] or clock_path.stat().st_size != evidence[0]["bytes"]
            or cfg.get("r64_music_clocks_sha256") != clock_sha):
        raise ValueError("Original published R64 clock bytes/hash or preregistered SHA differs")
    clocks = read(clock_path)
    code = repo / "src/dunhuang/research/timing_r64/music.py"
    code_sha = sha(code)
    if expected_reward_sha is not None and code_sha != expected_reward_sha:
        raise ValueError("R64 reward source differs from sealed R67 dependency binding")
    # Execute just the exact frozen pure function, not heavyweight audio loaders.
    module = ast.parse(code.read_bytes().decode("utf-8-sig"))
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_reward"]
    if len(functions) != 1:
        raise ValueError("Exact original R64 _reward function missing")
    namespace = {"np":np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(code),"exec"),namespace)
    reward = namespace["_reward"]
    inputs.extend((manifest_path,directory,clock_path,code))
    def evaluate(local_frames,A,clock):
        events = np.asarray(local_frames,dtype=float)+A
        accents = np.asarray(clock["accent_frames"],dtype=float)
        strength = np.interp(accents,np.arange(clock["frames"]),clock["accent_strength"])
        b = reward(events,clock["beat_frames"],3)
        a = reward(events,accents,3,strength)
        return {"primary":.5*b["score"]+.5*a["score"] if b["score"] is not None and a["score"] is not None else None,
                "beat":project(b),"weighted_accent":project(a)}
    rows=[]
    for record in records:
        case = cases[record["case_key"]]
        for key in ("source_uid","source_interval","target_interval","songidx"):
            if record[key] != case[key]:
                raise ValueError("R66 common-ROI row differs from its frozen source/music identity")
        clock = clocks[str(case["songidx"])]
        if clock["audio_sha256"] != case["audio_sha256"]:
            raise ValueError("R66 source and true full audio clock differ")
        A,B = record["target_interval"]
        if B-A != 240 or not 0 <= A < B <= clock["frames"]:
            raise ValueError("Common ROI requires the same real 240-frame audio support")
        row={"case_key":record["case_key"],"method":record["method"],"source_uid":record["source_uid"],
             "source_interval":interval(record["source_interval"]),"target_interval":interval(record["target_interval"]),
             "audio_sha256":case["audio_sha256"],"status":record["status"],
             "mapped_peak_count":None,"redetected_peak_count":None,"mapped_source_proxy":None,
             "redetected_output_proxy":None,"mapped_minus_output_primary":None}
        meta=record.get("metrics")
        if not meta:
            row["reason"]=(record.get("warp") or {}).get("reason","NO_ACTUAL_OUTPUT_METRICS")
            rows.append(row);continue
        mapped=[float(p["target_frame"]) for p in meta["source_phase_diagnostics"]
                if p["type"].endswith("speed_peak") and p["source_evidence_whole_inside_core"]
                and 3 <= p["target_frame"] < 237]
        output=[float(p["frame"]) for p in meta["actual_output_peaks"].get("peaks",[]) if 3 <= p["frame"] < 237]
        old_score,new_score=evaluate(mapped,A,clock),evaluate(output,A,clock)
        x,y=old_score["primary"],new_score["primary"]
        row.update(mapped_peak_count=len(mapped),redetected_peak_count=len(output),mapped_source_proxy=old_score,
                   redetected_output_proxy=new_score,mapped_minus_output_primary=x-y if x is not None and y is not None else None,
                   reason=None if x is not None and y is not None else "EMPTY_PEAK_SET_OR_AUDIO_EVIDENCE_NOT_A_ZERO_SCORE")
        rows.append(row)
    summaries={}
    for method in sorted({r["method"] for r in rows}):
        own=[r for r in rows if r["method"]==method];groups=defaultdict(list)
        for row in own:groups[(row["source_uid"],tuple(row["source_interval"]))].append(row)
        source_means=[]
        for (uid,span),group in sorted(groups.items()):
            scored=[r for r in group if r["mapped_minus_output_primary"] is not None]
            values=[r["mapped_minus_output_primary"] for r in scored]
            source_means.append({"source_uid":uid,"source_interval":list(span),"registered_music_cases":len(group),
                "evaluable_common_ROI_cases":len(scored),
                "mean_mapped_minus_output_primary":sum(values)/len(values) if values else None,
                "positive_gap_cases":sum(v>1e-12 for v in values),"tied_gap_cases":sum(abs(v)<=1e-12 for v in values),
                "negative_gap_cases":sum(v < -1e-12 for v in values),
                "mean_mapped_peak_count":sum(r["mapped_peak_count"] for r in scored)/len(scored) if scored else None,
                "mean_redetected_peak_count":sum(r["redetected_peak_count"] for r in scored)/len(scored) if scored else None})
        available=[r["mean_mapped_minus_output_primary"] for r in source_means if r["mean_mapped_minus_output_primary"] is not None]
        summaries[method]={"registered_cases":len(own),"registered_source_intervals":len(groups),
            "evaluable_cases":sum(r["mapped_minus_output_primary"] is not None for r in own),
            "evaluable_source_intervals":len(available),"source_interval_means":source_means,
            "source_interval_balanced_mean_mapped_minus_output_primary":sum(available)/len(available) if available else None,
            "positive_gap_source_interval_means":sum(v>1e-12 for v in available),
            "tied_gap_source_interval_means":sum(abs(v)<=1e-12 for v in available),
            "negative_gap_source_interval_means":sum(v < -1e-12 for v in available),"ci95":None}
    return {"schema":"R67_PUBLICATION_R66_COMMON_ROI_PROXY_GAP_V1","records":rows,"methods":summaries,
        "registered_records":len(rows),"ROI_local_half_open":[3,237],"source_must_have_whole_evidence_inside_Core":True,
        "output_peaks_source":"ALREADY_REDETECTED_ON_ACTUAL_R66_OUTPUT_NO_NEW_DETECTOR",
        "audio_clock":"TRUE_FULL_AUDIO_A_PLUS_LOCAL_FRAME_ONCE_NO_SHIFT_CONTROL",
        "reward":{"implementation":"ORIGINAL_R64_MUSIC_PY__reward","sigma_frames":3,"beat_weight":.5,"weighted_accent_weight":.5,
                  "accent_strength":"INTERPOLATED_FROM_FROZEN_FULL_CLOCK_AT_ACCENT_EVENT_FRAMES"},
        "derivation":"DECLARED_PUBLICATION_DIAGNOSTIC_FROM_FROZEN_R66_R64; NOT_PART_OF_RUNNING_R67_SCIENTIFIC_MANIFEST",
        "independent_replay_coverage":"NOT_AUTOMATICALLY_INHERITED_FROM_R67_REPLAY",
        "balance":"CROSS_MUSIC_CASES_WITHIN_SOURCE_INTERVAL_THEN_EQUAL_SOURCE_INTERVALS; NO_12_CASE_IID_CI",
        "scope":"DESCRIPTIVE_OVERESTIMATION_DIAGNOSTIC_NOT_HUMAN_ACCENT_TRUTH",
        "difference_definition":"MAPPED_SOURCE_PROXY_MINUS_REDETECTED_OUTPUT_PROXY; POSITIVE_IS_HIGHER_MAPPED_PROXY",
        "event_count_or_peak_location_difference_is_not_a_bug_or_artistic_label":True,
        "inputs":{"R66_manifest_sha256":sha(manifest_path),"R66_result_sha256":sha(result_path),
                  "R66_population_sha256":sha(population_path),"R64_clock_sha256":clock_sha,
                  "R64_public_scientific_directory_sha256":sha(directory),"R64_reward_implementation_sha256":code_sha},
        "human_reviewed":False,"artistic_quality":"NOT_EVALUABLE","Native24":"NOT_EVALUABLE","can_route":False}


def _ties(values):
    available = [(i, v) for i, v in enumerate(values) if v is not None]
    if not available:
        return []
    if not all(isinstance(v, (int,float)) and not isinstance(v, bool) and math.isfinite(v) for _,v in available):
        raise ValueError("Nonfinite candidate score")
    maximum = max(v for _, v in available)
    return [i for i, v in available if abs(v-maximum) <= 1e-12]


def main_projection(stage, banks, records):
    sealed_files = read(stage/"manifest.json")["files"]
    by_case = defaultdict(dict)
    for row in records:
        if row["candidate_uid"] in by_case[row["case_key"]]:
            raise ValueError("Duplicate executed main candidate")
        by_case[row["case_key"]][row["candidate_uid"]] = row
    rows, cases, ranks = [], [], []
    for entry in banks:
        key = entry.get("case_key", entry.get("key")); bank = entry["bank"]
        ids = [c["uid"] for c in bank["candidates"]]
        executed = by_case.pop(key, {})
        if set(ids) != set(executed):
            raise ValueError("Every shared-bank candidate must retain its executed score/body/connection row")
        own = [executed[uid] for uid in ids]
        relative = "selections/" + key + ".json"
        if relative not in sealed_files:
            raise ValueError("Executed rank-state file is outside its original scientific manifest")
        raw = read(stage / relative)
        ranker_names = sorted({name for row in own for name in row["scores"]})
        for name in ranker_names:
            scores = [row["scores"].get(name) for row in own]
            raw_ties = _ties(scores)
            screened = _ties([v if row["connectivity"].get("passes_original_screen") is True else None
                              for row,v in zip(own,scores)])
            if raw["raw_score_ties"].get(name) != raw_ties or raw["G1_screened_score_ties"].get(name) != screened:
                raise ValueError("Saved raw/G1-screened rank state no longer matches the executed table")
            ranks.append({"case_key": key, "method": name, "raw_score_tied_indices": raw_ties,
                          "raw_score_tied_candidate_uids": [ids[i] for i in raw_ties],
                          "G1_screened_score_tied_indices": screened,
                          "G1_screened_score_tied_candidate_uids": [ids[i] for i in screened],
                          "screened_abstained": not screened,
                          "display_scope": raw.get("selection_display"), "no_fallback": True})
        cases.append({"case_key": key, "registered_candidates": len(own),
                      "legal_path_count": bank["legal_path_count"],
                      "G1_constructed": sum(row["connectivity"].get("constructible") is True for row in own),
                      "G1_passed_original_screen": sum(row["connectivity"].get("passes_original_screen") is True for row in own),
                      "numerical_screen_not_Native24_or_quality": True})
        for row in own:
            public = project({k:v for k,v in row.items() if k not in ("scores", "source_interval", "target_interval")})
            public["scores"] = {name: value for name,value in row["scores"].items()}
            public["source_interval"] = interval(row["source_interval"])
            public["target_interval"] = interval(row["target_interval"])
            before = row.get("encoding",{}).get("model_digest_before")
            after = row.get("encoding",{}).get("model_digest_after")
            if before != after or not isinstance(before,str) or not HEX.fullmatch(before):
                raise ValueError("Main candidate does not retain its unchanged frozen encoder fingerprint")
            public["frozen_encoder_unchanged"] = True
            rows.append(public)
    if by_case:
        raise ValueError("Executed main rows exist outside the frozen candidate banks")
    return {"records": rows, "cases": cases, "ranks": ranks,
            "registered_candidates": len(rows), "registered_cases": len(cases),
            "G1_constructed_candidate_checks": sum(c["G1_constructed"] for c in cases),
            "G1_passed_original_screen_candidate_checks": sum(c["G1_passed_original_screen"] for c in cases),
            "candidate_checks_not_necessarily_unique_geometry_solves": True,
            "quality_winner": None, "can_route": False}


def independent(path, original_manifest):
    if not path.is_file():
        return {"status": "NOT_AVAILABLE", "scientific_outputs_exact": None}
    doc = read(path)
    public = project({k:v for k,v in doc.items() if k != "binding"}, scalar_lists=True)
    public["binding"] = binding(doc.get("binding",{}))
    match = original_manifest.is_file() and doc.get("source_manifest_sha256") == sha(original_manifest)
    public["source_manifest_binding_exact"] = match
    public["status"] = "EXACT" if doc.get("scientific_outputs_exact") is True and not doc.get("differences") and match else "NOT_EXACT_OR_UNBOUND"
    public["receipt_sha256"] = sha(path)
    return public


RECEIPT_KEYS = {"schema","status","mode","stage","phase","commit","code_commit","server_commit","github_commit",
                "executed_commit","exit_code","exit_status","passed","pid","tmux_session","started_at","finished_at",
                "completed_at","created_at","timestamp","sha256","result_sha256","log_sha256","manifest_sha256",
                "receipt_sha256","source_manifest_sha256","scientific_outputs_exact","training_steps","neural_training_steps",
                "new_planner_searches","optimizer_calls","geometry_bridge_solves","bridge_solves","tests_passed",
                "tests_skipped","success","reason","cleanup_status"}


def receipt(doc):
    output = {}
    if not isinstance(doc,dict):
        return output
    for key,value in doc.items():
        if key in RECEIPT_KEYS and scalar(value):
            output[key] = value
        elif isinstance(value,dict) and key not in FORBIDDEN:
            child = receipt(value)
            if child:
                output[key] = child
    return output


def failed_receipt(doc):
    if not isinstance(doc,dict):
        return False
    for key,value in doc.items():
        if key in ("exit_code","exit_status") and type(value) is int and value != 0:
            return True
        if key == "status" and isinstance(value,str) and any(word in value.upper() for word in ("FAILED","FAILURE","ERROR")):
            return True
        if isinstance(value,dict) and failed_receipt(value):
            return True
    return False


def archived_failures(root, inputs):
    """A failed partial run is evidence, never a completed scientific stage."""
    directory = root/"failures"; records=[]
    if not directory.is_dir():
        return {"records":records,"completed_scientific_stages":0}
    for folder in sorted(p for p in directory.iterdir() if p.is_dir()):
        item={"archive":folder.name,"scope":"RETAINED_FAILED_RUN_NOT_COMPLETED_MAIN",
              "execution_receipts":[],"log_hashes":[],"partial_banks":None,
              "raw_arrays_or_log_text_copied":False}
        for path in sorted(folder.glob("execution*.json")):
            inputs.append(path)
            item["execution_receipts"].append({"path":path.relative_to(root).as_posix(),
                "source_sha256":sha(path),"projection":receipt(read(path))})
        for path in sorted(folder.glob("*.log")):
            inputs.append(path)
            item["log_hashes"].append({"path":path.relative_to(root).as_posix(),
                                      "bytes":path.stat().st_size,"sha256":sha(path)})
        banks=folder/"partial_main/banks.json"
        if banks.is_file():
            inputs.append(banks)
            item["partial_banks"]={"source_sha256":sha(banks),"banks":public_banks(read(banks)),
                                    "scope":"PREPARED_SHARED_BANK_METADATA_NOT_EXECUTED_CANDIDATE_RESULTS"}
        records.append(item)
    return {"records":records,"completed_scientific_stages":0,
            "archive_hashes_are_not_completed_scientific_manifest":True}


PRESENTATION_NAMES=("phase_presentation_fix.json","phase_browser_acceptance.json","phase_delivery.json",
                    "main_presentation_fix.json","main_browser_acceptance.json","main_delivery.json")


def presentation_receipts(repo,root,execution,explicit,science,inputs):
    paths=[]
    for folder in (execution,root/"receipts",root/"delivery"):
        paths.extend(folder/name for name in PRESENTATION_NAMES if (folder/name).is_file())
    for path in explicit:
        path=Path(path);path=path.resolve() if path.is_absolute() else (repo/path).resolve()
        if not path.is_file():
            raise ValueError("Explicit presentation receipt is missing: "+str(path))
        paths.append(path)
    records=[];overlays=[]
    sealed_html={mode:next((v["sha256"] for v in science.get(mode,{}).get("files",[])
                            if v["path"]=="viewer/index.html"),None)
                 for mode in ("phase_audit","main")}
    for path in sorted(set(paths)):
        doc=read(path);inputs.append(path)
        public=project({k:v for k,v in doc.items() if k!="files"})
        checks=doc.get("registeredCasesChecked")
        if checks is not None:
            if not isinstance(checks,list) or not all(isinstance(row,dict) for row in checks):
                raise ValueError("Browser case checks must retain scalar metadata records")
            public["registeredCasesChecked"]=[project(row) for row in checks]
            selected=[row.get("selected") for row in checks]
            public["registeredCaseCheckSummary"]={
                "checked_record_count":len(checks),
                "distinct_selected_indices":len({value for value in selected if type(value) is int}),
                "all_records_have_four_panels":bool(checks) and all(row.get("panels")==4 for row in checks),
                "scope":"RECORDED_UI_CHECKS_NOT_ARTISTIC_OR_PHYSICAL_QUALITY"}
        public["files"]=[]
        for row in doc.get("files",[]):
            item={k:v for k,v in row.items() if k in ("path","old_sha256","new_sha256","sha256","bytes","changed_constants","change_scope","scope")}
            if "path" in item:
                name=str(item["path"]).replace("\\","/")
                if ":" in name or Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("Presentation receipt must identify relative artifacts, not private absolute paths")
                item["path"]=name
            for key in ("old_sha256","new_sha256","sha256"):
                if key in item and (not isinstance(item[key],str) or not HEX.fullmatch(item[key])):
                    raise ValueError("Presentation receipt has an invalid artifact digest")
            public["files"].append(item)
            parts=Path(item.get("path","")).parts
            mode=parts[-3] if len(parts)>=3 and parts[-2:]==("viewer","index.html") else None
            if mode in sealed_html and "new_sha256" in item:
                old_html=sealed_html[mode]
                overlays.append({"stage":mode,"artifact":item["path"],"sealed_scientific_html_sha256":old_html,
                    "receipt_old_sha256":item.get("old_sha256"),"declared_local_overlay_sha256":item["new_sha256"],
                    "scope":item.get("scope",item.get("change_scope")),
                    "old_hash_matches_scientific_manifest":old_html is not None and item.get("old_sha256")==old_html,
                    "local_file_rehashed_by_this_exporter":False,
                    "original_delivery_sha_check_applies_before_overlay_only":True})
        records.append({"receipt":path.name,"source_sha256":sha(path),"projection":public,
                        "scope":"PRESENTATION_OR_UI_RECEIPT_NOT_ARTISTIC_OR_SCIENTIFIC_REPLAY"})
    return {"records":records,"phase_html_overlays":[v for v in overlays if v["stage"]=="phase_audit"],
            "main_html_overlays":[v for v in overlays if v["stage"]=="main"],"html_overlays":overlays,
            "local_presentation_may_differ_from_server_scientific_html":bool(overlays),
            "browser_acceptance_does_not_grant_quality_or_Native":True,
            "browser_receipt_overlay_binding":"ONLY_AS_EXPLICITLY_RECORDED; NOT_INFERRED_FROM_SIMULTANEOUS_EXISTENCE"}


def export(repo=".", out="reports/r67_release", root="outputs/sdmr_warp_rank_01",
           execution="outputs/sdmr_warp_rank_execution_01", presentation=()):
    repo = Path(repo).resolve()
    resolve = lambda path: Path(path).resolve() if Path(path).is_absolute() else (repo/path).resolve()
    out, root, execution = resolve(out), resolve(root), resolve(execution)
    if out.exists() or out.is_relative_to(root) or out.is_relative_to(execution):
        raise ValueError("Publication directory must be fresh and outside scientific/execution roots")
    files, science, inputs, states = {}, {}, [], {}
    required = {"phase_audit": ("summary.json","r66_detector_consistency.json","registration.json"),
                "main": ("records.json","banks.json","population.json","registration.json","viewer_manifest.json"),
                "controls": ("summary.json","rows.json","banks.json","registration.json")}
    for mode,names in required.items():
        stage = root/mode; states[mode] = stage_ready(stage,names)
        if states[mode] != "SEALED":
            continue
        science[mode] = hash_manifest(stage,mode)
        inputs.extend(stage/name for name in names)
        inputs.append(stage/"manifest.json")
        reg = read(stage/"registration.json")
        files[mode+"_registration.json"] = registration(reg)
        if mode == "phase_audit":
            value = read(stage/"summary.json")
            files["phase_audit_all_lags.json"] = project(value,scalar_lists=True)
            expected_lags = reg.get("protocol",{}).get("lags")
            if expected_lags is not None and any([v["lag"] for v in row["lags"]] != expected_lags for row in value["records"]):
                raise ValueError("Phase audit lost/reordered registered lags; do not publish best-lag selection")
            files["r66_detector_consistency.json"] = detector_consistency(read(stage/"r66_detector_consistency.json"))
        elif mode == "main":
            banks = read(stage/"banks.json"); records = read(stage/"records.json")
            files["main_banks.json"] = public_banks(banks)
            files["main_executed_candidates_and_ranks.json"] = main_projection(stage,banks,records)
            files["main_population.json"] = population(read(stage/"population.json"))
            inputs.extend(sorted((stage/"selections").glob("*.json")))
            viewer = read(stage/"viewer_manifest.json")
            files["main_viewer_hashes.json"] = {"files": viewer["files"], "media_bytes_copied": False,
                "browser_acceptance_or_media_delivery_inferred": False,
                "scope": "HASH_DIRECTORY_ONLY_NO_HTML_COORDINATES_OR_AUDIO_BYTES"}
        else:
            files["controls_summary.json"] = project(read(stage/"summary.json"),scalar_lists=True)
            files["controls_rows.json"] = project(read(stage/"rows.json"),scalar_lists=True)
            files["controls_banks.json"] = public_banks(read(stage/"banks.json"))
    if not science:
        raise ValueError("No sealed R67 scientific stage is available")
    for mode in required:
        path = root/("replay_"+mode)/"independent.json"
        files["independent_"+mode+".json"] = independent(path,root/mode/"manifest.json")
        if path.is_file():
            inputs.append(path)
    receipts = []
    if execution.is_dir():
        for path in sorted(execution.rglob("*.json")):
            receipts.append({"receipt":path.relative_to(execution).as_posix(),"source_sha256":sha(path),
                             "projection":receipt(read(path)),"scope":"WHITELISTED_EXECUTION_METADATA"})
            inputs.append(path)
    archives=archived_failures(root,inputs)
    files["archived_failures.json"]=archives
    archived_receipts=[value for archive in archives["records"] for value in archive["execution_receipts"]]
    failures=[value for value in receipts+archived_receipts if failed_receipt(value["projection"])]
    # Copying an old failed receipt into an archive must not inflate run counts.
    failures=list({value["source_sha256"]:value for value in failures}.values())
    files["execution_receipts.json"] = {"records":receipts,"retained_failed_receipts":failures,
                                        "missing_receipt_does_not_imply_success":True}
    files["presentation_receipts.json"]=presentation_receipts(repo,root,execution,presentation,science,inputs)
    files["scientific_files.json"] = {"stages":science,"raw_artifact_bytes_copied":False}
    protocol = repo/"src/dunhuang/research/sdmr_warp_rank/protocol.json"
    if not protocol.is_file():
        raise ValueError("Original R67 checked-out protocol is required")
    protocol_bytes = protocol.read_bytes(); inputs.append(protocol)
    cfg = json.loads(protocol_bytes.decode("utf-8-sig"))
    reward_hashes = {value["binding"].get("dependencies",{}).get("src/dunhuang/research/timing_r64/music.py")
                     for value in science.values()}
    reward_hashes.discard(None)
    if len(reward_hashes) != 1:
        raise ValueError("One unambiguous sealed R64 reward implementation hash is required")
    files["R66_common_roi_proxy_gap.json"] = r66_common_roi(repo,cfg,inputs,reward_hashes.pop())
    files["qualification.json"] = {
        "stages":states,"human_reviewed":False,"independent_human_labels":0,"semantic_event_truth":"UNKNOWN",
        "source_commit_by_sealed_stage":{mode:value["binding"].get("commit") for mode,value in science.items()},
        "stage_commits_may_differ_after_recorded_runtime_bugfix":True,"retained_failed_execution_receipts":len(failures),
        "failed_partial_main_banks_are_not_completed_main_results":True,
        "local_P0_presentation_overlay_is_not_server_scientific_html":True,
        "local_main_presentation_overlay_is_not_server_scientific_html":True,
        "artistic_quality":"NOT_EVALUABLE","Native24":"NOT_EVALUABLE","quality_winner":None,
        "operational_qualified":False,"can_route":False,"neural_training_steps":0,"new_planner_searches":0,
        "hypothesis":"FROZEN_R41_WITH_LEGAL_COMMON_WARP_BANK_AND_READOUT_ONLY",
        "P0_scope":"ALL_REGISTERED_LAGS_DESCRIPTIVE; NO_BEST_LAG_REWARD_SELECTION",
        "R66_common_ROI_scope":"DESCRIPTIVE_OVERESTIMATION_DIAGNOSTIC_NOT_HUMAN_ACCENT_TRUTH",
        "R66_common_ROI_replay":"NEW_DECLARED_PUBLICATION_DERIVATION_NOT_AUTOMATICALLY_COVERED_BY_R67_ARRAY_REPLAY",
        "main_scope":"HISTORICALLY_EXPOSED_CHANGE; BODY22_CORE_AND_RECOMPUTED_FIXED48_G1_NUMERICAL_DIAGNOSTICS",
        "controls_scope":"SIX_EXPOSED_AIST_VAL_MUSIC_GROUPS; KNOWN_COMPOSED_CLOCK_NOT_ARTISTIC_TRUTH",
        "interpolation_scope":"BODY_DOUBLE_RESAMPLING_ERROR_IS_NOT_NEURAL_REPERFORMANCE_ERROR",
        "candidate_capacity_scope":"LEGAL_PATH_COUNT_IS_NOT_UNIQUE_Q_COUNT; COST_PREFIX_DEDUP_MAY_YIELD_K_SHORTFALL",
        "rank_scope":"RAW_AND_G1_SCREENED_STATES_SEPARATE; NO_FALLBACK_TO_MAKE_FAILURE_PASS",
        "new_independent_music":False,"raw_arrays_tokens_audio_media_published":False}
    files["publication_inputs.json"] = {"files":[{
        "path":p.relative_to(repo).as_posix() if p.is_relative_to(repo) else p.name,
        "bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(set(inputs))]}
    out.mkdir(parents=True,exist_ok=False)
    for name,value in files.items():
        write(out/name,value)
    (out/"protocol.json").write_bytes(protocol_bytes)
    main = files.get("main_executed_candidates_and_ranks.json",{})
    report = (
        "# R67 冻结 R41、合法 Warp 候选与多尺度读出：公开诊断\n\n"
        f"阶段封存状态：{json.dumps(states,ensure_ascii=False)}。\n\n"
        f"各阶段原始代码提交分别保留：{json.dumps({mode:value['binding'].get('commit') for mode,value in science.items()},ensure_ascii=False)}。"
        f"已保留 {len(failures)} 份可识别失败运行回执；未封存的失败运行不被发布成完成实验。P0 与修复后主实验可以来自不同代码修订，不能宣称同一提交全部成功。\n\n"
        "main_706 失败归档若存在，仅发布失败回执、日志哈希与剔除逐帧 q 的候选准备元数据，不将 partial_main 解释为已执行的主实验。"
        "本地 P0 微秒帧显示修正与主候选连接状态显示修正，其 overlay 新哈希分别与各自原服务器封存 HTML 分列；原回传验 SHA 仅适用于各自修正前文件。"
        "浏览交互验收见独立 presentation_receipts，不等于科学复验或艺术质量评价。\n\n"
        f"主候选已登记 {main.get('registered_cases','未完成')} 个案例、{main.get('registered_candidates','未完成')} 次候选检查；"
        f"G1 构造 {main.get('G1_constructed_candidate_checks','未完成')} 次、原数值筛查通过 {main.get('G1_passed_original_screen_candidate_checks','未完成')} 次。"
        "这些是候选检查次数；实际几何调用及缓存复用须查看 registration，不能混为独立求解次数。\n\n"
        "P0 保留全部登记滞后，不挑最佳滞后。R66 旧有 alignment 两组 ROI 不同，只保留原始读数，不能据其差值判断系统高估。"
        "新增 R66_common_roi_proxy_gap 在相同 [3,237) 输出区间上，使用原 R64 sigma3 与加权重音奖励，分别记录映射源峰、输出重检峰的数量及分数。"
        "mapped−output 差先在来源区间内跨音乐平均，再作等来源区间描述；不同峰数或位置本身不是实现错误。"
        "这项预先声明的公开派生诊断不在正在执行的 R67 科学数组封存内，不能自动继承其独立复验。它也不是人工重音真值或艺术质量证据。\n\n"
        "主实验保留每条合法候选的分数、来源动态标量与连接筛查状态，并保留原始排序和 G1 筛后排序；筛后无候选即弃权。"
        "合法路径数量不是不同实际时钟数量，K 短缺不靠重复填充。\n\n"
        "配对控制按 untouched / injected 分开，采用相同候选域、平局期望误差、prefix 与 full-bank oracle regret。"
        "2×2 交互量先在同一 query 上计算，再按录制及音乐组均衡；六个已曝光音乐组不构成新音乐盲测。"
        "原始与两次 SO(3) 插值后的身体角误差是重采样误差，不是神经动作重演误差。\n\n"
        "冻结 checkpoint 与网络指纹前后状态见 registration；不存在本轮神经训练。"
        "独立复验只在 scientific_outputs_exact=true、差异清单为空、且源 manifest SHA 绑定一致时标 EXACT；缺失或未绑定不算通过。\n\n"
        "本发布不包含坐标、旋转、逐帧 q、token、音频、观看页内嵌轨迹或 G1 内部矩阵。"
        "没有人工语义真值、艺术质量胜出或 Native24 资格结论，can_route=false。\n")
    (out/"报告.md").write_bytes(report.encode("utf-8"))
    public = [{"path":p.name,"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(out.iterdir()) if p.is_file()]
    write(out/"public_manifest.json",{"files":public,"self_excluded":True,"scope":"METADATA_SCALARS_ONLY_NO_QUALITY_WINNER"})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",default=".")
    parser.add_argument("--out",default="reports/r67_release")
    parser.add_argument("--root",default="outputs/sdmr_warp_rank_01")
    parser.add_argument("--execution",default="outputs/sdmr_warp_rank_execution_01")
    parser.add_argument("--presentation-receipt",action="append",default=[])
    args = parser.parse_args()
    print(export(args.repo,args.out,args.root,args.execution,args.presentation_receipt))


if __name__ == "__main__":
    main()
