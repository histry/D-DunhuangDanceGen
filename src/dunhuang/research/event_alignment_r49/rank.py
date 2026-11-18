"""Score actual R48 clocks and publish watchable source-reference previews."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

from dunhuang.research.complete_r36.core import Library, read, relative_fk, rotation_sample, sha, write
from .common import BASELINE, Frozen, MULTI_WEIGHT, R48, WARP_WEIGHT, audio_signals, cosine, multiscale, sealed

SONGS = Path("outputs/transition_r37_02")
CATALOG = Path("src/dunhuang/research/complete_r36/config.json")
PREVIEW_ROOT = Path("outputs/event_alignment_r49_01")


def source_clock(event, variant, linear=False):
    a, b = event["source_interval"]
    m, n = variant["target_interval"]
    warp = np.asarray(variant["clock"]["time_warp"], dtype=float)
    increments = np.diff(warp, axis=0)
    if (not np.isfinite(warp).all() or not np.array_equal(warp[[0,-1]], [[a,m],[b-1,n-1]])
            or (increments <= 0).any()):
        raise ValueError("Invalid frozen event time graph")
    rates = increments[:,1]/increments[:,0]
    if np.any((rates < .75-1e-8) | (rates > 1.33+1e-8)):
        raise ValueError("Frozen clock violates E1-W rate limits")
    source = np.interp(np.arange(m,n), warp[:,1], warp[:,0])
    for block in variant["clock"]["windows"]:
        lo, hi = block["target_interval"]
        if not np.array_equal(source[lo-m:hi-m], block["source_positions"]):
            raise ValueError("Window changes the common source clock")
    if linear:
        source = np.linspace(a,b-1,n-m)
        rates = np.array([(n-m-1)/(b-a-1)])
    penalty = float(np.average(np.abs(np.log(rates)), weights=None if linear else increments[:,0]))
    return source, penalty


def main_scores(net, rotations, positions, music, energy, target_activity):
    R = rotation_sample(rotations, positions)
    values, mask = net.motion(R)
    motion, valid = net.motion_tokens(values, mask)
    tokens, music_valid = music
    common = valid & music_valid
    if not common.any():
        return None
    fk = relative_fk(R, net.joints, net.parents)
    activity = np.linalg.norm(np.diff(fk, axis=0, prepend=fk[:1])*30, axis=-1).mean(1)
    return dict(local=cosine(tokens[common], motion[common]),
                multi=multiscale(energy, activity),
                heuristic=-float(abs(np.clip(activity.mean()/2, 0, 1)-target_activity)),
                valid_frames=int(common.sum()), total_tokens=len(common),
                used_music=True)


def run(out, controls=False):
    if controls and not read(PREVIEW_ROOT/"browser_receipt.json")["passed"]:
        raise ValueError("Deliver and inspect the main effect before ablations")
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh output required")
    net = Frozen()
    from dunhuang.research.song_transfer.music import runtime
    runtime(read("inputs/clock_assets_v2.json"))
    from evaluation.paired_alignment_observations_v3 import full_audio_observations
    prior = sealed(R48, "result.json")
    catalog = read(CATALOG)
    if sha(catalog["catalog"]) != catalog["catalog_sha256"]:
        raise ValueError("Frozen ChangE catalog mismatch")
    lib = Library(catalog["catalog"])
    events = read(prior)["events"]
    grouped = defaultdict(list)
    for event in events:
        lo, hi = event["source_validity"]["accepted_component"]
        a,b = event["source_interval"]
        if not (lo <= a < b <= hi and event["source_validity"]["unknown_frames_used"] == 0):
            raise ValueError("Event escapes accepted source support")
        entry = lib.rows[event["source_uid"]][0]
        if entry["reference"]["sha256"] != event["native_sha256"] or entry["source_sha256"] != event["source_sha256"]:
            raise ValueError("R48 source identity changed")
        for variant in event["clock_variants"]:
            grouped[(variant["song"], *variant["target_interval"])].append((event, variant))
    tracks, audio_paths, inputs = {}, {}, {}
    for song in range(3):
        name = f"dunhuangwu{song+2}.wav"
        audio = sealed(SONGS, "viewer_final/audio/"+name)
        music = sealed(Path("outputs/complete_r36_01"), f"planning/song{song}_music.json")
        audio_paths[song] = audio
        signal = audio_signals(audio)
        values, mask, _ = full_audio_observations(audio)
        tracks[song] = (values, mask, signal, read(music))
        inputs[song] = dict(audio_sha256=sha(audio), structure_sha256=sha(music))
    rows, previews = [], []
    # One fixed preview slot per music, closest to the temporal midpoint.
    demos = {song:min((k for k in grouped if k[0]==song),
                       key=lambda k:(abs(k[1]+k[2]-tracks[song][3]["frames"]), k[2]-k[1], k))
             for song in tracks}
    for key, bank in sorted(grouped.items()):
        song, a, b = key
        values, mask, signal, structure = tracks[song]
        if b > len(values) or b > len(signal["energy"]):
            rows.append(dict(song=song, interval=[a,b], reasons={"MUSIC_SUPPORT":len(bank)}, candidates=[]))
            continue
        tokens = net.music_tokens(values[a:b], mask[a:b])
        energy = np.array(signal["energy"][a:b])
        target_activity = np.mean(structure["energy_percentile"][a:b])
        candidates, reasons = [], defaultdict(int)
        identities = set()
        for event, variant in bank:
            uid = event["event_uid"]
            if uid in identities:
                raise ValueError("Unequal variant search budget for this event/slot")
            identities.add(uid)
            if variant["music_sha256"] != inputs[song]["structure_sha256"]:
                raise ValueError("Clock and music structure provenance differ")
            rotation = lib.rotations[event["source_uid"]]
            positions, penalty = source_clock(event, variant)
            m = main_scores(net, rotation, positions, tokens, energy, target_activity)
            if m is None:
                reasons["NO_COMMON_TOKEN_SUPPORT"] += 1
                continue
            row = dict(event_uid=uid, source_uid=event["source_uid"], source_interval=event["source_interval"],
                       time_warp=variant["clock"]["time_warp"], anchor_source_frames=variant["clock"]["anchor_source_frames"],
                       warp_penalty=penalty, truncated_phase_evidence=event["stage_evidence_truncated"],
                       **m, score=m["local"]+MULTI_WEIGHT*m["multi"]-WARP_WEIGHT*penalty)
            if controls:
                baseline, _ = source_clock(event, variant, linear=True)
                l = main_scores(net, rotation, baseline, tokens, energy, target_activity)
                donor = tracks[(song+1)%3]
                # Intersect real donor support; do not loop or pad tracks.
                if b <= len(donor[0]):
                    wrong = main_scores(net, rotation, positions,
                                        net.music_tokens(donor[0][a:b], donor[1][a:b]),
                                        np.array(donor[2]["energy"][a:b]), target_activity)
                else:
                    wrong = None
                row["controls"] = dict(B0_activity_proxy=m["heuristic"],
                                       B1_linear_local=l["local"] if l else None,
                                       B2_linear_local_multi=l["local"]+MULTI_WEIGHT*l["multi"] if l else None,
                                       B3_warped=row["score"],
                                       B4_coverage=row["score"] if m["valid_frames"]==m["total_tokens"] else None,
                                       swapped_local=wrong["local"] if wrong else None)
            candidates.append(row)
        candidates.sort(key=lambda r:(-r["score"], r["event_uid"]))
        rows.append(dict(song=song, interval=[a,b], reasons=dict(reasons), candidates=candidates))
        if key == demos[song] and candidates:
            options = sorted(candidates, key=lambda r:(-r["heuristic"],r["event_uid"]))
            previews.append((rows[-1], candidates[0], options[0]))
    out.mkdir(parents=True)
    write(out/"result.json", dict(baseline=BASELINE, model=net.binding(), inputs=inputs,
                                 event_index_sha256=sha(prior), rows=rows,
                                 registered_variants=sum(map(len, grouped.values())),
                                 scored_variants=sum(len(r["candidates"]) for r in rows),
                                 controls=controls, generated=False, can_route=False))
    if not controls:
        from .view import render
        render(out, previews, audio_paths, lib, net.joints, net.parents)
    print("R49_RANK",len(rows),sum(len(r["candidates"]) for r in rows),len(previews),flush=True)
