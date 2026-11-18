"""Independent diagnostics for phase, geometric connection and model condition eligibility."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from dunhuang.io import checked
from dunhuang.research.bridge_audit_r46.admission import certificate
from dunhuang.research.bridge_dynamics_r48.geometry import optimize
from dunhuang.research.bridge_dynamics_r48.study import metrics
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.complete_r36.core import Library, read, rotation_sample, sha, write
from dunhuang.research.manifold_bridge_r47.geometry import segmented
from .common import R41, R48, sealed
from .rank import CATALOG, PREVIEW_ROOT


def select_pairing(a, b, ra, rb):
    """Retain unique mutual neighbors with local order and real context support."""
    cost = Rotation.from_matrix((ra[:,None].swapaxes(-1,-2)@rb[None]).reshape(-1,3,3))
    cost = cost.magnitude().reshape(len(a),len(b),22).mean(-1)
    top = np.argsort(cost, axis=1, kind="stable")[:,:3]
    reverse = np.argsort(cost, axis=0, kind="stable")[:3]
    candidates = []
    for i, neighbors in enumerate(top):
        mutual = [int(j) for j in neighbors if i in reverse[:,j]]
        if len(mutual)!=1:
            continue
        j = mutual[0]
        if int(neighbors[0])!=j or int(reverse[0,j])!=i:
            continue
        if len(neighbors)<2 or cost[i,neighbors[1]]-cost[i,j]<=.02:
            continue
        candidates.append((i,j,float(cost[i,j])))
    return [(a[i],b[j],v) for k,(i,j,v) in enumerate(candidates)
            if k>0 and k+1<len(candidates) and candidates[k-1][1]<j<candidates[k+1][1]]


def phase():
    parent = sealed(Path("outputs/music_phase_r47_01"),"main/result.json")
    source_rows = read(sealed(R41,"data/manifest.json"))["sources"]
    sources = {r["source_uid"]:r for r in source_rows}
    selected = {}
    for p in read(parent)["phase_pairs"]:
        selected.setdefault(p["audio_sha256"],p)
    matches, denominator, timings, records = 0, 0, [], []
    for key,pair in sorted(selected.items()):
        a,b = (sources[k] for k in pair["source_pair"])
        assert a["role"]==b["role"]=="train"
        banks, rotations = [], []
        for row in (a,b):
            bank = defaultdict(list)
            for p in row["phases"]:
                lo,hi = np.asarray(p["evidence_span"])+row["audio_offset_frames"]
                if any(x<=lo and hi<=y for x,y in pair["actual_audio_support"]):
                    bank[p["type"],p["body_region"]].append(p["frame"])
            banks.append(bank)
            with np.load(checked(row["source_arrays"]),allow_pickle=False) as arrays:
                rotations.append(arrays["R"])
        chosen = []
        for kind,left in sorted(banks[0].items()):
            right = banks[1].get(kind,[])
            denominator += len(left)
            if not right:
                continue
            chosen.extend((kind,i,j,v) for i,j,v in select_pairing(left,right,rotations[0][left],rotations[1][right]))
        dt = [abs(i+a["audio_offset_frames"]-j-b["audio_offset_frames"]) for _,i,j,_ in chosen]
        matches += len(chosen)
        timings.extend(dt)
        records.append(dict(music_sha256=key, sources=[a["source_uid"],b["source_uid"]],
                            pose_only_matches=chosen, clock_errors_frames=dt))
    return dict(pairs=len(records), source_phases=denominator, accepted=matches,
                coverage=matches/max(denominator,1), within_15frames=sum(t<=15 for t in timings),
                stage_truth="UNKNOWN", source_sha256=sha(parent), records=records)


def geometry(ranked):
    G = Path("outputs/bridge_dynamics_r48_01")
    reference = read(sealed(G,"main/result.json"))["reference"]
    cfg = read("src/dunhuang/research/event_feasibility_r48/config.json")
    catalog = Library(read(CATALOG)["catalog"])
    skel = np.load(sealed(R41,"data/skeleton.npz"),allow_pickle=False)
    J,parents = skel["J"],skel["parents"]
    tasks = []
    a,b = cfg["gap"]
    for song in range(3):
        slots = [r for r in ranked["rows"] if r["song"]==song and r["candidates"]]
        pairs = [(l,r) for l in slots for r in slots if r["interval"][0]>=l["interval"][1]]
        if not pairs:
            tasks.append(dict(song=song,status="NO_REGISTERED_SLOT_PAIR_HAS_FIXED_48_FRAME_BRIDGE_ROOM"))
            continue
        left, right = min(pairs,key=lambda p:(abs(p[1]["interval"][0]-p[0]["interval"][1]-(b-a)),
                                             abs(p[0]["interval"][1]-1800),p[0]["interval"],p[1]["interval"]))
        gap_room = right["interval"][0]-left["interval"][1]
        for r in right["candidates"][:4]:
            l = left["candidates"][0]
            R = np.full((b+3,22,3,3),np.nan)
            for row,slot,index in ((l,left,slice(a-3,a)),(r,right,slice(b,b+3))):
                warp = np.asarray(row["time_warp"])
                lo,hi = slot["interval"]
                q = np.arange(hi-3,hi) if row is l else np.arange(lo,lo+3)
                R[index] = rotation_sample(catalog.rotations[row["source_uid"]],np.interp(q,warp[:,1],warp[:,0]))
            options = {}
            old = certificate(R,(a,b))
            if old["chart_eligible"]:
                options["G0"] = observed_bridge(R,(a,b))
            g1, _ = segmented(R,(a,b),(4,),cfg["chart_radius_rad"])
            if g1 is not None:
                options["G1"] = g1
            g3, cert = optimize(R,J,parents,reference,cfg)
            if g3 is not None:
                options["G3"] = g3
            # The R48 metric needs one more observed frame for its boundary stencil.
            tail = np.arange(right["interval"][0],right["interval"][0]+4)
            rw = np.asarray(r["time_warp"])
            P = rotation_sample(catalog.rotations[r["source_uid"]],np.interp(tail,rw[:,1],rw[:,0]))
            scores = {name:metrics(np.concatenate((Q[a-3:],P[3:])),J,parents,(3,b-a+3)) for name,Q in options.items()}
            valid = [n for n,m in scores.items() if m["gap_D2"]<=reference["D2"]
                     and m["gap_D3"]<=reference["D3"] and m["shoulder_wrist_min_m"]>=reference["shoulder_wrist_lower_m"]]
            choice = min(valid,key=lambda n:(scores[n]["gap_D3"],n)) if valid else None
            tasks.append(dict(song=song,left=l["event_uid"],right=r["event_uid"],left_slot=left["interval"],
                              right_slot=right["interval"],choice=choice, metrics=scores,G3=cert,
                              status="BODY_RELATIVE_NUMERICAL_PROPOSAL" if choice else "REJECTED",
                              target_gap_frames=gap_room, fixed_bridge_fits_target=gap_room==b-a,
                              root_contact="UNKNOWN"))
    return dict(tasks=tasks, can_route=False)


def run(out):
    ranked = read(PREVIEW_ROOT/"result.json")
    r43 = read("reports/r43_release/summary.json")
    write(out,dict(phase=phase(), geometry=geometry(ranked), P4=dict(
        condition_schema=["music","source_rotations","inverse_warp","phase","context_left","context_right","validity"],
        required_interventions=["right_music","wrong_music","zero_music","wrong_source","zero_phase",
                                "remove_left","remove_right"],
        same_noise_and_target_required=True,
        source_condition_evidence={"legacy_release_sha256":sha("reports/r43_release/summary.json"),
                                   "release_fields":sorted(r43)},
        stage_supervision="UNKNOWN", training_admitted=False, native24_status="NOT_EVALUABLE_COMMON22"),
        shadow=True))
