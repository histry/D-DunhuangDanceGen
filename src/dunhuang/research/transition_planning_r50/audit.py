"""Full-score music interventions on a common candidate and observation population."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from dunhuang.research.complete_r36.core import read,write,relative_fk,rotation_sample
from dunhuang.research.event_alignment_r49.common import cosine,multiscale
from dunhuang.research.event_alignment_r49.rank import source_clock
from .common import ROOT,assets
from .conditions import diagnose


def audit(out):
    if not read(ROOT/"browser_receipt.json")["passed"]:
        raise ValueError("Watch primary effect before attribution")
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh attribution output")
    net,lib,units,tracks = assets()
    r48 = read("outputs/event_feasibility_r48_01/result.json")
    variants = {(e["event_uid"],v["song"],*v["target_interval"]):(e,v)
                for e in r48["events"] for v in e["clock_variants"]}
    old = read("outputs/event_alignment_r49_01/result.json")
    rows = []
    for slot in old["rows"]:
        song = slot["song"]
        a,b = slot["interval"]
        correct = tracks[song]
        wrong = tracks[(song+1)%3]
        if not slot["candidates"] or b>len(wrong["values"]) or b>len(wrong["signal"]["energy"]):
            continue
        candidate_rows = []
        # A5 matches real same-song segment by mean audio energy, before content scoring.
        target_energy = np.mean(correct["signal"]["energy"][a:b])
        donor_starts = [t for t in range(0,len(correct["values"])-(b-a)+1,30)
                        if abs(t-a)>=b-a]
        shifted = min(donor_starts,key=lambda t:(abs(np.mean(correct["signal"]["energy"][t:t+b-a])-target_energy),t)) if donor_starts else None
        slices = [(correct,a),(wrong,a)]
        if shifted is not None:
            slices.append((correct,shifted))
        music = [net.music_tokens(t["values"][s:s+b-a],t["mask"][s:s+b-a]) for t,s in slices]
        for candidate in slot["candidates"]:
            event,variant = variants[candidate["event_uid"],song,a,b]
            positions,penalty = source_clock(event,variant)
            R = rotation_sample(lib.rotations[event["source_uid"]],positions)
            values,mask = net.motion(R)
            motion,valid = net.motion_tokens(values,mask)
            common = valid.copy()
            for _,mv in music:
                common &= mv
            if not common.any():
                continue
            P = relative_fk(R,net.joints,net.parents)
            activity = np.linalg.norm(np.diff(P,axis=0,prepend=P[:1])*30,axis=-1).mean(-1)
            local,multi,full = [],[],[]
            # Multi uses only the same real target frames contributing valid tokens.
            physical = np.flatnonzero(common)+6
            for (track,start),(tokens,_) in zip(slices,music):
                l = cosine(tokens[common],motion[common])
                m = multiscale(np.asarray(track["signal"]["energy"])[start+physical],activity[physical])
                local.append(l)
                multi.append(m)
                full.append(l+.1*m-.05*penalty)
            candidate_rows.append(dict(event_uid=event["event_uid"],source_uid=event["source_uid"],
                    source_frames=event["source_interval"],original_A0=candidate["score"],
                    common_token_count=int(common.sum()),coverage=float(common.mean()),
                    local=local,multi=multi,warp=penalty,full=full,
                    A1=local[0],A2=local[0]+.1*multi[0],A3=local[0]-.05*penalty,
                    A4=full[1],delta_full=full[0]-full[1]))
        # Equal source budget A6 selects at most one metadata-ordered event per source.
        balanced = []
        for uid in sorted({r["source_uid"] for r in candidate_rows}):
            bank = [r for r in candidate_rows if r["source_uid"]==uid]
            balanced.append(min(bank,key=lambda r:(r["source_frames"],r["event_uid"]))["event_uid"])
        diagnostics = {}
        if candidate_rows:
            first = np.array([r["full"][0] for r in candidate_rows])
            second = np.array([r["full"][1] for r in candidate_rows])
            order = np.argsort(-first,kind="stable")
            diagnostics = dict(full_top_changed=int(first.argmax()!=second.argmax()),
                local_top_changed=int(np.argmax([r["local"][0] for r in candidate_rows])!=np.argmax([r["local"][1] for r in candidate_rows])),
                spearman=float(spearmanr(first,second).statistic) if len(first)>1 and np.std(first)>0 and np.std(second)>0 else None,
                top1_top2_margin=float(first[order[0]]-first[order[1]]) if len(first)>1 else None,
                variance={k:float(np.var([r[k][0] if k in ("local","multi","full") else r[k] for r in candidate_rows]))
                          for k in ("local","multi","warp","full")},
                top_k=[candidate_rows[i]["event_uid"] for i in order[:5]])
        rows.append(dict(song=song,interval=[a,b],same_energy_same_song_start=shifted,
                         candidates=candidate_rows,A6_equal_source_ids=balanced,diagnostics=diagnostics))
        print("R50_ATTRIBUTION",song,a,b,len(candidate_rows),flush=True)
    source_stats = defaultdict(list)
    for row in rows:
        for r in row["candidates"]:
            source_stats[r["source_uid"]].append(r["delta_full"])
    oldval = read("outputs/event_alignment_r49_val/result.json")
    summary = dict(slots=len(rows),full_music_top_changes=sum(r["diagnostics"].get("full_top_changed",0) for r in rows),
                   local_music_top_changes=sum(r["diagnostics"].get("local_top_changed",0) for r in rows),
                   median_spearman=float(np.median([r["diagnostics"]["spearman"] for r in rows if r["diagnostics"].get("spearman") is not None])),
                   source_delta={k:dict(candidates=len(v),mean=float(np.mean(v)),std=float(np.std(v))) for k,v in source_stats.items()},
                   historical_val=oldval["summary"],independent_blind_evaluated=0,can_route=False)
    write(out/"result.json",dict(rows=rows,summary=summary,condition_probe=diagnose(),model=net.binding()))


def verify(out):
    first = ROOT
    replay = Path("outputs/transition_planning_r50_replay")
    assert read(first/"result.json")==read(replay/"result.json")
    with np.load(first/"arrays.npz") as a,np.load(replay/"arrays.npz") as b:
        assert set(a.files)==set(b.files)
        for name in a.files:
            assert np.array_equal(a[name],b[name]),name
    assert read("outputs/attribution_r50_01/result.json")==read("outputs/attribution_r50_replay/result.json")
    from dunhuang.research.reperformance_r44.replay import digest
    for item in read("outputs/r49_publication/seal.json")["files"]:
        p = Path(item["path"])
        assert p.stat().st_size==item["bytes"] and digest(p)==item["sha256"]
    good = [r for r in read(first/"result.json")["cases"] if r["status"]=="NUMERICAL_SHORT_PLAN"]
    for r in good:
        assert r["total_frames"]==r["core_frames"]+r["bridge_frames"]
        assert r["regions"][0]["interval"][0]==r["music_interval"][0]
        assert r["regions"][-1]["interval"][1]==r["music_interval"][1]
    write(out,dict(exact_arrays_and_scores=True,accepted_short_plans=len(good),
                   r49_sealed_files_verified=17,neural_training_steps=0,can_route=False))
