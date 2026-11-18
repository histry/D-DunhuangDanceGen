"""Joint source/time/bridge search on a fixed, bounded population."""
from collections import Counter
from itertools import product
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import multi_warp, relative_fk, rotation_sample, write, read
from dunhuang.research.bridge_r38.geometry import observed_bridge
from dunhuang.research.bridge_audit_r46.admission import certificate
from dunhuang.research.manifold_bridge_r47.geometry import segmented
from dunhuang.research.bridge_dynamics_r48.study import metrics
from dunhuang.research.event_alignment_r49.rank import main_scores
from .common import ROOT, assets, layout


def bridge(left, right, frames, net, reference):
    R = np.full((frames+6,22,3,3),np.nan)
    R[:3],R[-3:] = left[-3:],right[:3]
    gap = (3,3+frames)
    options = {}
    cert = certificate(R,gap)
    if cert["chart_eligible"]:
        options["G0"] = observed_bridge(R,gap)
    candidate, guard = segmented(R,gap,(4,8,16),np.pi-.05)
    if candidate is not None:
        options["G1"] = candidate
    records = {}
    for name,Q in options.items():
        # Four observed samples support the outer finite-difference stencil.
        observed = np.concatenate((left[-4:-3],Q,right[3:4]))
        m = metrics(observed,net.joints,net.parents,(4,4+frames))
        records[name] = m
    eligible = [n for n,m in records.items()
                if m["gap_D2"]<=reference["D2"] and m["gap_D3"]<=reference["D3"]
                and m["shoulder_wrist_min_m"]>=reference["shoulder_wrist_lower_m"]]
    name = min(eligible,key=lambda n:(records[n]["gap_D3"],n)) if eligible else None
    evidence = dict(selected=name,methods=records,G0=cert,G1=guard,frames=frames,
                    physical_qualified=False)
    return (options[name][3:-3] if name else None),evidence


def run(out):
    out = Path(out)
    if out.exists():
        raise ValueError("Fresh R50 output required")
    net,lib,units,songs = assets()
    reference = read("outputs/bridge_dynamics_r48_01/main/result.json")["reference"]
    # Equal source budget chosen from time metadata, before any target score.
    bank = []
    for uid in sorted(lib.rows):
        options = sorted((u for u in units if u["source_uid"]==uid and 135<=u["duration_frames"]<=399),
                         key=lambda u:u["source_interval"])
        bank.extend(options[k] for k in sorted(set((len(options)//3,2*len(options)//3))))
    if len(bank)!=16:
        raise ValueError("Registered two-per-source population unavailable")
    cases,positions = [], {}
    for song,track in enumerate(songs):
        for count in (2,3):
            for transition in (24,48,72):
                duration = 240
                length = count*duration+(count-1)*transition
                start = (min(len(track["values"]),track["structure"]["frames"])-length)//2
                regions = layout(start,[duration]*count,[transition]*(count-1))
                cores = [r["interval"] for r in regions if r["kind"]=="core"]
                candidates = []
                for a,b in cores:
                    pool = []
                    music = net.music_tokens(track["values"][a:b],track["mask"][a:b])
                    energy = np.array(track["signal"]["energy"][a:b])
                    for unit in bank:
                        phases = [dict(p,type=p["body_region"]+"_"+p["type"]) for p in unit["phases"]]
                        mapped = multi_warp(unit["source_interval"],[a,b],phases,track["structure"]["beats"])
                        if mapped is None:
                            continue
                        w = np.array(mapped["time_warp"])
                        q = np.interp(np.arange(a,b),w[:,1],w[:,0])
                        score = main_scores(net,lib.rotations[unit["source_uid"]],q,music,energy,
                                            float(np.mean(track["structure"]["energy_percentile"][a:b])))
                        if score is None:
                            continue
                        R = rotation_sample(lib.rotations[unit["source_uid"]],q)
                        rates = np.diff(w[:,1])/np.diff(w[:,0])
                        total = score["local"]+.1*score["multi"]-.05*float(np.mean(abs(np.log(rates))))
                        pool.append(dict(unit=unit,warp=mapped,R=R,score=total))
                    # Four distinct sources retained for bounded pairwise search.
                    chosen, seen = [],set()
                    for c in sorted(pool,key=lambda c:(-c["score"],c["unit"]["unit_uid"])):
                        uid = c["unit"]["source_uid"]
                        if uid not in seen:
                            chosen.append(c)
                            seen.add(uid)
                        if len(chosen)==4:
                            break
                    candidates.append(chosen)
                edges = {}
                for k in range(count-1):
                    for i,l in enumerate(candidates[k]):
                        for j,r in enumerate(candidates[k+1]):
                            if l["unit"]["source_uid"]==r["unit"]["source_uid"]:
                                continue
                            curve,evidence = bridge(l["R"],r["R"],transition,net,reference)
                            edges[k,i,j] = (curve,evidence)
                solutions = []
                for ids in product(*(range(len(c)) for c in candidates)):
                    if any((k,ids[k],ids[k+1]) not in edges or edges[k,ids[k],ids[k+1]][0] is None
                           for k in range(count-1)):
                        continue
                    cost = sum(candidates[k][i]["score"] for k,i in enumerate(ids))
                    cost -= .01*sum(edges[k,ids[k],ids[k+1]][1]["methods"][
                        edges[k,ids[k],ids[k+1]][1]["selected"]]["gap_D3"]/max(reference["D3"],1e-9)
                        for k in range(count-1))
                    solutions.append((cost,ids))
                key = f"song{song}_events{count}_gap{transition}"
                record = dict(key=key,song=song,events=count,transition_frames=transition,
                              music_interval=[start,start+length],regions=regions,
                              core_frames=count*duration,bridge_frames=(count-1)*transition,
                              total_frames=length,candidate_counts=list(map(len,candidates)),
                              edge_trials=len(edges),eligible_edges=sum(v[0] is not None for v in edges.values()),
                              failures=dict(Counter("GEOMETRY_OR_DYNAMIC_REJECT" for v in edges.values() if v[0] is None)),
                              can_route=False,root_contact="UNKNOWN",semantic_truth="UNKNOWN")
                if solutions:
                    score,ids = max(solutions,key=lambda c:(c[0],tuple(-i for i in c[1])))
                    chunks,selected,connections = [],[],[]
                    for k,i in enumerate(ids):
                        c = candidates[k][i]
                        chunks.append(c["R"])
                        selected.append(dict(event_uid=c["unit"]["unit_uid"],source_uid=c["unit"]["source_uid"],
                                             source_interval=c["unit"]["source_interval"],warp=c["warp"],
                                             score=c["score"],phase_evidence_truncated=c["unit"]["stage_evidence_truncated"]))
                        if k<count-1:
                            q,e = edges[k,i,ids[k+1]]
                            chunks.append(q)
                            connections.append(e)
                    R = np.concatenate(chunks)
                    assert len(R)==length
                    positions[key] = relative_fk(R,net.joints,net.parents).astype("<f4")
                    positions[key+"_R"] = R
                    record.update(status="NUMERICAL_SHORT_PLAN",selected=selected,
                                  connections=connections,score=score,
                                  whole_clip_D3=float(np.quantile(np.sqrt(np.mean(
                                      (np.diff(positions[key],n=3,axis=0)*27000)**2,axis=(1,2))),.95)))
                else:
                    record["status"]="REJECTED"
                cases.append(record)
                print("R50_PLAN",key,record["status"],flush=True)
    out.mkdir(parents=True)
    np.savez_compressed(out/"arrays.npz",**positions,J=net.joints,parents=net.parents)
    write(out/"result.json",dict(cases=cases,model=net.binding(),source_bank=[u["unit_uid"] for u in bank],
          reference=reference,search_budget=dict(core_frames=240,top_sources=4,events=[2,3],transitions=[24,48,72]),
          generated=False,can_route=False))
    from .view import render
    render(out,songs)
