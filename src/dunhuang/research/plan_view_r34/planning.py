"""Fresh E1-W plan replay with frozen actual score provenance; no SDMR or generator training."""
import argparse,collections
import numpy as np
import soundfile as sf
from .common import *
from dunhuang.research.planner_integration.study import candidates,search
from dunhuang.research.song_transfer.music import shortlist
from dunhuang.planning.planner import _search
from dunhuang.planning.event_plan import semantic_plan_id,candidate_identity
from dunhuang.planning.feasible_time_map import solve

def run(root,out,replay=None):
    out=fresh(out);forbid_optimizers();old=Path("outputs/exposed_full_song_01")
    publication=read("reports/exposed_full_song_summary.json");receipt=read("reports/exposed_full_song_receipt.json")
    assert sha("reports/exposed_full_song_summary.json")==receipt["summary_sha256"]
    for name in ("music/protocol.json","music/arrays.npz","music/inventory.json"):assert sha(old/name)==publication["evidence"][name]
    protocol=read(old/"music/protocol.json");inventory=read(checked(protocol["inventory"]));scores=np.load(old/"music/arrays.npz")
    parent=read("outputs/data_event_v3_04/main/protocol.json");catalog=read(checked(parent["catalog"]));new=read("outputs/data_event_v3_04/main/event_catalog_v2_shadow.json")["entries"]
    shadow=read(old/"music/shadow_private.json");songs=[];all_banks={}
    for item in inventory:
        case=item["case"];audio=resolve_audio(item["audio"]);binding=next(b for b in protocol["bindings"] if b["case"]==case and b["timeline"]=="acoustic_v1" and b["arm"]=="E1_W")
        previous=read(checked(binding["plan"]));timing=read(old/f"music/{case}_acoustic_v1_timeline.json")
        assert timing["frames"]==item["clock"]["frames"] and timing["audio_sha256"]==item["audio"]["sha256"]
        banks,_=candidates(catalog["entries"],new,timing,"E1_W");chosen=search(banks,5,3)
        assert chosen is not None
        plan=_search(catalog,timing,item["audio"]["sha256"],top_k=5,beam_width=3,fixed_events={i:e for i,e in enumerate(chosen[1])})[0]
        assert semantic_plan_id(plan)==semantic_plan_id(previous)
        folder=out/f"song{case}";folder.mkdir();write(folder/"event_plan.json",plan)
        slots=[]
        for si,(slot,bank,selected) in enumerate(zip(timing["slots"],banks,plan["events"])):
            options=shortlist(bank);gs=scores[f"{case}__{si}__global"];ss=scores[f"{case}__{si}__sync"]
            assert len(options)==len(gs)==len(ss) and np.isfinite(gs).all() and np.isfinite(ss).all()
            original=next(r for r in shadow if r["case"]==case and r["slot"]==si)
            assert options[int(np.argmax(gs))]["event_uid"]==original["top_event"]
            cached={e["candidate_uid"]:dict(global_score=float(g),sync_score=float(s)) for e,g,s in zip(options,gs,ss)}
            display=[next(e for e in bank if e["candidate_uid"]==selected["candidate_uid"])]
            display.extend(e for e in options if e["candidate_uid"]!=selected["candidate_uid"])
            rejected=[];reasons=collections.Counter()
            for e in new:
                entry=dict(source_interval=e["source_interval"],center=e["safe_anchor"]["frame"] if e["safe_anchor"] else None,normalization_support=e["normalization_support"])
                result=solve(entry,slot,timing["beats"]);reasons[result["status"]]+=1
                if result["status"]!="PASS" and len(rejected)<8:rejected.append(dict(event_uid=e["event_uid"],source_uid=e["source_uid"],reason=result["status"]))
            views=[]
            for e in display[:3]:
                score=cached.get(e["candidate_uid"],dict(global_score=None,sync_score=None))
                certificate=solve(dict(source_interval=e["source_interval"],center=e["center"],normalization_support=True),slot,timing["beats"])
                assert certificate["status"]=="PASS" and certificate["time_warp"]==e["time_warp"] and candidate_identity(e)==e["candidate_uid"]
                views.append(dict(e,display_scores=dict(heuristic=float(e["scores"]["heuristic"]),**score),
                    score_scope="Heuristic/activity used by planner; frozen SDMR global and sync are unqualified Shadow, not style accuracy",
                    timing_certificate=certificate,link_id=identity(item["audio"]["sha256"],e)))
            slots.append(dict(slot=si,start=slot["start"],end=slot["end"],selected_candidate_uid=selected["candidate_uid"],
                selected_link_id=identity(item["audio"]["sha256"],selected),legal_candidates=len(bank),candidates=views,rejection_counts=dict(reasons),rejected_examples=rejected))
        all_banks[str(case)]=[[dict(candidate_uid=e["candidate_uid"],event_uid=e["event_uid"],source_uid=e["source_uid"],time_warp=e["time_warp"]) for e in b] for b in banks]
        wave,sr=sf.read(audio,dtype="float32",always_2d=True);mono=wave.mean(1);bins=np.linspace(0,len(mono),1201,dtype=int)
        waveform=[float(abs(mono[a:b]).max()) for a,b in zip(bins[:-1],bins[1:])]
        song=dict(id=f"song{case}",case=case,name=item["name"],frames=timing["frames"],fps=30,audio=ref(audio),scope="HISTORICALLY_EXPOSED_DEVELOPMENT_MUSIC",
            beats=timing["beats"],waveform=waveform,slots=slots,plan=ref(folder/"event_plan.json"),semantic_plan_id=semantic_plan_id(plan),
            phrase_scope=timing.get("phrase_method","ACOUSTIC_PROXY_NOT_REVIEWED_PHRASES"),catalog=parent["catalog"])
        songs.append(song);print("R34_PLAN_VISIBLE",case,len(slots),flush=True)
    write(out/"candidate_banks.json",all_banks);write(out/"songs.json",songs)
    result=dict(songs=3,full_song_frames=[s["frames"] for s in songs],event_slots=[len(s["slots"]) for s in songs],optimizer_steps=0,sampler_calls=0,
        actual_E1W_replayed=True,shadow_scores_reused=True,shadow_used_for_selection=False,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),r7_music_protocol=ref(old/"music/protocol.json"),r7_scores=ref(old/"music/arrays.npz"),
        scores="Exact frozen shortlist ordering and source/map/musical slot reproduced; no new fit",original_summary=ref("reports/exposed_full_song_summary.json")))
    if replay:
        for n in ("songs.json","candidate_banks.json","result.json"):
            left=json.dumps(read(out/n),sort_keys=True).replace(str(out.resolve()),"<RUN>")
            right=json.dumps(read(Path(replay)/n),sort_keys=True).replace(str(Path(replay).resolve()),"<RUN>")
            assert left==right
        for s in songs:assert read(out/Path(s["plan"]["path"]).parent.name/"event_plan.json")==read(Path(replay)/Path(s["plan"]["path"]).parent.name/"event_plan.json")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(a.root,a.out,a.replay_from)
