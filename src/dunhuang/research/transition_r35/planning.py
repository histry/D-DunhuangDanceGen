"""Independent Canonical-body support, retimed boundary graph and music-bounded beam."""
import argparse,copy,json,collections
import numpy as np
from scipy.spatial.transform import Rotation
from dunhuang.contracts import matrices
from dunhuang.generation.soft_reference import resample_native
from dunhuang.planning.time_map import source_positions
from dunhuang.planning.event_plan import candidate_identity,semantic_plan_id,validate
from dunhuang.research.planner_integration.study import candidates
from dunhuang.research.plan_view_r34.common import *
from .graph import descriptor,edges,zones,pool,search,path_cost

class Sources:
    def __init__(self):
        a=read("outputs/event_plan_01/prepare/assets.json")
        self.origins={r["source_uid"]:r for r in read(checked(a["catalog_source"]))["source_records"]}
        self.native={};self.audit={}
        a=np.load("outputs/coverage_r34_01/plan_generation/arrays.npz")
        self.J=a["skeleton_J22"];self.parents=a["parents"]
    def get(self,event):
        uid=event["source_uid"]
        if uid not in self.native:
            o=self.origins[uid];raw=np.load(checked(o["raw_receipt"]));native=np.load(checked(event["reference"]))
            if len(raw["poses"])!=len(native) or float(raw["mocap_framerate"])!=30:raise ValueError("Native/raw clock")
            maximum=0.
            for lo in range(0,len(native),2048):
                r=Rotation.from_rotvec(raw["poses"][lo:lo+2048,:66].reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
                n=matrices(native[lo:lo+2048,7:].reshape(-1,24,6))[:,:22]
                maximum=max(maximum,float(Rotation.from_matrix((r.swapaxes(-1,-2)@n).reshape(-1,3,3)).magnitude().max()))
            if maximum>1e-5:raise ValueError("Raw body parity failed")
            self.native[uid]=native
            self.audit[uid]=dict(raw=o["raw_receipt"],native=event["reference"],raw_native_max_angle_rad=maximum,
                source_observation="RAW_SMPL_FIRST22_VERIFIED",representation="R31_CANONICAL_BODY132_ROW6D_IDENTITY_BOUNDS",
                Root_translation="UNKNOWN",contact="UNKNOWN",physical_transition="UNKNOWN",native24="NOT_EVALUABLE_22_OF24")
        if self.audit[uid]["native"]["sha256"]!=event["reference"]["sha256"]:raise ValueError("Source alias changed bytes")
        return self.native[uid]
    def body(self,e,ticks):
        n=self.get(e);pos=source_positions(e,np.asarray(ticks))
        v=resample_native(n,pos);R=matrices(v[:,7:].reshape(-1,24,6))[:,:22]
        return R,relative_fk(R,self.J,self.parents)
    def describe(self,e):
        a,b=e["music_interval"]
        # Complete event differentiation, then read both neighborhoods.
        r,p=self.body(e,np.arange(a,b));return descriptor(r,p)

def edge_row(matrix,a,b):
    return {k:float(v[a,b]) for k,v in matrix.items()}

def run(cfg,out,replay=None):
    out=fresh(out);forbid_optimizers();parent=Path(cfg["r34"])
    assert sha(parent/"artifacts_manifest.json")==cfg["r34_manifest_sha256"]
    sources=Sources();p=read("outputs/data_event_v3_04/main/protocol.json")
    catalog=read(checked(p["catalog"]));new=copy.deepcopy(read("outputs/data_event_v3_04/main/event_catalog_v2_shadow.json")["entries"])
    # Do not reuse the EDGE upper8 scaler admission as Lodge body admission.
    for e in new:e["normalization_support"]=True
    songs=[];stats=[];arrays={};oldrows=read(parent/"planning/songs.json")
    for song in oldrows:
        oldplan=read(checked(song["plan"]));timing=read(f"outputs/exposed_full_song_01/music/{song['case']}_acoustic_v1_timeline.json")
        allbanks,_=candidates(catalog["entries"],new,timing,"E1_W")
        # Verify source observation before retaining any candidate. Unknown Root remains unknown.
        for bank in allbanks:
            for e in bank:sources.get(e)
        banks=[pool(b,e,cfg) for b,e in zip(allbanks,oldplan["events"])]
        desc=[[sources.describe(e) for e in b] for b in banks]
        mats=[edges(a,b,x,y) for a,b,x,y in zip(desc[:-1],desc[1:],banks[:-1],banks[1:])]
        oldindices=[next(j for j,e in enumerate(b) if e["candidate_uid"]==o["candidate_uid"]) for b,o in zip(banks,oldplan["events"])]
        cost,indices,regret=search(banks,mats,oldplan["events"],cfg)
        assert regret/len(banks)<=cfg["music_mean_regret_max"]+1e-9
        sequence=[copy.deepcopy(banks[i][j]) for i,j in enumerate(indices)]
        plan=copy.deepcopy(oldplan);plan["events"]=sequence
        plan["research_transition"]=dict(schema="body22_transition_graph_v1",music_mean_regret=regret/len(banks),world_Root="UNKNOWN",contact="UNKNOWN",physical_feasibility="UNKNOWN",
            old_semantic_plan_id=semantic_plan_id(oldplan),source_weights="FROZEN_R33_DYN; NO_CULTURAL_QUALIFICATION",transition_weight=cfg["transition_weight"])
        validate(plan,song["audio"]["sha256"],song["frames"])
        folder=out/song["id"];folder.mkdir();write(folder/"event_plan.json",plan)
        boundaries=[]
        for i,m in enumerate(mats):
            before=edge_row(m,oldindices[i],oldindices[i+1]);after=edge_row(m,indices[i],indices[i+1])
            a=sequence[i];b=sequence[i+1];za=zones(a);zb=zones(b)
            transition=dict(boundary_frame=a["music_interval"][1],edit_interval=[za["exit_edit"][0],zb["entry_edit"][1]],
                protected_left=za["protected_core"],protected_right=zb["protected_core"],left_event_uid=a["event_uid"],right_event_uid=b["event_uid"],Root="UNKNOWN",contact="UNKNOWN")
            boundaries.append(dict(index=i,old=before,new=after,contract=transition,
                alternative_right=[dict(event_uid=e["event_uid"],source_uid=e["source_uid"],candidate_uid=e["candidate_uid"],music_score=e["scores"]["heuristic"],
                    transition_proxy=edge_row(m,indices[i],j)) for j,e in enumerate(banks[i+1])]))
            for k,v in m.items():arrays[f"{song['id']}_{i}_{k}"]=v
        candidates_view=[]
        for i,(bank,j) in enumerate(zip(banks,indices)):
            selected=sequence[i];rank=sorted(range(len(bank)),key=lambda k:(-bank[k]["scores"]["heuristic"],bank[k]["event_uid"]))
            chosen=[j]+[k for k in rank if k!=j][:2]
            candidates_view.append(dict(slot=i,start=selected["music_interval"][0],end=selected["music_interval"][1],legal_candidates=len(allbanks[i]),retained_candidates=len(bank),
                selected_candidate_uid=selected["candidate_uid"],candidates=[dict(bank[k],display_scores=dict(heuristic=bank[k]["scores"]["heuristic"],global_score=None,sync_score=None),
                score_scope="Actual activity matching; new transition graph; SDMR not used",timing_certificate=dict(status="E1W_TIME_PASS"),link_id=identity(song["audio"]["sha256"],bank[k])) for k in chosen],
                selected_link_id=identity(song["audio"]["sha256"],selected),rejection_counts={},rejected_examples=[]))
        row=dict(song,slots=candidates_view,plan=ref(folder/"event_plan.json"),old_plan=song["plan"],semantic_plan_id=semantic_plan_id(plan),boundaries=boundaries)
        songs.append(row)
        stats.append(dict(song=song["id"],slots=len(banks),boundaries=len(boundaries),retained_per_slot=[len(b) for b in banks],
            same_source_multiple_candidates=sum(any(n>1 for n in collections.Counter(e["source_uid"] for e in b).values()) for b in banks),
            original_music_cost=float(sum(-e["scores"]["heuristic"] for e in oldplan["events"])),new_music_cost=float(sum(-e["scores"]["heuristic"] for e in sequence)),music_mean_regret=regret/len(banks),
            old_graph_cost=path_cost(oldindices,banks,mats,cfg),new_graph_cost=cost,changed_slots=sum(x["candidate_uid"]!=y["candidate_uid"] for x,y in zip(sequence,oldplan["events"])),
            old_mean_edge=float(np.mean([b["old"]["cost"] for b in boundaries])),new_mean_edge=float(np.mean([b["new"]["cost"] for b in boundaries])),
            proxy_selection_metric_not_independent_quality=True))
        print("R35_PLAN",stats[-1],flush=True)
    np.savez_compressed(out/"edges.npz",**arrays)
    write(out/"songs.json",songs);write(out/"source_audit.json",sources.audit)
    result=dict(code_commit=commit(),songs=stats,source_count=len(sources.audit),boundary_count=sum(s["boundaries"] for s in stats),optimizer_steps=0,sampler_calls=0,
        Root_contact_unknown=True,operational_qualified=False,whole_song_generations=0)
    write(out/"result.json",result)
    if replay:
        same_arrays(out/"edges.npz",Path(replay)/"edges.npz")
        for n in ("songs.json","source_audit.json","result.json"):
            x=json.dumps(read(out/n),sort_keys=True).replace(str(out.resolve()),"<RUN>")
            y=json.dumps(read(Path(replay)/n),sort_keys=True).replace(str(Path(replay).resolve()),"<RUN>")
            assert x==y
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read("src/dunhuang/research/transition_r35/config.json"),a.out,a.replay_from)
