"""Additional read-only lineage, inclusion and phase audit, after the frozen study."""
from pathlib import Path
import sys,argparse,json,subprocess,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from dunhuang.io import write,sha,checked
from dunhuang.alignment.clock_data import read
from dunhuang.alignment.event_study import donor_indices
from dunhuang.alignment.event_metrics import proxy_retrieval
from dunhuang.alignment.clock_metrics import grouped

def run(out,source,dest):
    out=Path(out);source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    replayed=read(checked(read(out/"replay_verified.json"))) if (out/"replay_verified.json").exists() else read(out/"replay/result.json")
    if read(out/"result.json")["status"]!="COMPLETED_EVENT_IDENTIFIABILITY_SHADOW" or replayed["status"]!="FRESH_EVENT_IDENTIFIABILITY_REPLAY_VERIFIED":
        raise ValueError("Frozen training and fresh replay must complete before additional audit")
    protocol=read(out/"protocol.json");config=protocol["config"];data=read(out/"data/manifest.json");rows=data["rows"]
    bank=np.load(out/"data/events.npz");events=read(out/"data/selected_events.json");a=np.load(out/"data/operator.npz")
    population={};composition={}
    for role in ("train","val"):
        ids=[i for i,r in enumerate(rows) if r["role"]==role]
        keep=[i for i in ids if bank["eligible"][i]]
        population[role]=dict(source_rows=len(ids),eligible_rows=len(keep),excluded_rows=len(ids)-len(keep),
            source_components=len({rows[i]["source_group"] for i in ids}),eligible_components=len({rows[i]["source_group"] for i in keep}),
            weak_supported_rows=int(sum(bank["weak_weight"][i].sum()>0 for i in keep)),
            weak_supported_events=int(sum((bank["weak_weight"][i]>0).sum() for i in keep)),
            reason="nonempty detected events; own/other-capture source-clock positives, two non-overlapping same-song candidates and energy-matched unrelated controls")
        composition[role]={side:dict(collections.Counter(e["event_type"] for i in ids for e in events[i][side])) for side in ("music","motion")}
    phase={}
    for channel,name in ((66,"mean_angular_speed"),(67,"p90_angular_speed"),(72,"upper_squared_speed"),(73,"lower_squared_speed")):
        old=a["old"][:,15:135,channel];current=a["current"][:,15:135,channel]
        differences=abs(old.argmax(-1)-current.argmax(-1))/30
        phase[name]=dict(scope="argmax of a numerical motion channel on same 120-frame support; not labelled semantic events",
            mean_seconds=float(differences.mean()),p95_seconds=float(np.percentile(differences,95)),max_seconds=float(differences.max()),
            equal_peak_fraction=float((differences==0).mean()))
    ids=[i for i,r in enumerate(rows) if r["role"]=="val" and bank["eligible"][i]]
    rng=np.random.default_rng(config["seed"]+9001);chosen=rng.choice(ids,config["synthetic_sequences"],replace=len(ids)<config["synthetic_sequences"])
    donors=donor_indices(rows);recipe=[]
    for k,i in enumerate(chosen):
        i=int(i);j=donors[i]
        recipe.append(dict(sequence=k,source_row=i,donor_row=j,source_window=rows[i]["window_uid"],donor_window=rows[j]["window_uid"],
            source_role=rows[i]["role"],donor_role=rows[j]["role"],source_audio_sha256=rows[i]["audio"]["sha256"],
            source_motion_sha256=rows[i]["raw_motion"]["sha256"],donor_audio_sha256=rows[j]["audio"]["sha256"],
            donor_motion_sha256=rows[j]["raw_motion"]["sha256"],branch="music" if k%2 else "motion",seed=config["seed"]+k+70001))
    write(dest/"synthetic_recipe.json",recipe)
    retrieval={}
    positive=bank["positive"][ids];valid=bank["valid"][ids]
    same=np.asarray([[rows[j]["music_uid"]==rows[i]["music_uid"] for j in bank["candidates"][i]] for i in ids])
    def restricted(values):
        result={}
        for name,mask in (("same_song_wrong_time",valid&same),("other_song_energy_matched",valid&(~same|positive))):
            metrics=proxy_retrieval(values,positive,mask);win=[]
            for scores,p,v in zip(values,positive,mask):
                best=scores[p].max();negative=scores[v&~p]
                win.append(float(np.mean((best>negative+1e-7)+.5*(np.abs(best-negative)<=1e-7))))
            metrics["pair_win_grouped"]=grouped(win,[rows[i] for i in ids]);result[name]=metrics
        return result
    for mode in config["arms"]:
        predictions=np.load(out/mode/"measurements.npz");frozen=np.load(out/mode/"frozen_measurements.npz")
        retrieval[mode]=dict(final={k:restricted(predictions[k]) for k in ("scores","zero","shuffle","swap")},frozen=restricted(frozen["scores"]))
    baseline=np.load(out/"baseline_scores.npz")
    for k in baseline.files:retrieval[k]=restricted(baseline[k])
    histories={m:read(out/m/"history.json")[-1]["step"] for m in config["arms"]}
    diagnostics={}
    for m in config["arms"]:
        records=read(out/m/"path_diagnostics.json")
        if not records:continue
        diagnostics[m]={}
        for key in ("hard_path_crossings","crossing_mass","expected_position_violations","music_null_fraction","motion_null_fraction","coverage"):
            values=[r[key] for r in records if key in r]
            if values:diagnostics[m][key]=dict(mean=float(np.mean(values)),max=float(np.max(values)),sum=float(np.sum(values)))
    public=dict(schema="event_identifiability_additional_audit_v1",execution_commit=protocol["code_commit"],
        analysis_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),population=population,event_composition=composition,
        operator_peak_phase=phase,restricted_source_clock_retrieval=retrieval,path_diagnostics=diagnostics,completed_steps=histories,total_head_optimizer_updates=sum(histories.values()),
        total_probe_optimizer_updates=4*config["probe_steps"],synthetic_source_roles=sorted({r["source_role"] for r in recipe}),
        synthetic_donor_roles=sorted({r["donor_role"] for r in recipe}),synthetic_recipe_sha256=sha(dest/"synthetic_recipe.json"),
        data_manifest_sha256=sha(out/"data/manifest.json"),synthetic_truth_sha256=sha(out/"synthetic_truth.npz"),
        independent_human_labels=0,operational_qualified=False,new_optimizer_calls=0)
    write(dest/"public_audit.json",public)
    print(json.dumps(public,ensure_ascii=True))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("out","source","dest"):p.add_argument("--"+k,required=True)
    q=p.parse_args();run(q.out,q.source,q.dest)
