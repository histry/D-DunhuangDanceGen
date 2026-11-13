"""Verify paired clocks and freeze honest limits before any R7 GPU calls."""
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import ref
from dunhuang.alignment.audio_clock import same_music_time
from .common import read,commit,fresh,prior,frozen
def run(config,out,replay_from=None):
    out=fresh(out);root,r6,prep=prior(config);_,_,_,manifest=frozen(r6);index={r["window_uid"]:r for r in manifest["rows"]};pairs=read(checked(prep["pairs"]))
    audits=[];groups={};raw={};audio={}
    for cohort in config["cohorts"]:
        pop=pairs[cohort]
        if len(pop)!=4:raise ValueError("Frozen four-pair populations required")
        groups[cohort]=set();raw[cohort]=set();audio[cohort]=set()
        for case,pair in enumerate(pop):
            a,b=[index[pair[k]] for k in ("a","b")]
            if a["role"]!="train" or b["role"]!="train" or not same_music_time(a,b) or a["raw_motion"]["sha256"]==b["raw_motion"]["sha256"]:raise ValueError("Distinct real same-clock Train performances required")
            groups[cohort].add(pair["source_group"]);raw[cohort].update(r["raw_motion"]["sha256"] for r in (a,b));audio[cohort].add(a["audio"]["sha256"])
            audits.append(dict(cohort=cohort,case=case,same_verified_music_clock=True,role="train",provided_references=2,artistic_negative_label=False))
    if groups["calibration"]&groups["disjoint_probe"] or raw["calibration"]&raw["disjoint_probe"] or audio["calibration"]&audio["disjoint_probe"]:raise ValueError("R6 probe source separation changed")
    # Registered review evidence is still a preparation-only pilot, not human judgments.
    human=read("reports/review_viewer_v2_summary.json");hr=read("reports/review_viewer_v2_receipt.json")
    if sha("reports/review_viewer_v2_summary.json")!=hr["summary_sha256"] or hr["exit_status"]!=0:raise ValueError("Current viewer receipt differs")
    music=[]
    for name in ("dunhuangwu2.wav","dunhuangwu3.wav","dunhuangwu4.wav"):
        path=Path(config["historical_music_root"])/name
        if path.exists():
            info=sf.info(path);music.append(dict(file=ref(path),sample_rate=info.samplerate,samples=info.frames,channels=info.channels,
                historical_role="test",work_identity="USER_DESIGNATED_DUNHUANG",new_independent_input_qualified=False,exploratory_transfer_input_qualified=True))
    if len(music)!=3:raise ValueError("All three user-designated full songs required")
    write(out/"historical_music_inventory.json",music)
    result=dict(pair_clock_audits=audits,known_train_pairs=8,source_group_overlap=0,audio_sha_overlap=0,raw_sha_overlap=0,
        independent_evaluation_pairs=0,independent_source_qualification="NOT_EVALUABLE_NO_REGISTERED_UNEXPOSED_POPULATION",
        pretrained_exposure="UNKNOWN",checkpoint_finetuning_population="EXPOSED_TRAIN",paired_objective_deployable=False,
        registered_human_records=human["result"]["actual_human_records"],R7_B="PENDING_REAL_INDEPENDENT_REVIEW",
        historical_music_assets=len(music),R7_C="ADMITTED_EXPOSED_FULL_SONG_TRANSFER_INPUTS",
        waveform_tensor_reads=0,paired_test_feature_reads=0,authorized_exposed_transfer_assets=len(music),jukebox_extract_calls=0,planner_calls=0,model_training_calls=0,train_mechanism_admitted=True,operational_qualified=False)
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),config=config,source_prepared=ref(root/"prepared/protocol.json"),pairs=prep["pairs"],normalizer=prep["normalizer"],
        source_manifest=prep["source_manifest"],frozen_probe_protocol=ref(root/"probe/protocol.json"),frozen_probe_arrays=ref(root/"probe/arrays.npz"),viewer_receipt=ref("reports/review_viewer_v2_receipt.json"),
        inputs="two actually provided visible references; hidden motion evaluation only; no Val/Test objective probes",inventory=ref(out/"historical_music_inventory.json")))
    if replay_from:
        prev=Path(replay_from)
        if read(prev/"result.json")!=result or read(prev/"historical_music_inventory.json")!=music:raise ValueError("Preflight metadata replay differs")
        write(out/"replay.json",dict(status="FRESH_R7_PREFLIGHT_REPLAY",optimizer_calls=0,test_feature_reads=0))
    print("R7_PREFLIGHT_COMPLETED",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from");a=p.parse_args();run(read(a.config),a.out,a.replay_from)
