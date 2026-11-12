"""Public delivery receipt: status, aggregate hashes, versions; no source identities."""
import argparse,json,subprocess,sys,re,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import write,sha,checked
from dunhuang.alignment.clock_data import read

def run(out,receipt,dest):
    out=Path(out);receipt=Path(receipt)
    protocol=read(out/"protocol.json");replay=read(checked(read(out/"replay_verified.json")))
    if (receipt/"exit_status.txt").read_text().strip()!="0" or replay["status"]!="FRESH_EVENT_IDENTIFIABILITY_REPLAY_VERIFIED":
        raise ValueError("Successful fresh replay and launcher exit required")
    study=read(out/"result.json")
    if study["status"]!="COMPLETED_EVENT_IDENTIFIABILITY_SHADOW":raise ValueError("Training incomplete")
    log=(receipt/"pytest.log").read_text()
    matches=re.findall(r"(\d+ passed, \d+ skipped)",log)
    if not matches:raise ValueError("Test receipt missing")
    frozen=["src/dunhuang/generation","src/dunhuang/planning","src/dunhuang/evaluation_benchmarks","src/dunhuang/physical.py"]
    changes=subprocess.check_output(["git","diff","--name-only","bc8f241","HEAD","--"]+frozen,text=True)
    if changes.strip():raise ValueError("Frozen generation/Native implementation changed")
    import numpy,scipy,torch
    record=dict(schema="public_event_delivery_receipt_v1",status="COMPLETED_AND_FRESH_REPLAY_VERIFIED",
        training_commit=protocol["code_commit"],replay_commit=replay["replay_commit"],analysis_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        launcher_exit_status=0,pytest=matches[-1],optimizer_calls_during_replay=replay["optimizer_calls"],
        source_observations_rebuilt=replay["source_observations_rebuilt"],scientific_array_fields_verified=replay["array_fields_verified"],
        steps_per_arm={m:study["steps_per_arm"] for m in protocol["config"]["arms"]},probe_updates=4*protocol["config"]["probe_steps"],
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        runtime=dict(python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)),
        public_summary_sha256=sha(out/"public_summary.json"),public_audit_sha256=sha(out/"additional_audit/public_audit.json"),
        result_sha256=sha(out/"result.json"),replay_result_sha256=sha(checked(read(out/"replay_verified.json"))),
        all_individual_source_identities_weights_and_arrays_retained_on_server=True,
        local_delivery_scope="whitelisted public aggregate JSON and receipt only",
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        generation_and_native_code_unchanged=True,test_evaluations=0,human_reviewed=False,operational_qualified=False)
    write(dest,record);print(json.dumps(record))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("out","receipt","dest"):p.add_argument("--"+k,required=True)
    a=p.parse_args();run(a.out,a.receipt,a.dest)
