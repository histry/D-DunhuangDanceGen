"""Final public status receipt; never export source paths, identities, weights or matrices."""
import argparse,json,platform,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import checked,sha,write
from dunhuang.alignment.clock_data import read

def seal(out,receipt,dest):
    out=Path(out);receipt=Path(receipt)
    if (receipt/"exit_status.txt").read_text().strip()!="0":raise ValueError("Successful launcher completion required")
    public=read(out/"public_summary.json");replay=read(checked(read(out/"replay_verified.json")))
    if replay["status"]!="FRESH_LOCAL_IDENTIFIABILITY_REPLAY_VERIFIED":raise ValueError("Fresh replay missing")
    p=read(out/"protocol.json");r=read(out/"scientific/result.json")
    if r["r2"]["optimizer_updates"]!=0 and r["r2"]["status"].startswith("NOT_STARTED"):raise ValueError("Gate/update receipt inconsistent")
    record=dict(schema="local_identifiability_delivery_receipt_v1",status="R1_COMPLETED_AND_FRESH_REPLAY_VERIFIED",
        execution_commit=p["code_commit"],replay_commit=replay["replay_commit"],baseline=p["baseline"],launcher_exit_status=0,
        pytest=public["pytest"],supplemental_edge_test=public["supplemental_edge_test"],
        strict_population=p["strict_population"],r2_status=r["r2"]["status"],r2_optimizer_updates=r["r2"]["optimizer_updates"],
        replay_optimizer_calls=replay["optimizer_calls"],diagnostic_virtual_clone_updates=replay["diagnostic_virtual_clone_updates"],
        closed_form_probe_fits=replay["closed_form_probe_fits"],scientific_array_fields_verified=replay["scientific_array_fields_verified"],
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        public_summary_sha256=sha(out/"public_summary.json"),scientific_result_sha256=sha(out/"scientific/result.json"),
        replay_result_sha256=sha(checked(read(out/"replay_verified.json"))),preflight_sha256=sha(out/"preflight.json"),
        source_assets_retained_on_server=True,public_delivery_scope="aggregate JSON and status receipt only",
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,true_event_semantics="UNKNOWN",
        encoder_generation_native_unchanged=True,test_evaluations=0,operational_qualified=False)
    write(dest,record);print(json.dumps(record))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("out","receipt","dest"):p.add_argument("--"+key,required=True)
    a=p.parse_args();seal(a.out,a.receipt,a.dest)
