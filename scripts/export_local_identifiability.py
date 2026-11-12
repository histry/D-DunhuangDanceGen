"""Only aggregate public evidence. Raw identities, matrices, weights remain on the server."""
from pathlib import Path
import argparse,json,re,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import write,sha,checked
from dunhuang.alignment.clock_data import read

def export(out,receipt,dest):
    out=Path(out);receipt=Path(receipt);p=read(out/"protocol.json");result=read(out/"scientific/result.json")
    replay=read(checked(read(out/"replay_verified.json")))
    if replay["status"]!="FRESH_LOCAL_IDENTIFIABILITY_REPLAY_VERIFIED":raise ValueError("Fresh replay incomplete")
    testlog=(receipt/"pytest.log").read_text();tests=re.findall(r"\d+ passed, \d+ skipped",testlog)
    edge=(receipt/"edge_check.log").read_text()
    if not tests or "1 passed" not in edge:raise ValueError("Dependency-complete test evidence missing")
    frozen=["src/dunhuang/alignment/model.py","src/dunhuang/generation","src/dunhuang/planning","src/dunhuang/evaluation_benchmarks","src/dunhuang/physical.py"]
    if subprocess.check_output(["git","diff","--name-only",p["baseline"],"HEAD","--"]+frozen,text=True).strip():raise ValueError("Frozen generation/encoder/Native code changed")
    public=dict(schema="public_local_identifiability_v1",execution_commit=p["code_commit"],baseline=p["baseline"],config=p["config"],
        result=result,replay=replay,pytest=tests[-1],supplemental_edge_test="1 passed",
        evidence=dict(protocol_sha256=sha(out/"protocol.json"),preflight_sha256=sha(out/"preflight.json"),
            conditional_candidates_sha256=sha(out/"conditional.npz"),result_sha256=sha(out/"scientific/result.json"),
            replay_sha256=sha(checked(read(out/"replay_verified.json"))),pytest_sha256=sha(receipt/"pytest.log"),
            frozen_encoder_generation_native_unchanged=True),
        input_scope="preverified paired AIST Train/Val; cache reused; no new blind source; no raw data publication",
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    encoded=json.dumps(public)
    if any(s in encoded for s in ("/home/","known_component:","aist_capture:","window_uid","smpl_poses")):raise ValueError("Private payload leaked into public export")
    write(dest,public)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("out","receipt","dest"):p.add_argument("--"+k,required=True)
    a=p.parse_args();export(a.out,a.receipt,a.dest)
