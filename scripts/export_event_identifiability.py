"""Explicit aggregate whitelist. No raw identities, absolute paths or licensed data."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write,checked
def export(out,receipt,dest):
    out=Path(out);receipt=Path(receipt)
    result=json.loads((out/"result.json").read_text());protocol=json.loads((out/"protocol.json").read_text())
    pre=json.loads((out/"preflight.json").read_text());data=json.loads((out/"data/manifest.json").read_text())
    replay=json.loads(checked(json.loads((out/"replay_verified.json").read_text())).read_text()) if (out/"replay_verified.json").exists() else json.loads((out/"replay/result.json").read_text())
    if replay["status"]!="FRESH_EVENT_IDENTIFIABILITY_REPLAY_VERIFIED":raise ValueError("Fresh replay incomplete")
    operator=result["operator"]
    safe_operator={k:v for k,v in operator.items() if k!="frozen_heads"}
    safe_operator["frozen_heads"]={k:{a:b for a,b in v.items() if a!="checkpoint"} for k,v in operator["frozen_heads"].items()}
    value=dict(schema="event_identifiability_public_summary_v1",status=result["status"],
        execution_commit=protocol["code_commit"],config=protocol["config"],results=result["results"],
        baselines=result["baselines"],probes=result["probes"],operator=safe_operator,initializations=result["initializations"],
        steps_per_arm=result["steps_per_arm"],source_rows=result["source_rows"],validation_queries=result["validation_queries"],
        validation_components=result["validation_components"],test_evaluations=0,operational_qualified=False,
        downstream_generation_executed=False,qualification_blockers=result["qualification_blockers"],seconds=result["seconds"],
        population=dict(train_rows=sum(r["role"]=="train" for r in data["rows"]),val_rows=sum(r["role"]=="val" for r in data["rows"]),
            train_eligible=data["train_eligible"],val_eligible=data["val_eligible"],old_schedule=pre["prior_schedule"],new_schedule=pre["schedule"]),
        weak_label_states=data["weak_label_states"],human_reviewed=False,replay=replay,
        provenance=dict(protocol_sha256=sha(out/"protocol.json"),data_sha256=sha(out/"data/events.npz"),
            synthetic_truth_sha256=sha(out/"synthetic_truth.npz"),preflight_sha256=sha(out/"preflight.json"),
            heads={m:sha(out/m/"head.pt") for m in protocol["config"]["arms"]},
            measurements={m:sha(out/m/"measurements.npz") for m in protocol["config"]["arms"]},
            pytest_log_sha256=sha(receipt/"pytest.log")))
    encoded=json.dumps(value)
    if any(x in encoded for x in ("/home/","C:\\","D:\\","window_uid","known_component:")):
        raise ValueError("Raw identity or private path in public aggregate")
    write(dest,value)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("out","receipt","dest"):p.add_argument("--"+k,required=True)
    a=p.parse_args();export(a.out,a.receipt,a.dest)
