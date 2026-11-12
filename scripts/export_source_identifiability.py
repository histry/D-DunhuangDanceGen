"""Only aggregate results leave the server; no raw identities, paths or weights."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text())
def export(out,receipt):
    out=Path(out);receipt=Path(receipt)
    g=read(out/"g2/result.json");a=read(out/"a2/result.json")
    gr=read(out/"g2_replay/replay.json");ar=read(out/"a2_replay/replay.json")
    if gr["optimizer_calls"] or ar["optimizer_calls"]:raise ValueError("Optimizer-free replay absent")
    gp=read(out/"g2/protocol.json");ap=read(out/"a2/protocol.json")
    configs={key:{k:v for k,v in obj.items() if k!="event_source"} for key,obj in (("g2",gp["config"]),("a2",ap["config"]))}
    if g["generations"]!=44 or g["native_evaluations"]!=52 or gr["regenerated"]!=44:raise ValueError("Registered generation budget incomplete")
    counts=lambda name:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1]
    public=dict(schema="public_g2_a2_v1",execution_commit=(receipt/"execution_commit.txt").read_text().strip(),
        configs=configs,g2=g,a2=a,replay=dict(g2=gr,a2=ar),tests=dict(main=counts("pytest.log"),edge=counts("edge_check.log")),
        evidence={name:sha(out/name) for name in ("g2/protocol.json","g2/result.json","g2/denoising.npz","g2/exchange_layers.npz",
            "g2/training_schedule.npy","g2/donor_schedule.npy","a2/protocol.json","a2/candidates.npz","a2/result.json","g2_replay/replay.json","a2_replay/replay.json")},
        model_hashes={arm:sha(out/"g2"/arm/"adapter.pt") for arm in ("g2_1","g2_2","g2_3")},
        adapter_parameters=gp["adapter_parameters"],initial_parameter_digest=gp["initial_parameter_digest"],frozen_parameter_digest=gp["frozen_parameter_digest"],initialization_sha256=gp["initial_adapter"]["sha256"],frozen_base_sha256=gp["base"]["sha256"],
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False)
    text=json.dumps(public)
    if any(x in text for x in ("/home/","aist_capture:","known_component:","window_uid","smpl_poses")):raise ValueError("Private report rejected")
    if not public["tests"]["main"] or not public["tests"]["edge"]:raise ValueError("Regression test evidence absent")
    write(out/"public_summary.json",public)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True)
    a=p.parse_args();export(a.out,a.receipt)
