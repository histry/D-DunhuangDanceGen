"""Whitelist aggregate scientific evidence; leave source identities and private artifacts server-side."""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import write,sha

def read(p):return json.loads(Path(p).read_text())
def export(out,receipt,dest):
    out=Path(out);receipt=Path(receipt);g=read(out/"g1/result.json");d=read(out/"a1/result.json")
    # config carries a private event-source location; publish only registered scalar settings.
    cfg=read(out/"g1/protocol.json")["config"];cfg={k:v for k,v in cfg.items() if k!="event_source"}
    tests=re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/"pytest.log").read_text())
    if not tests or "7 passed" not in (receipt/"edge_check.log").read_text():raise ValueError("Required test evidence absent")
    public=dict(schema="public_dense_condition_v1",execution_commit=(receipt/"execution_commit.txt").read_text().strip(),
        g1_config=cfg,a1_config=read(out/"a1/protocol.json")["config"],a1=d,
        g1=dict(generations_executed=g["generations_executed"],native_passes=g["native_passes"],source_preferences=g["source_preferences"],
            condition_effective_on_registered_cases=g["condition_effective_on_registered_cases"],weights_unchanged=g["weights_unchanged"],
            trained_visibility_policy=g["trained_visibility_policy"],new_visibility_policy_trained=False,optimizer_calls=0,
            generations=[{k:r[k] for k in ("case","control","native_ok","continuous","dynamics")} for r in g["generations"]],
            denoising=[dict(case=r["case"],timestep=r["timestep"],metrics=r["metrics"],sensitivity=r.get("sensitivity")) for r in g["denoising"]],
            boundary_generation_started=False,protected_sqp_started=False,operational_qualified=False),
        replay=dict(g1=read(out/"g1_replay/replay.json"),a1=read(out/"a1_replay/replay.json")),
        tests=tests[-1],supplemental_edge_tests="7 passed",new_blind_source=False,operational_qualified=False,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,
        evidence={str(p.relative_to(out)):sha(p) for p in (out/"g1/protocol.json",out/"g1/result.json",out/"g1/denoising.npz",out/"g1/layer_deltas.npz",
            out/"a1/protocol.json",out/"a1/data/manifest.json",out/"a1/data/nodes.npz",out/"a1/data/conditional.npz",out/"a1/result.json")},
        scientific_scope="existing Train/Val source-clock challenge and legacy adapter inference audit; no new paired generation truth or human semantic labels")
    encoded=json.dumps(public)
    if any(k in encoded for k in ("/home/","aist_capture:","known_component:","window_uid","smpl_poses")):raise ValueError("Private export blocked")
    write(dest,public)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k in ("out","receipt","dest"):p.add_argument("--"+k,required=True)
    a=p.parse_args();export(a.out,a.receipt,a.dest)
