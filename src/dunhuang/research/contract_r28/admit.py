"""Bind completed A evidence and fixed paired training recipe; never tune on held-out sampling."""
import argparse
from pathlib import Path
from dunhuang.research.state_r27.common import read,write,sha,commit
from .evidence import factorial_dir

def run(root):
    root=Path(root);out=root/"training_admission.json";assert not out.exists()
    a=read(root/"data/result.json");assert a==read(root/"data_replay/result.json")
    folder=factorial_dir(root);f=read(folder/"result.json");assert f==read(folder.with_name(folder.name+"_replay")/"result.json")
    assert read(root/"readout/result.json")==read(root/"readout_replay/result.json")
    assert read(root/"fine_audio/result.json")==read(root/"fine_audio_replay/result.json")
    audit=read(root/"input_domain_audit.json");replay=read(root/"input_domain_audit_replay.json")
    assert {k:v for k,v in audit.items() if k!="code_commit"}=={k:v for k,v in replay.items() if k!="code_commit"}
    assert all(r["official_first_frame_parity_max"]<2e-6 for r in a["records"])
    assert all(r["physical_reference_fk_max_m"]<2e-5 for r in f["reference_parities"])
    write(out,dict(code_commit=commit(),arms=["T0","T1"],steps_per_arm=1200,per_target_presentations=1200,
        initialization="FROZEN_R26_C1_NOT_R27_P",canonical_reference=True,Core_C4_loss_applied=False,
        rationale="Published floor/C4 package differs; compare both at equal real-source dose without measured-contact claims",
        factorial_cells=f["cells"],scaler_transform_clips=audit["scaler_transform_clips"],
        evidence={str(p.relative_to(root)):sha(p) for p in (root/"data/result.json",folder/"result.json",root/"readout/result.json",root/"input_domain_audit.json")},
        no_DEV_or_ChangE_model_selection=True,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
