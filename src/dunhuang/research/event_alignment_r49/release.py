"""Build measured release statistics and recheck new and frozen science artifacts."""
from collections import Counter
from pathlib import Path
import subprocess
import numpy as np
from dunhuang.research.complete_r36.core import read, write
from dunhuang.research.reperformance_r44.replay import digest

from .rank import PREVIEW_ROOT

PUB = Path("outputs/r49_publication")
CONTROLS = Path("outputs/event_alignment_r49_controls")
VAL = Path("outputs/event_alignment_r49_val")


def verify():
    primary = read(PREVIEW_ROOT/"result.json")
    controls = read(CONTROLS/"result.json")
    assert primary["model"]==controls["model"]
    for a,b in zip(primary["rows"],controls["rows"]):
        actual = {**b,"candidates":[{k:v for k,v in r.items() if k!="controls"} for r in b["candidates"]]}
        assert a==actual
    for path, replay in ((CONTROLS,Path(str(CONTROLS)+"_replay")),(VAL,Path(str(VAL)+"_replay"))):
        assert digest(path/"result.json")==digest(replay/"result.json"), str(path)
    assert digest(PUB/"qualification.json")==digest(PUB/"qualification_replay.json")
    watch = read(PREVIEW_ROOT/"browser_receipt.json")
    assert watch["passed"] and watch["protocol_sha256"]==digest(PREVIEW_ROOT/"viewer/protocol.json")
    assert watch["verified_unix"] < (CONTROLS/"result.json").stat().st_mtime
    parents = []
    # Recheck all R48 science archives listed in its authoritative publication.
    sealed = read(Path("reports/r48_release/seal_recheck.json"))["records"]
    for entry in sealed:
        root = Path(entry["root"])
        manifest = root/"artifacts_manifest.json"
        assert digest(manifest)==entry["manifest_sha256"]
        for file in read(manifest)["files"]:
            path = root/file["path"]
            assert path.stat().st_size==file["bytes"] and digest(path)==file["sha256"], str(path)
        parents.append(entry)
    write(PUB/"independent.json",dict(exact_main_against_two_recomputations=True,
          exact_controls_and_val_and_qualification=True, primary_before_controls=True,
          verified_parent_files=sum(p["files"] for p in parents), parent_roots=parents))


def release():
    independent = read(PUB/"independent.json")
    r = read(CONTROLS/"result.json")
    v = read(VAL/"result.json")
    q = read(PUB/"qualification.json")
    rows = [p for p in r["rows"] if p["candidates"]]
    metrics = {}
    for method in ("B0_activity_proxy","B1_linear_local","B2_linear_local_multi","B4_coverage","swapped_local"):
        totals = Counter()
        for slot in rows:
            support = [x for x in slot["candidates"] if x["controls"][method] is not None]
            if not support:
                continue
            compare = "local" if method=="swapped_local" else "score"
            base = max(support,key=lambda a:(a[compare],a["event_uid"]))
            best = max(support,key=lambda a:(a["controls"][method],a["event_uid"]))
            totals["common_slots"] += 1
            totals["changed_top"] += int(base["event_uid"]!=best["event_uid"])
            totals["candidate_count"] += len(support)
        metrics[method]=dict(totals)
    phase = {k:x for k,x in q["phase"].items() if k!="records"}
    geometry = [{k:x for k,x in t.items() if k not in ("metrics","G3")} for t in q["geometry"]["tasks"]]
    checked = dict(model=r["model"], r48_event_sha256=r["event_index_sha256"], songs=r["inputs"],
        ranked_slots=len(rows), registered_variants=r["registered_variants"], scored_variants=r["scored_variants"],
        rank_changes=metrics, val=v["summary"], val_role=v["split"], blind=v["registered_blind_test"],
        phase=phase, geometry=geometry, P4=q["P4"], independent=independent, can_route=False,
        test_summary=(PUB/"pytest.log").read_text().strip().splitlines()[-1],
        code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    write(PUB/"summary.json",checked)
    files = []
    for root in (PREVIEW_ROOT,CONTROLS,VAL,Path(str(CONTROLS)+"_replay"),Path(str(VAL)+"_replay")):
        for f in sorted(root.rglob("*")):
            if f.is_file():
                files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
    for p in ("summary.json","independent.json","qualification.json","qualification_replay.json"):
        f=PUB/p
        files.append(dict(path=str(f),bytes=f.stat().st_size,sha256=digest(f)))
    write(PUB/"seal.json",dict(files=files))
    print("R49_RELEASE",len(files),v["summary"],flush=True)


def recheck():
    files = read(PUB/"seal.json")["files"]
    for f in files:
        p = Path(f["path"])
        assert p.stat().st_size==f["bytes"] and digest(p)==f["sha256"], str(p)
    write(PUB/"seal_recheck.json",dict(files=len(files),all_hashes_verified=True,
          code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()))
