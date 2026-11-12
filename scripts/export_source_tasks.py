"""Aggregate-only release; training and diagnostic gates remain distinct."""
import argparse,json,re,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text())
def export(out,receipt,seal=False):
    out=Path(out);receipt=Path(receipt);execution=(receipt/"execution_commit.txt").read_text().strip()
    parts={"effect":"effect_main","pairs":"pairs","observations":"content_data","cf":"cf","content":"content"}
    public=dict(schema="public_source_tasks_v1",execution_commit=execution,
        results={k:read(out/v/"result.json") for k,v in parts.items()},
        replays={k:read(out/(v+"_replay" if k!="effect" else "effect_replay")/"replay.json") for k,v in parts.items()},
        tests={k:(re.findall(r"\d+ passed(?:, \d+ skipped)?",(receipt/name).read_text()) or [None])[-1] for k,name in (("main","pytest.log"),("edge","edge_check.log"))},
        evidence={v+"/"+name:sha(out/v/name) for v in parts.values() for name in ("protocol.json","result.json")},
        model_sha256={},test_tensor_reads=0,human_reviewed=False,operational_qualified=False,g3_started=False,
        temporary_ssh_files_created=0,temporary_ssh_files_remaining=0)
    if any(v["optimizer_calls"] for v in public["replays"].values()):raise ValueError("Replay optimizer calls detected")
    for v in parts.values():
        if read(out/v/"protocol.json")["code_commit"]!=execution:raise ValueError("Execution code changed within study")
    cg=public["results"]["pairs"]["qualification"]["qualified"];ag=public["results"]["observations"]["qualification"]["qualified"]
    if public["results"]["cf"]["generations"]!=(44 if cg else 0) or public["replays"]["cf"]["regenerated"]!=(44 if cg else 0):raise ValueError("CF generation budget differs")
    for part,qualified,arms,model in (("cf",cg,("C0","C1","C2","C3"),"adapter.pt"),("content",ag,("15","30","dual","dual_shuffle"),"content.pt")):
        if public["results"][part]["optimizer_updates"]!=(1600 if qualified else 0):raise ValueError("Training budget differs")
        if qualified:
            for arm in arms:public["model_sha256"][part+"/"+arm]=sha(out/part/arm/model)
    if not all(public["tests"].values()):raise ValueError("Test result missing")
    text=json.dumps(public)
    if any(t in text for t in ("/home/","window_uid","source_uid","aist_capture:","password")):raise ValueError("Private source content in public output")
    if not seal:write(out/"public_summary.json",public);return
    if (receipt/"exit_status.txt").read_text().strip()!="0" or public!=read(out/"public_summary.json"):raise ValueError("Successful sealed launcher required")
    baseline="acce669df14f21c1780412ca54b61acfd2e2ccf0"
    old=subprocess.check_output(["git","ls-tree","-r","--name-only",baseline,"--","src","configs"],text=True).splitlines()
    frozen=[p for p in old if p!="src/dunhuang/pipeline.py"]
    if subprocess.check_output(["git","diff","--name-only",baseline,execution,"--",*frozen],text=True).strip():raise ValueError("Frozen baseline implementation changed")
    write(out/"public_receipt.json",dict(schema="source_tasks_sealed_receipt",baseline=baseline,execution_commit=execution,
        verifier_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),exit_status=0,
        baseline_except_explicit_pipeline_conflict_guard_unchanged=True,pipeline_guard_sha256=sha("src/dunhuang/pipeline.py"),
        main_optimizer_updates=sum(public["results"][k]["optimizer_updates"] for k in ("cf","content")),replay_optimizer_calls=0,
        started_utc=(receipt/"started_utc.txt").read_text().strip(),finished_utc=(receipt/"finished_utc.txt").read_text().strip(),
        summary_sha256=sha(out/"public_summary.json"),launcher_sha256=sha("scripts/run_source_tasks.sh"),
        receipt_sha256={p.name:sha(p) for p in receipt.iterdir() if p.is_file()},temporary_ssh_files_created=0,temporary_ssh_files_remaining=0,operational_qualified=False))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);p.add_argument("--receipt",required=True);p.add_argument("--seal",action="store_true");a=p.parse_args();export(a.out,a.receipt,a.seal)
