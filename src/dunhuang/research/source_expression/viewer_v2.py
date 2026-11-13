"""Presentation-only correction; preserve frozen motion and never certify foot contact."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write
from dunhuang.data.sources import ref
from dunhuang.data.split_registry import stable
from .common import fresh,read,commit
EXPECTED_CATALOG="d33cd546ae746227cfc6e5cad1dee2923c88742631a3c69433b63f4dfa85421a"
EXPECTED_MANIFEST="4d0fc5b837ec196aa711d868379444e78007d0059915663b2eed0fd2e8691d1c"
PARENTS=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]
def old_points(text):
    match=re.search(r"const xyz=(.*?),parents=",text)
    if not match:raise ValueError("Frozen viewer coordinates missing")
    value=np.asarray(json.loads(match.group(1)),dtype=np.float64)
    if value.ndim!=3 or value.shape[1:]!=(22,3) or len(value)<2 or not np.isfinite(value).all():raise ValueError("Invalid frozen 22-joint media")
    if not np.all(value[:,0]==0):raise ValueError("Expected frozen root-relative presentation")
    return value
def modes(relative,root):
    relative=np.asarray(relative,dtype=np.float64);root=np.asarray(root,dtype=np.float64)
    if root.shape!=(len(relative),3) or not np.isfinite(root).all():raise ValueError("Finite same-clock fitted Root required")
    # One constant translation leaves the fitted trajectory unchanged.
    offset=root-root[:1]
    return dict(pose=relative,world=relative+offset[:,None,:])
def page(relative,root,task,audio=None):
    values=modes(relative,root)
    payload=json.dumps(dict(pose=values["pose"].tolist(),world=np.round(values["world"],6).tolist()),separators=(",",":"))
    template=Path(__file__).with_name("viewer_v2.html").read_text(encoding="utf-8")
    return template.replace("__DATA__",payload).replace("__PARENTS__",json.dumps(PARENTS)).replace("__TASK__",task).replace("__AUDIO__",'<audio controls preload="metadata" src="'+audio+'"></audio>' if audio else "")
def index_page(manifest):
    template=Path(__file__).with_name("review_ui.html").read_text(encoding="utf-8")
    template=template.replace("<!doctype html>","<!doctype html><html lang='zh-CN' translate='no'><meta name='google' content='notranslate'>")
    template=template.replace("Independent review V1 — Preparation only","独立盲审 · 播放修正版（仅准备）")
    template=template.replace("Use your assigned private copy. Work independently. No actual human labels exist in this package.","每人使用独立副本，独立判断。当前没有真实人工审核记录。")
    template=template.replace("UNSURE is an allowed result; add a reason. EVENT and POSE are separate from MUSIC_MATCH compatibility.","UNSURE 表示无法判断，须填写原因；EVENT、POSE 与 MUSIC_MATCH 分开审核。")
    template=template.replace("Reviewer ID","审核人编号").replace("Private roster reference","私有名册引用").replace("Save original review","保存本人的审核记录").replace("Download original submissions","下载原始审核记录")
    return template.replace("__ITEMS__",json.dumps(manifest)).replace("__VERSION__","independent_dance_review_v1")

def build(source,catalog,out,replay_from=None):
    source=Path(source);catalog=Path(catalog);out=fresh(out);public=out/"reviewer";admin=out/"administrator";public.mkdir();admin.mkdir()
    if sha(catalog)!=EXPECTED_CATALOG:raise ValueError("Frozen Catalog differs")
    summary=read("reports/source_expression_summary.json");receipt=read("reports/source_expression_receipt.json")
    if sha("reports/source_expression_summary.json")!=receipt["summary_sha256"] or receipt["exit_status"]!=0:raise ValueError("R6 successful receipt differs")
    if not summary["review"]["actual_records"]==0:raise ValueError("Existing human reviews require a separately registered media migration")
    # Reconstruct the old source mapping and verify all opaque identities against the actual packet.
    previous_protocol=source.parent/"protocol.json";previous_binding=source.parent/"administrator/bindings.json"
    if sha(previous_protocol)!=summary["evidence"]["review/protocol.json"]:raise ValueError("Previous preparation protocol changed")
    old_protocol=read(previous_protocol)
    if sha(previous_binding)!=old_protocol["bindings"]["sha256"]:raise ValueError("Previous private binding changed")
    old_bindings={b["blind_item_id"]:b for b in read(previous_binding)}
    cfg=read("configs/source_expression_r6_v1.json");entries=read(catalog)["entries"];selected=[]
    for uid in dict.fromkeys(e["source_uid"] for e in entries):
        selected.extend(sorted([e for e in entries if e["source_uid"]==uid],key=lambda e:stable([cfg["selection_seed"],e["event_uid"]]))[:2])
    tasks=[(e,t) for e in selected for t in ("EVENT","POSE")]+[(selected[i],"MUSIC_MATCH") for i in range(3)]
    if sha(source/"items.json")!=EXPECTED_MANIFEST:raise ValueError("Frozen manifest differs")
    items=read(source/"items.json")
    if len(tasks)!=35 or len(items)!=35:raise ValueError("Frozen review population differs")
    manifest=[];bindings=[];diagnostics=[];world_checks=[];native_cache={}
    for i,((event,task),item) in enumerate(zip(tasks,items)):
        bid=hashlib.sha256((str(cfg["selection_seed"])+str(i)+event["event_uid"]+task).encode()).hexdigest()[:24]
        if item["blind_item_id"]!=bid or item["task_type"]!=task or item["media"]!=bid+".html":raise ValueError("Frozen blind identity differs")
        old=source/item["media"]
        if sha(old)!=item["media_sha256"]:raise ValueError("Frozen reviewer media changed")
        binding=old_bindings[bid]
        if binding["source_uid"]!=event["source_uid"] or binding["event_uid"]!=event["event_uid"] or [binding["start_frame"],binding["end_frame"]]!=event["source_interval"] or binding["native"]["sha256"]!=event["reference"]["sha256"]:raise ValueError("Independent source binding differs")
        uid=event["source_uid"]
        if uid not in native_cache:
            path=catalog.parent/(uid+".npy")
            if sha(path)!=event["reference"]["sha256"]:raise ValueError("Local native source is not the frozen original")
            native_cache[uid]=np.load(path)
        native=native_cache[uid];a,b=event["source_interval"];lo=max(0,a-12);hi=min(len(native),b+12)
        relative=old_points(old.read_text(encoding="utf-8"));root=native[lo:hi,4:7].astype(np.float64)
        if len(relative)!=hi-lo:raise ValueError("Source display clock differs")
        views=modes(relative,root);audio=bid+".wav" if task=="MUSIC_MATCH" else None
        if audio:shutil.copyfile(source/audio,public/audio)
        html=page(relative,root,task,audio);media=public/(bid+".html");media.write_text(html,encoding="utf-8")
        manifest.append(dict(item,media_sha256=sha(media),presentation_version="viewer_v2"))
        payload=re.search(r"const motion=(.*?),parents=",html)
        reconstructed=json.loads(payload.group(1))
        if not np.array_equal(np.array(reconstructed["pose"]),relative):raise ValueError("Pose samples changed")
        if not np.allclose(np.array(reconstructed["world"])-np.array(reconstructed["pose"]),root[:,None,:]-root[0,None,:],atol=1.01e-6,rtol=0):raise ValueError("World reconstruction is not the original fitted Root translation")
        world_checks.append(float(np.max(np.abs(np.array(reconstructed["world"])-views["world"]))))
        if task=="EVENT":
            step=np.linalg.norm(np.diff(relative,axis=0),axis=-1).mean(-1);p95=float(np.quantile(step,.95));seam=float(np.linalg.norm(relative[-1]-relative[0],axis=-1).mean())
            diagnostics.append(dict(blind_item_id=bid,frames=len(relative),old_loop_seam_m=seam,within_clip_step_p95_m=p95,seam_to_p95=seam/max(p95,1e-9),
                fitted_root_extent_m=np.ptp(root,axis=0).tolist(),kinematic_proxy_only=True))
        bindings.append(dict(binding,**dict(blind_item_id=bid,task_type=task,source_uid=uid,event_uid=event["event_uid"],raw_motion_sha256=event["source_sha256"],native_sha256=event["reference"]["sha256"],
            start_frame=a,end_frame=b,display_interval=[lo,hi],fps=30,media_sha256=sha(media),previous_media_sha256=item["media_sha256"],media=ref(media),
            fitted_root_observation="UNVERIFIED_FIT_NOT_WORLD_OR_CONTACT_TRUTH",presentation_version="viewer_v2",audio_sha256=sha(public/audio) if audio else None,audio=ref(public/audio) if audio else None)))
    write(public/"items.json",manifest)
    (public/"index.html").write_text(index_page(manifest),encoding="utf-8")
    (public/"README.txt").write_text("PREPARATION_ONLY / viewer_v2. Original discrete frames, no smoothing, no automatic loop. World view includes unverified fitted Root, not contact ground truth. Pose view fixes Root and cannot establish foot sliding. Old viewer packet is retained. Reviewer identity and evidence admission remain required.\n",encoding="utf-8")
    write(admin/"bindings.json",bindings);write(out/"diagnostics.json",diagnostics)
    result=dict(schema="blind_viewer_presentation_v2",items=len(manifest),unique_events=len(diagnostics),all_pose_frames_unchanged=True,
        fitted_root_sources_sha_verified=len(native_cache),world_rounding_max_m=max(world_checks),fps=30,clock="requestAnimationFrame with elapsed wall/audio clock",
        automatic_loop=False,smoothing=False,root_truth="UNKNOWN",contact_truth="UNKNOWN",human_reviewed=False,actual_human_records=0,operational_qualified=False,
        loop_jump_ratio_min=min(d["seam_to_p95"] for d in diagnostics),loop_jump_ratio_max=max(d["seam_to_p95"] for d in diagnostics))
    write(out/"result.json",result);write(out/"protocol.json",dict(code_commit=commit(),catalog_sha256=sha(catalog),original_manifest_sha256=sha(source/"items.json"),
        original_successful_summary_sha256=receipt["summary_sha256"],viewer_source_sha256=sha(Path(__file__)),viewer_template_sha256=sha(Path(__file__).with_name("viewer_v2.html")),
        bindings=ref(admin/"bindings.json"),previous_protocol_sha256=sha(previous_protocol),outputs={p.name:sha(p) for p in public.iterdir() if p.is_file()},binding_sha256=sha(admin/"bindings.json"),diagnostics_sha256=sha(out/"diagnostics.json")))
    if replay_from:
        previous=Path(replay_from)
        if read(previous/"result.json")!=result or read(previous/"diagnostics.json")!=diagnostics:raise ValueError("Viewer replay metadata differs")
        for file in public.iterdir():
            if sha(file)!=sha(previous/"reviewer"/file.name):raise ValueError("Viewer replay bytes differ")
        write(out/"replay.json",dict(status="FRESH_VIEWER_V2_MEDIA_REPLAY",exact_media_files=len(list(public.iterdir())),optimizer_calls=0,actual_human_records=0))
    print("VIEWER_V2_BUILT",len(manifest),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--catalog",required=True);p.add_argument("--out",required=True);p.add_argument("--replay-from")
    a=p.parse_args();build(a.source,a.catalog,a.out,a.replay_from)
