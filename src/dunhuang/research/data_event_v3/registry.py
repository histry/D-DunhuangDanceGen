"""Three-dataset source ledger; official roles and new task roles remain independent."""
import ast,collections,json,re
from pathlib import Path
from dunhuang.io import checked,sha,write
from dunhuang.data.sources import raw_path
from dunhuang.data.split_registry import partition,attest,leave_one_recording_out,stable
from dunhuang.data.single_person_admission import admit

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))

def official_finedance(path):
    tree=ast.parse(Path(path).read_text(encoding="utf-8"))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_train_test_list")
    def literal(n):
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add):return literal(n.left)+literal(n.right)
        return ast.literal_eval(n)
    result={}
    for branch in ast.walk(fn):
        if not isinstance(branch,ast.If) or not isinstance(branch.test,ast.Compare):continue
        test=branch.test
        if not isinstance(test.left,ast.Name) or test.left.id!="datasplit" or not isinstance(test.comparators[0],ast.Constant):continue
        name=test.comparators[0].value;values={}
        for n in branch.body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ("test_list","ignor_list"):values[n.targets[0].id]=literal(n.value)
        if not values:continue
        ignore=set(values["ignor_list"]);test=set(values["test_list"]);all_ids={str(i).zfill(3) for i in range(1,212)}
        result[name]=dict(train=all_ids-test-ignore,test=test-ignore,ignore=ignore,declared_test=test)
    if set(result)!={"cross_genre","cross_dancer"}:raise ValueError("Official split function structure changed")
    return result

def source_exposure(row):
    return dict(status="HISTORICAL_OR_EXPOSURE_UNCERTIFIED",historical_role=row.get("purpose_role"),previous_feature_read=row.get("previous_fixed_observation_consumed",False),
        declared_exposure=row.get("historical_project_exposure","UNKNOWN"),fresh_blind=False,
        pretrained_backbone_exposure="UNKNOWN_NOT_CERTIFIED")

def build(assets,population,config,out):
    reference=Path(assets["asset_root"]);datasets=reference/"assets/datasets"
    population_file=Path(population);raw=read(population_file)["records"]
    paired=read(checked(assets["paired_generation_manifest"]))
    if not paired["recording_independence_verified"] or paired["blind_test_claimed"]:raise ValueError("Historical paired identity/exposure differs")
    checked(paired["split_attestation"])
    pair_by_sha={}
    for r in paired["rows"]:
        pair_by_sha.setdefault(r["raw_motion"]["sha256"],[]).append(r)
    source_catalog=read(checked(assets["catalog_source"]))
    roster_path=reference/"assets/motion/smpl_official_14/sources.json";roster=read(roster_path)
    official_root=datasets/"aistpp/annotations/splits"
    official={role:set((official_root/("crossmodal_"+role+".txt")).read_text().split()) for role in ("train","val","test")}
    if any(official[a]&official[b] for a,b in (("train","val"),("train","test"),("val","test"))):raise ValueError("Official sequence lists overlap")
    fd_path=datasets/"finedance/metadata/official_FineDance_dataset.py"
    fd_meta_path=datasets/"finedance/metadata/official_format_split_summary.json";fd_meta=read(fd_meta_path)
    expected=next(x["sha256"] for x in fd_meta["source_snapshots"] if x["path"]==fd_path.name)
    if sha(fd_path)!=expected:raise ValueError("Official FineDance code snapshot seal differs")
    fd=official_finedance(fd_path)
    snapshots=[ref(population_file),assets["paired_generation_manifest"],assets["catalog_source"],ref(roster_path),ref(fd_path),ref(fd_meta_path)]
    snapshots+=[ref(official_root/("crossmodal_"+r+".txt")) for r in official]
    sources=[]
    for old in sorted(raw,key=lambda r:(r["dataset_id"],r["sequence_id"])):
        dataset=old["dataset_id"];raw_file=None;audio_file=None
        for candidate_root in (Path("assets/datasets"),datasets):
            try:
                raw_file=raw_path(candidate_root,old["motion_asset"],required=True)
                audio_file=raw_path(candidate_root,old["audio_asset"],required=True)
                break
            except (ValueError,FileNotFoundError):raw_file=None;audio_file=None;continue
        if raw_file is None or audio_file is None:raise ValueError("Registered raw assets are not materialized")
        raw_sha=old["motion_asset"]["sha256"];windows=pair_by_sha.get(raw_sha,[])
        if windows:
            sample=windows[0]
            if any(any(w[k]!=sample[k] for k in ("recording_uid","music_uid","choreography_uid","role")) for w in windows):raise ValueError("Source identity or old role differs across windows")
            for w in windows:
                proof=read(checked(w["alignment_attestation"]))
                if proof["status"]!="verified" or proof["motion_sha256"]!=raw_sha or proof["audio_sha256"]!=w["audio"]["sha256"]:raise ValueError("Paired clock evidence differs")
        row=dict(source_uid=dataset+":"+raw_sha,dataset_id=dataset,source_sha256=raw_sha,
            raw_source=dict(path=str(raw_file),sha256=raw_sha),audio_sha256=old["audio_asset"]["sha256"],
            audio_version_uid="sha256:"+old["audio_asset"]["sha256"],recording_uid=old.get("recording_uid"),
            sequence_id=old["sequence_id"],performer_uid=None,performer_count=None,solo_verified=False,solo_evidence=None,
            music_work_uid=None,music_work_aliases=[],choreography_uid=None,genre=old.get("style","UNKNOWN"),
            historical_role=old["purpose_role"],historical_source_group=old["group_uid"],exposure=source_exposure(old),
            roles={},observed_joints=list(range(24 if dataset=="aistpp" else 22)),full151_target_ready=bool(windows),
            pairing_status="VERIFIED" if windows else "PUBLISHER_PAIR_METADATA_ONLY",
            audio_offset_seconds=windows[0]["audio_offset_seconds"] if windows else old.get("audio_offset_seconds"),
            windows_with_full151_and_jukebox=len(windows),observation_schema="smpl24_native151" if dataset=="aistpp" else "smplh52_body22_partial",
            new_statistics_fitted=False)
        if dataset=="aistpp":
            name=old["sequence_id"]
            matches=[role for role,ids in official.items() if name in ids]
            if len(matches)>1:raise ValueError("Official AIST membership ambiguous")
            row["roles"]["official_crossmodal"]=matches[0] if matches else "OUTSIDE_OFFICIAL_SPLIT"
            match=re.fullmatch(r"(g[A-Z]{2})_(sBM|sFM)_cAll_(d[0-9]{2})_(m[A-Z]{2}[0-5])_(ch[0-9]{2})",name)
            if windows and match:
                genre,situation,dancer,music,ch=match.groups()
                row.update(recording_uid=windows[0]["recording_uid"],performer_uid="aist_official_dancer:"+dancer,
                    performer_count=1,solo_verified=True,solo_evidence=dict(method="publisher_individual_dancer_identity_plus_verified_refined_clock",clock=windows[0]["alignment_attestation"]),
                    music_work_uid="aist_publisher_track:"+music,choreography_uid=windows[0]["choreography_uid"],genre=genre,
                    work_identity_scope="publisher canonical music ID; cross-ID composition aliases not certified",pairing_status="VERIFIED")
        elif dataset=="finedance":
            for name,rule in fd.items():
                row["roles"]["official_"+name]="excluded" if old["sequence_id"] in rule["ignore"] else "test" if old["sequence_id"] in rule["test"] else "train" if old["sequence_id"] in rule["train"] else "UNKNOWN"
            row.update(work_identity_scope="UNKNOWN: labels/name or WAV bytes alone do not authenticate a composition",
                full151_target_ready=False,single_track_array=True,solo_evidence=None,
                representation_blockers=["two unobserved canonical hand joints","units/coordinate and source-clock admission incomplete"],
                val_status="PENDING_VERIFIED_WORK_IDENTITY_WITHIN_OFFICIAL_TRAIN")
        row["admission"]=admit(row);sources.append(row)
    selected_source_shas={r["source_sha256"] for r in source_catalog["source_records"]}
    for record in roster["sources"]:
        raw_file=reference/"assets/motion/smpl_official_14"/record["file"];checked(dict(path=str(raw_file),sha256=record["sha256"]))
        row=dict(source_uid="change:"+record["sha256"],dataset_id="change",source_sha256=record["sha256"],raw_source=ref(raw_file),
            sequence_id=record["source_id"],recording_uid=record["recording_uid"],performer_uid=None,
            performer_count=record["recording_performer_count"],solo_verified=record["solo_compatible"] is True and record["recording_performer_count"]==1,
            solo_evidence=dict(method="upstream_solo_aware_manifest_v3",manifest=ref(roster_path),review_status=record["solo_review_status"]),
            music_work_uid=None,audio_version_uid=None,audio_sha256=None,choreography_uid=None,genre=record.get("dance_category","UNKNOWN"),
            pairing_status="UNPAIRED",observed_joints=list(range(22)),observation_schema="body22_native151_partial",
            full151_target_ready=False,historical_role=next((s["historical_recording_split"] for s in source_catalog["source_records"] if s["source_sha256"]==record["sha256"]),"INVENTORY_ONLY_OR_UNKNOWN"),exposure=dict(status="HISTORICAL_OR_EXPOSURE_UNCERTIFIED",fresh_blind=False,performer_identity="UNKNOWN"),
            roles=dict(new_music_known_catalog="CATALOG" if record["sha256"] in selected_source_shas else "EXCLUDED_MULTI_PERFORMER"),
            catalog_member=record["sha256"] in selected_source_shas,new_statistics_fitted=False)
        row["admission"]=admit(row)
        if row["catalog_member"] and not row["admission"]["solo_admitted"]:raise ValueError("Shared Catalog contains unresolved multi-performer source")
        sources.append(row)
    change=[r for r in sources if r["dataset_id"]=="change" and r["catalog_member"]]
    if len(change)!=8 or {r["source_sha256"] for r in change}!=selected_source_shas:raise ValueError("Frozen eight-source Catalog changed")
    protocols={}
    eligible=[r for r in sources if r["dataset_id"]=="aistpp" and r["pairing_status"]=="VERIFIED" and r["admission"]["solo_admitted"]]
    official_eligible=[r for r in eligible if r["roles"]["official_crossmodal"] in ("train","val","test")]
    roles={r["source_uid"]:r["roles"]["official_crossmodal"] for r in official_eligible}
    report=attest(official_eligible,roles,"official_crossmodal")
    protocols["AIST_Official_Crossmodal"]=dict(roles=roles,attestation=report,identity_scope="official music/choreography; Val/Test music sharing allowed",
        ready_for_new_training=False,new_train_only_statistics_required=True,reason="AUDIT_ONLY_NEW_STATS_AND_CHECKPOINT_LINEAGE_REQUIRED")
    for task in ("strict_unseen_song","unseen_choreography","unseen_performer"):
        subset=[r for r in eligible if r.get({"strict_unseen_song":"music_work_uid","unseen_choreography":"choreography_uid","unseen_performer":"performer_uid"}[task])]
        roles,report=partition(subset,task,config["seed"])
        protocols["AIST_"+task]=dict(roles=roles,attestation=report,identity_scope=task+" within verified publisher identities",
            ready_for_new_training=False,new_train_only_statistics_required=True,reason="AUDIT_ONLY_NEW_STATS_AND_CHECKPOINT_LINEAGE_REQUIRED",
            composition_aliases_outside_publisher_ids="UNKNOWN",pretrained_backbone_exposure="UNKNOWN_NOT_CERTIFIED")
        for r in subset:r["roles"][task]=roles.get(r["source_uid"],"NOT_EVALUABLE")
    for name in fd:
        subset=[r for r in sources if r["dataset_id"]=="finedance"]
        roles={r["source_uid"]:r["roles"]["official_"+name] for r in subset}
        protocols["FineDance_"+name]=dict(roles=roles,identity_scope="publisher source ID membership only; song/dancer identities pending",
            attestation=dict(status="NOT_EVALUABLE_UNVERIFIED_SOLO_WORK_CLOCK",official_membership_counts=dict(collections.Counter(roles.values()))),
            ready_for_new_training=False,normalization_scope="NOT_FITTED",test_membership_changed=False,inner_val_created=False)
    folds=leave_one_recording_out(change,config["seed"])
    protocols["ChangE_Catalog"]=dict(roles={r["source_uid"]:"CATALOG" for r in change},identity_scope="fixed shared eight-source resource, not unseen-source evaluation",
        ready_for_new_training=False,music_pairing_truth="UNKNOWN_UNPAIRED")
    protocols["ChangE_Source_Heldout"]=dict(roles={},folds=folds,identity_scope="whole recording and raw SHA, eight outer folds",
        ready_for_new_training=False,existing_full_catalog_statistics_reuse_qualified=False)
    unknown_music=dict(status="NOT_EVALUABLE_UNAUTHENTICATED_WORKS",work_identity_inferred_from_filenames=False)
    old_music_doc=reference/"docs/unpaired_data_protocol_result.json"
    if old_music_doc.exists():
        d=read(old_music_doc);unknown_music["historical_metadata"]=ref(old_music_doc)
        unknown_music["historical_inventory_evidence"]=d.get("music",{})
        # Private historical packet retained in full; no inherited role authorization.
    registry=dict(schema="source_split_registry_v3",sources=sources,protocols=protocols,source_snapshots=snapshots,
        music_query_registry=unknown_music,statistics_fit_calls=0,optimizer_calls=0,human_reviewed=False,operational_qualified=False)
    registry["identity_digest"]=stable(dict(sources=[{k:v for k,v in r.items() if k not in ("raw_source","solo_evidence")} for r in sources],
        roles={k:v["roles"] for k,v in protocols.items()}))
    overlaps=sum(p.get("attestation",{}).get("forbidden_identity_overlaps",0) for p in protocols.values())
    exposure=dict(schema="data_exposure_audit_v3",all_sources_preserve_prior_exposure_or_unknown=True,fresh_blind_claimed=False,
        old_roles_rewritten=False,old_checkpoint_qualification_inherited=False,pretrained_backbone_exposure="UNKNOWN_NOT_CERTIFIED",
        source_counts=dict(collections.Counter(r["dataset_id"] for r in sources)),unknown_performer_sources=sum(not r.get("performer_uid") for r in sources),
        unknown_music_work_sources=sum(not r.get("music_work_uid") for r in sources),optimizer_calls=0)
    attestation=dict(schema="split_attestation_v3",forbidden_identity_overlaps=overlaps,
        protocol_status={k:p.get("attestation",{}).get("status","CATALOG_OR_FOLD_METADATA_ONLY") for k,p in protocols.items()},
        all_tasks_ready_for_training=False,statistics_refitted=False,source_identity_digest=registry["identity_digest"],
        reason="Per-task isolation and unknown identities remain explicit; this stage performs zero fitting")
    write(out/"source_registry_v3.json",registry);write(out/"split_attestation_v3.json",attestation);write(out/"data_exposure_audit.json",exposure)
    return registry,attestation,exposure
