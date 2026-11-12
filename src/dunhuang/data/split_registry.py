"""Versioned task-specific identity isolation; never rewrite historical roles."""
import hashlib,json,math
from collections import Counter

TASK_KEYS={
 "strict_unseen_song":("music_work_uid","recording_uid","source_sha256","audio_sha256"),
 "unseen_choreography":("choreography_uid","recording_uid","source_sha256"),
 "unseen_performer":("performer_uid","recording_uid","source_sha256"),
 "unseen_motion_source":("recording_uid","source_sha256"),
}
REQUIRED={"strict_unseen_song":"music_work_uid","unseen_choreography":"choreography_uid","unseen_performer":"performer_uid","unseen_motion_source":"recording_uid"}

def stable(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def components(rows,task):
    if task not in TASK_KEYS:raise ValueError("Unknown isolation task")
    required=REQUIRED[task]
    if any(not r.get(required) for r in rows):raise ValueError("UNKNOWN task identity cannot be partitioned")
    parent=list(range(len(rows)));seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        for key in TASK_KEYS[task]:
            values=[r[key]] if r.get(key) else []
            if key=="music_work_uid":values+=r.get("music_work_aliases",[])
            for v in values:
                pair=(key,v)
                if pair in seen:parent[root(i)]=root(seen[pair])
                else:seen[pair]=i
    grouped={}
    for i in range(len(rows)):grouped.setdefault(root(i),[]).append(i)
    return sorted(grouped.values(),key=lambda ids:stable(sorted(rows[i]["source_uid"] for i in ids)))

def partition(rows,task,seed,fractions=(.8,.1,.1)):
    if len(fractions)!=3 or not math.isclose(sum(fractions),1):raise ValueError("Explicit train/val/test fractions required")
    groups=components(rows,task)
    if len(groups)<3:return {},dict(status="NOT_EVALUABLE_INSUFFICIENT_GROUPS",groups=len(groups))
    keyed=[]
    for ids in groups:
        genres={rows[i].get("genre","UNKNOWN") for i in ids}
        genre=next(iter(genres)) if len(genres)==1 else "MULTI"
        keyed.append((stable([seed,task,sorted(rows[i]["source_uid"] for i in ids)]),genre,ids))
    buckets={}
    for uid,genre,ids in sorted(keyed):buckets.setdefault(genre,[]).append((uid,ids))
    genres=sorted(buckets,key=lambda g:stable([seed,g]));selected={}
    targets={"test":max(1,round(len(groups)*fractions[2])),"val":max(1,round(len(groups)*fractions[1]))}
    offset=0
    for role in ("test","val"):
        count=0
        while count<targets[role]:
            progressed=False
            for k in range(len(genres)):
                genre=genres[(offset+k)%len(genres)]
                # Keep at least one group from every known stratum in Train.
                if len(buckets[genre])>1:
                    uid,ids=buckets[genre].pop(0)
                    for i in ids:selected[rows[i]["source_uid"]]=role
                    count+=1;progressed=True
                    if count==targets[role]:break
            if not progressed:break
        offset=(offset+count)%len(genres)
        if count!=targets[role]:return {},dict(status="NOT_EVALUABLE_STRATUM_SUPPORT",groups=len(groups),requested=targets)
    for bucket in buckets.values():
        for _,ids in bucket:
            for i in ids:selected[rows[i]["source_uid"]]="train"
    result=attest(rows,selected,task)
    result.update(groups=len(groups),seed=seed,algorithm="sha256_stratified_group_round_robin_v1",target_group_counts=targets)
    return selected,result

def attest(rows,roles,task):
    if task not in TASK_KEYS and task!="official_crossmodal":raise ValueError("Unknown role attestation")
    if any(roles.get(r["source_uid"]) not in ("train","val","test") for r in rows):raise ValueError("Missing explicit source role")
    if task in REQUIRED and any(not r.get(REQUIRED[task]) for r in rows):raise ValueError("UNKNOWN required identity cannot pass attestation")
    keys=TASK_KEYS.get(task,("recording_uid","source_sha256","choreography_uid","music_work_uid","audio_sha256"))
    membership={};violations=[]
    for r in rows:
        role=roles[r["source_uid"]]
        for key in keys:
            for v in ([r[key]] if r.get(key) else [])+(r.get("music_work_aliases",[]) if key=="music_work_uid" else []):
                pair=(key,v)
                for other in membership.get(pair,set()):
                    shared_official=(task=="official_crossmodal" and key in ("music_work_uid","audio_sha256") and {role,other}=={"val","test"})
                    if role!=other and not shared_official:violations.append(dict(key=key,roles=sorted((role,other))))
                membership.setdefault(pair,set()).add(role)
    return dict(status="PASSED_KNOWN_IDENTITY_ISOLATION" if not violations else "FAILED_IDENTITY_OVERLAP",
        forbidden_identity_overlaps=len(violations),violations=violations,role_sources=dict(Counter(roles[r["source_uid"]] for r in rows)),
        role_music={role:len({r.get("music_work_uid") for r in rows if roles[r["source_uid"]]==role and r.get("music_work_uid")}) for role in ("train","val","test")},
        fresh_blind_claimed=False,checkpoint_qualification_inherited=False)

def leave_one_recording_out(rows,seed):
    groups=components(rows,"unseen_motion_source")
    if len(groups)<3:raise ValueError("At least three independent recordings required")
    order=sorted(groups,key=lambda ids:stable([seed,sorted(rows[i]["source_uid"] for i in ids)]));folds=[]
    for outer,test in enumerate(order):
        validation=order[(outer+1)%len(order)];roles={}
        for i,r in enumerate(rows):roles[r["source_uid"]]="test" if i in test else "val" if i in validation else "train"
        report=attest(rows,roles,"unseen_motion_source")
        folds.append(dict(fold=outer,roles=roles,attestation=report,index_rebuild_required=True,train_only_statistics_required=True,
            existing_full_catalog_checkpoint_qualified=False,training_executed=False))
    return folds

def catalog_subset(catalog,source_roles,role="train"):
    if role not in ("train","val","test"):raise ValueError("Explicit fold role required")
    entries=[e for e in catalog["entries"] if source_roles.get(e["source_uid"])==role]
    return dict(catalog,entries=entries,statistics_fitted=False,fold_dependent=True,can_route=False)

def checkpoint_binding(registry,protocol,training_source_uids):
    p=registry["protocols"][protocol]
    intended=sorted(uid for uid,role in p["roles"].items() if role=="train")
    if not intended or sorted(training_source_uids)!=intended:raise ValueError("Checkpoint population differs from task protocol")
    return stable(dict(schema="task_checkpoint_binding_v3",protocol=protocol,roles=p["roles"],train=intended,
        identity_scope=p["identity_scope"],normalization_scope="train_only",registry_identity=registry["identity_digest"]))

def training_population(registry,protocol):
    p=registry["protocols"][protocol]
    if p.get("ready_for_new_training") is not True:raise ValueError("Split V3 population is NOT_EVALUABLE or not qualified")
    return [r for r in registry["sources"] if p["roles"].get(r["source_uid"])=="train"]
