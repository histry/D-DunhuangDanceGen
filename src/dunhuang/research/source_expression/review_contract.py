"""User's independent EVENT/POSE/MUSIC_MATCH review V1. No automatic human truth."""
import datetime,hashlib,json
from collections import Counter

TASKS={"EVENT":{"VALID","INVALID","UNSURE"},"POSE":{"VALID","INVALID","UNSURE"},"MUSIC_MATCH":{"COMPATIBLE","INCOMPATIBLE","UNSURE"}}
REASONS={"VISIBILITY_INSUFFICIENT","BOUNDARY_AMBIGUOUS","SOURCE_INCOMPLETE","CULTURAL_SEMANTICS_UNKNOWN","MUSIC_CONTEXT_INSUFFICIENT","TECHNICAL_FAILURE","OTHER"}
VERSION="independent_dance_review_v1"

def time_valid(value):
    if not isinstance(value,str):raise ValueError("Timezone-aware human review time required")
    try:t=datetime.datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError:raise ValueError("Invalid review time")
    if t.tzinfo is None or t.utcoffset() is None:raise ValueError("Review time lacks timezone")
    return t

def validate(record,binding,roster,roster_ref):
    required=("review_id","blind_item_id","reviewer_id","reviewer_roster_ref","task_type","media_sha256","reviewed_at","decision","confidence","reason_code","review_protocol_version","model_predictions_shown","label_origin")
    if any(k not in record or record[k] in (None,"") for k in required):raise ValueError("Incomplete original human submission")
    if record["blind_item_id"]!=binding["blind_item_id"] or record["task_type"]!=binding["task_type"]:raise ValueError("Blind task identity differs")
    if record["review_protocol_version"]!=VERSION or record["label_origin"]!="independent_human_review" or record["model_predictions_shown"] is not False:raise ValueError("Automatic/model-derived labels cannot become human truth")
    if record["reviewer_roster_ref"]!=roster_ref:raise ValueError("Private reviewer roster binding differs")
    reviewer=roster.get(record["reviewer_id"])
    if not reviewer or reviewer.get("identity_verified") is not True or reviewer.get("independent_attested") is not True or not reviewer.get("private_identity_ref"):raise ValueError("Unverified reviewer identity/independence")
    if record["media_sha256"]!=binding["media_sha256"]:raise ValueError("Reviewed evidence differs")
    if record["decision"] not in TASKS[binding["task_type"]] or record["confidence"] not in ("LOW","MEDIUM","HIGH"):raise ValueError("Typed human judgment required")
    if record["decision"]=="UNSURE" and record["reason_code"] not in REASONS:raise ValueError("UNSURE requires an explicit reason")
    if record["reason_code"]=="OTHER" and not record.get("notes","").strip():raise ValueError("OTHER requires explanation")
    time_valid(record["reviewed_at"])
    for key in ("media_sha256",):
        if len(record[key])!=64 or any(c not in "0123456789abcdef" for c in record[key]):raise ValueError("Invalid evidence SHA256")
    # Source identity is administrator-bound after review, never shown in the blind UI.
    out=dict(record)
    for k in ("source_uid","raw_motion_sha256","event_uid","start_frame","end_frame","fps"):
        if k not in binding:raise ValueError("Missing original source identity/clock")
        if k in record and record[k]!=binding[k]:raise ValueError("Post-review source binding differs")
        out[k]=binding[k]
    if not 0<=out["start_frame"]<out["end_frame"] or out["fps"]!=30:raise ValueError("Unsupported source clock")
    return out

def summarize(bindings,records,roster,roster_ref,adjudications=()):
    index={b["blind_item_id"]:b for b in bindings};accepted=[];seen=set();by_item={}
    if len(index)!=len(bindings):raise ValueError("Duplicate blind item identity")
    for record in records:
        if record.get("review_id") in seen:raise ValueError("Duplicate original review ID")
        seen.add(record.get("review_id"));item=index.get(record.get("blind_item_id"))
        if item is None:raise ValueError("Unknown blind item")
        r=validate(record,item,roster,roster_ref);accepted.append(r);by_item.setdefault(r["blind_item_id"],[]).append(r)
    decisions=[];conflicts=0;agreements=0;reasons=Counter();states=Counter()
    adjudication_index={a["blind_item_id"]:a for a in adjudications}
    if len(adjudication_index)!=len(adjudications) or set(adjudication_index)-set(index):raise ValueError("Invalid duplicate/unknown adjudication")
    adj_ids=[a.get("adjudication_id") for a in adjudications]
    if any(not isinstance(i,str) or not i for i in adj_ids) or len(set(adj_ids))!=len(adj_ids):raise ValueError("Unique adjudication IDs required")
    for item in bindings:
        reviews=by_item.get(item["blind_item_id"],[]);ids=[r["reviewer_id"] for r in reviews]
        if len(ids)!=len(set(ids)):raise ValueError("One reviewer cannot cast duplicate task votes")
        if not reviews:state="UNREVIEWED";decision=None
        elif len(reviews)<2:state="PARTIAL_REVIEW";decision=None
        else:
            a,b=reviews[:2]
            if a["decision"]==b["decision"]:
                if any(r["decision"]!=a["decision"] for r in reviews[2:]):raise ValueError("Additional contradictory votes require a separately registered adjudication protocol")
                decision=a["decision"];agreements+=1
            else:
                conflicts+=1;adj=adjudication_index.get(item["blind_item_id"]);decision=None
                if adj:
                    if set(adj.get("original_review_ids",[]))!={a["review_id"],b["review_id"]}:raise ValueError("Adjudication lost original opinions")
                    third=next((r for r in reviews if r["review_id"]==adj.get("adjudicator_review_id")),None)
                    if not third or third["reviewer_id"] in (a["reviewer_id"],b["reviewer_id"]):raise ValueError("Independent third review required")
                    if time_valid(adj.get("adjudicated_at"))<max(time_valid(r["reviewed_at"]) for r in (a,b,third)):raise ValueError("Adjudication precedes its original evidence")
                    decision=third["decision"]
                    for r in reviews:r["adjudication_id"]=adj["adjudication_id"]
            state="UNSURE" if decision=="UNSURE" else "HUMAN_REVIEW_VERIFIED" if decision is not None else "PARTIAL_REVIEW"
        states[state]+=1
        for r in reviews:
            if r["decision"]=="UNSURE":reasons[r["reason_code"]]+=1
        decisions.append(dict(blind_item_id=item["blind_item_id"],task_type=item["task_type"],state=state,decision=decision,original_review_ids=[r["review_id"] for r in reviews],adjudication_id=adjudication_index.get(item["blind_item_id"],{}).get("adjudication_id")))
    return dict(states=dict(states),original_records=len(accepted),reviewers=len({r["reviewer_id"] for r in accepted}),initial_agreements=agreements,initial_disagreements=conflicts,
        unsure_reasons=dict(reasons),data_review_verified=bool(bindings) and states["HUMAN_REVIEW_VERIFIED"]==len(bindings),
        model_evaluation_independence=False,semantic_event_truth="UNKNOWN",crossmodal_event_correspondence="NOT_EVALUABLE",operational_qualified=False),decisions,accepted
