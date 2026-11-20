"""R69 quality-record import. JSON validity never authenticates a reviewer.

Private administrator mapping and separately checked identity/exposure roster
are required for ADMIN_ATTESTED_ELIGIBLE. No labels are generated here.
"""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re

METHODS=('Uniform','E1W','BeatDP','RhythmDTW','Single64','Multi64')
DIMENSIONS=('music_coordination','naturalness','transition_timing','continuity','richness')
CONTRASTS=(('E1W','Uniform'),('BeatDP','E1W'),('RhythmDTW','E1W'),('Single64','E1W'),('Multi64','E1W'))
EXPOSURE=('method_identity_seen','prior_named_results_seen','other_reviews_seen')
_SHA=re.compile(r'^[0-9a-fA-F]{64}$')


def text(value,name):
    if not isinstance(value,str) or not value or value.strip()!=value:raise ValueError(name+' must be nonempty text')
    return value


def stamp(value,name):
    text(value,name)
    try:time=datetime.fromisoformat(value.replace('Z','+00:00'))
    except (TypeError,ValueError) as error:raise ValueError(name+' must be ISO8601 with timezone') from error
    if time.utcoffset() is None:raise ValueError(name+' needs a timezone')
    return time.isoformat()


def sha_text(value,name):
    if not isinstance(value,str) or not _SHA.fullmatch(value):raise ValueError(name+' must be SHA256')
    return value.lower()


def admin_index(admin):
    if not isinstance(admin,dict) or admin.get('schema')!='R69_ADMIN_V1':raise ValueError('Private R69 admin mapping required')
    text(admin.get('pack_id'),'pack_id');sha_text(admin.get('protocol_sha256'),'protocol_sha256')
    cases={};keys=set();all_variants=set()
    if not isinstance(admin.get('cases'),list) or not admin['cases']:raise ValueError('Nonempty sealed cases required')
    for case in admin['cases']:
        bid=text(case.get('blind_item_id'),'blind_item_id');key=text(case.get('case_key'),'case_key')
        if bid in cases or key in keys:raise ValueError('Duplicate sealed case identity')
        text(case.get('source_uid'),'source_uid');span=case.get('source_interval')
        if not isinstance(span,list) or len(span)!=2 or any(type(v) is not int for v in span) or not 0<=span[0]<span[1]:raise ValueError('Real source interval required')
        variants=case.get('variants')
        if not isinstance(variants,list) or len(variants)!=6 or {v.get('method') for v in variants}!=set(METHODS):raise ValueError('Exactly six sealed methods required')
        for v in variants:
            vid=text(v.get('blind_variant_id'),'blind_variant_id')
            if vid in all_variants:raise ValueError('Blind variant IDs must be globally unique')
            all_variants.add(vid)
            if type(v.get('available')) is not bool:raise ValueError('Explicit availability required')
            if v['available'] or v.get('trajectory_sha256') is not None:sha_text(v.get('trajectory_sha256'),'trajectory_sha256')
        cases[bid]=case;keys.add(key)
    return cases


def validate_document(doc,admin):
    """Normalize actual imported records; partial/missing cells are never filled."""
    cases=admin_index(admin)
    if not isinstance(doc,dict) or doc.get('schema')!='R69_QUALITY_REVIEW_V1':raise ValueError('Unknown review schema')
    out=deepcopy(doc)
    if out.get('pack_id')!=admin['pack_id'] or sha_text(out.get('protocol_sha256'),'protocol_sha256')!=admin['protocol_sha256'].lower():raise ValueError('Review pack/protocol identity mismatch')
    for field in ('review_id','reviewer_id','reviewer_roster_ref'):text(out.get(field),field)
    if out.get('form_id')!=admin.get('form_id','F01'):raise ValueError('Unknown sealed form')
    out['reviewed_at']=stamp(out.get('reviewed_at'),'reviewed_at')
    if 'status' in out and out['status'] not in ('COMPLETE','PARTIAL','UNSURE'):raise ValueError('Unknown declared review status')
    if not isinstance(out.get('independence'),dict) or type(out['independence'].get('own_judgment')) is not bool:raise ValueError('Explicit own_judgment required')
    exposure=out.get('exposure')
    if not isinstance(exposure,dict) or set(exposure)!=set(EXPOSURE) or any(exposure[k] not in ('YES','NO','UNSURE') for k in EXPOSURE):raise ValueError('Explicit YES/NO/UNSURE exposure observations required')
    if not isinstance(out.get('items'),list):raise ValueError('Items must be a list; partial is allowed')
    seen=set();counts=Counter();addressed_available=0
    for item in out['items']:
        bid=item.get('blind_item_id')
        if bid not in cases or bid in seen:raise ValueError('Unknown or duplicate review case')
        seen.add(bid);variants={v['blind_variant_id']:v for v in cases[bid]['variants']};visited=set()
        if not isinstance(item.get('ratings'),list):raise ValueError('Partial ratings must be a list')
        for rating in item['ratings']:
            vid=rating.get('blind_variant_id')
            if vid not in variants or vid in visited:raise ValueError('Unknown or duplicate blind variant')
            visited.add(vid);dimensions=rating.get('dimensions')
            if not isinstance(dimensions,dict) or not set(dimensions)<=set(DIMENSIONS):raise ValueError('Unknown dimension')
            for dimension,cell in dimensions.items():
                if not isinstance(cell,dict) or cell.get('status') not in ('RATED','UNSURE','UNRATED'):raise ValueError('Explicit rating cell status required')
                status=cell['status'];counts[status]+=1
                if status=='RATED':
                    if type(cell.get('value')) is not int or not 1<=cell['value']<=5:raise ValueError('Ordinal integer rating must be1..5')
                    if not variants[vid]['available']:raise ValueError('Unavailable media cannot receive a rating')
                elif cell.get('value') is not None:raise ValueError('UNSURE/UNRATED cannot contain a numerical default')
                if status=='UNSURE':text(cell.get('reason'),'UNSURE reason')
                if variants[vid]['available'] and status in ('RATED','UNSURE'):addressed_available+=1
    available=sum(v['available'] for c in cases.values() for v in c['variants'])*len(DIMENSIONS)
    out['structural_validation']='VALID_FORMAT_NOT_IDENTITY_AUTHENTICATION'
    out['coverage']=dict(status='COMPLETE' if available and addressed_available==available and len(seen)==len(cases) else 'PARTIAL',
        rated_cells=counts['RATED'],unsure_cells=counts['UNSURE'],explicit_unrated_cells=counts['UNRATED'],
        missing_cells=len(cases)*6*len(DIMENSIONS)-sum(counts.values()),registered_cases=len(cases),
        missing_is_not_zero=True)
    return out


def qualify(doc,admin,roster=None):
    """Administrative attestation is distinct from independent authentication."""
    fail=lambda reason:dict(status='UNVERIFIED',eligible=False,reason=reason,person_uid=None)
    if not doc['independence']['own_judgment'] or any(doc['exposure'][k]!='NO' for k in EXPOSURE):return fail('SELF_REPORTED_EXPOSURE_OR_INDEPENDENCE_NOT_CLEAR')
    if not isinstance(roster,dict) or roster.get('schema')!='R69_REVIEWER_ROSTER_V1' or roster.get('pack_id')!=admin['pack_id']:return fail('SEPARATE_PRIVATE_ROSTER_MISSING_OR_UNBOUND')
    entries=roster.get('reviewers')
    if not isinstance(entries,list):return fail('PRIVATE_ROSTER_ENTRIES_MISSING')
    matches=[r for r in entries if r.get('reviewer_id')==doc['reviewer_id']]
    if len(matches)!=1:return fail('REVIEWER_NOT_UNIQUELY_BOUND_IN_PRIVATE_ROSTER')
    person=matches[0]
    if person.get('reviewer_roster_ref')!=doc['reviewer_roster_ref']:return fail('PRIVATE_IDENTITY_REFERENCE_MISMATCH')
    if person.get('identity_manually_verified') is not True or person.get('exposure_manually_checked') is not True:return fail('IDENTITY_OR_EXPOSURE_NOT_MANUALLY_CHECKED')
    try:text(person.get('person_uid'),'person_uid');text(person.get('verified_by'),'verified_by');stamp(person.get('verified_at'),'verified_at')
    except ValueError:return fail('MANUAL_ATTESTATION_PROVENANCE_INCOMPLETE')
    exposures=person.get('exposure')
    if not isinstance(exposures,dict) or any(type(exposures.get(k)) is not bool or exposures[k] is not False for k in EXPOSURE):return fail('ADMIN_EXPOSURE_NOT_EXPLICITLY_FALSE')
    return dict(status='ADMIN_ATTESTED_ELIGIBLE',eligible=True,reason=None,person_uid=person['person_uid'])


def load_documents(paths,admin):
    return [validate_document(json.loads(Path(path).read_text(encoding='utf-8-sig')),admin) for path in paths]


def _cell(doc,bid,vid,dimension,available):
    if not available:return dict(status='UNAVAILABLE',value=None)
    item=next((v for v in doc['items'] if v['blind_item_id']==bid),None)
    rating=next((v for v in item['ratings'] if v['blind_variant_id']==vid),None) if item else None
    return rating['dimensions'].get(dimension,dict(status='MISSING',value=None)) if rating else dict(status='MISSING',value=None)


def summarize_reviews(admin,documents,roster=None):
    cases=admin_index(admin);docs=[validate_document(d,admin) for d in documents]
    if len({d['review_id'] for d in docs})!=len(docs):raise ValueError('Duplicate review_id; preserve versions without counting twice')
    qualifications={d['review_id']:qualify(d,admin,roster) for d in docs}
    results={};any_independent=False;eligible_people=set()
    for left,right in CONTRASTS:
        results[left+'_minus_'+right]={}
        for dim in DIMENSIONS:
            paired=[];unpaired=[];missing=Counter();person_case=defaultdict(list)
            for doc in docs:
                qualification=qualifications[doc['review_id']]
                for bid,case in cases.items():
                    variants={v['method']:v for v in case['variants']}
                    a=_cell(doc,bid,variants[left]['blind_variant_id'],dim,variants[left]['available'])
                    b=_cell(doc,bid,variants[right]['blind_variant_id'],dim,variants[right]['available'])
                    if a['status']!='RATED' or b['status']!='RATED':
                        missing[a['status']+'/'+b['status']]+=1;unpaired.append(dict(review_id=doc['review_id'],blind_item_id=bid,
                            left_status=a['status'],right_status=b['status'],left_reason=a.get('reason'),right_reason=b.get('reason')));continue
                    delta=a['value']-b['value']
                    paired.append(dict(review_id=doc['review_id'],blind_item_id=bid,left=a['value'],right=b['value'],difference=delta,
                        eligibility=qualification['status']))
                    if qualification['eligible']:
                        person_case[(bid,qualification['person_uid'])].append(delta);eligible_people.add(qualification['person_uid'])
            per_case=defaultdict(list)
            for (bid,person),values in person_case.items():per_case[bid].append(sum(values)/len(values))
            case_rows=[];source=defaultdict(list)
            for bid,case in cases.items():
                people=per_case.get(bid,[]);condition=len(people)>=2;mean=sum(people)/len(people) if condition else None
                any_independent|=condition
                case_rows.append(dict(blind_item_id=bid,case_key=case['case_key'],admin_attested_distinct_persons=len(people),
                    independent_comparison_condition_met=condition,paired_person_balanced_mean=mean))
                if condition:source[(case['source_uid'],tuple(case['source_interval']))].append(mean)
            source_rows=[dict(source_uid=uid,source_interval=list(span),cases=len(v),mean=sum(v)/len(v)) for (uid,span),v in sorted(source.items())]
            results[left+'_minus_'+right][dim]=dict(registered_document_case_slots=len(docs)*len(cases),paired_records=len(paired),
                missing_status_counts=dict(missing),raw_paired_records=paired,unpaired_records=unpaired,case_results=case_rows,
                evaluable_cases=sum(r['independent_comparison_condition_met'] for r in case_rows),
                evaluable_source_intervals=len(source_rows),source_interval_means=source_rows,
                source_interval_balanced_mean=sum(r['mean'] for r in source_rows)/len(source_rows) if source_rows else None,
                ci95=None,p_value=None,ordinal_difference_scope='SAME_REVIEWER_SAME_CASE; PERSONS_BALANCED_WITHIN_CASE; CASES_WITHIN_SOURCE_INTERVAL')
    public_qualification=[dict(review_id=d['review_id'],status=qualifications[d['review_id']]['status'],reason=qualifications[d['review_id']]['reason'],coverage=d['coverage']) for d in docs]
    return dict(schema='R69_QUALITY_RECORD_ANALYSIS_V1',status='ADMIN_ATTESTED_COMMON_COMPARISONS_AVAILABLE' if any_independent else 'NOT_EVALUABLE',
        imported_documents=len(docs),structurally_valid_documents=len(docs),registered_cases=len(cases),
        admin_attested_distinct_people_with_paired_scores=len(eligible_people),
        independent_comparison_condition_met=any_independent,human_reviewed=any_independent,
        authentication_scope='ADMIN_ATTESTATION_NOT_EXTERNAL_IDENTITY_PROOF; JSON_SELF_REPORT_ALONE_NEVER_SUFFICIENT',
        records=public_qualification,comparisons=results,quality_winner=None,can_route=False,Native24='NOT_EVALUABLE',
        uncertainty='UNSURE_UNRATED_MISSING_AND_UNAVAILABLE_NEVER_MAPPED_TO_ZERO_OR_NEUTRAL_SCORE')
