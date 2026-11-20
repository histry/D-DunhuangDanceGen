"""Non-additive R67 bank costs for R69 reuse; no experiments are executed.

Only request receipts establish actual geometry calls. Encoding counts are
inferred from frozen R67 run_main/encode_bank, not instrumented measurements.
"""
import hashlib
import json


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
        ensure_ascii=False,allow_nan=False).encode('utf8')).hexdigest()


def _name(value):
    if not isinstance(value,str) or not value or value.strip()!=value:
        raise ValueError('Nonempty exact frozen case/candidate identity required')
    return value


def _count(value,name,request=False):
    if value is None:return None
    if type(value) is not int or value<0 or request and value not in (0,1):
        raise ValueError('Invalid saved nonnegative count: '+name)
    return value


def _sum_known(values):
    known=[v for v in values if v is not None]
    return dict(total=sum(known) if len(known)==len(values) else None,
        known_sum=sum(known),known_requests=len(known),missing_requests=len(values)-len(known))


def shared_historical_costs(banks,records,registration):
    """Describe one shared historical bank, never allocate it to three arms.

    Inputs are the full frozen R67 banks/records/registration JSON values.
    Caller must separately verify their original manifest and file-byte SHA.
    Missing receipt counts remain unknown even when global counters exist.
    """
    if not isinstance(banks,list) or not banks or not isinstance(records,list) or not isinstance(registration,dict):
        raise ValueError('Original R67 structured JSON values required')
    indexed={}
    for row in records:
        if not isinstance(row,dict):raise ValueError('Every original candidate record must be structured')
        key=(_name(row.get('case_key')),_name(row.get('candidate_uid')))
        if key in indexed:raise ValueError('Duplicate original candidate record')
        indexed[key]=row
    ledgers=[];seen_cases=set();used=set();all_calls=[];all_cached=[]
    for entry in banks:
        if not isinstance(entry,dict):raise ValueError('Every frozen bank must be structured')
        case=_name(entry.get('case_key'));bank=entry.get('bank')
        if case in seen_cases:raise ValueError('Duplicate original case bank would count costs twice')
        seen_cases.add(case)
        candidates=bank.get('candidates') if isinstance(bank,dict) else None
        if not isinstance(candidates,list) or not candidates:raise ValueError('Nonempty original candidate bank required')
        receipts=[];calls=[];cached=[];identities=set()
        for candidate in candidates:
            if not isinstance(candidate,dict):raise ValueError('Structured original candidate required')
            uid=_name(candidate.get('uid'));key=(case,uid)
            if uid in identities:raise ValueError('Duplicate candidate UID inside frozen bank')
            if key not in indexed:raise ValueError('Original bank candidate lacks its saved record')
            identities.add(uid);used.add(key);row=indexed[key]
            receipt=row.get('connectivity')
            if receipt is None:receipt={}
            if not isinstance(receipt,dict):raise ValueError('Original connectivity receipt must be structured or absent')
            actual=_count(receipt.get('actual_geometry_calls_this_request'),'actual_geometry_calls_this_request',True)
            reuse=_count(receipt.get('cached_geometry_reuses_this_request'),'cached_geometry_reuses_this_request',True)
            if actual is not None and reuse is not None and actual+reuse!=1:
                raise ValueError('Saved request must be either one actual solve or one cache reuse')
            calls.append(actual);cached.append(reuse)
            receipts.append(dict(candidate_uid=uid,actual_geometry_calls=actual,cached_geometry_reuses=reuse,
                receipt_present=bool(receipt),cache_key=receipt.get('cache_key')))
        all_calls.extend(calls);all_cached.extend(cached)
        ledgers.append(dict(case_key=case,actual_candidate_count=len(candidates),
            motion_encoding_calls_inferred=len(candidates),music_encoding_calls_inferred=1,
            geometry_check_requests_inferred=len(candidates),actual_geometry_calls=_sum_known(calls),
            cached_geometry_reuses=_sum_known(cached),candidate_receipts=receipts))
    if used!=set(indexed):raise ValueError('Unmatched original records are not silently dropped from the ledger')
    counters=registration.get('connectivity_counts')
    if counters is None:counters={}
    if not isinstance(counters,dict):raise ValueError('Saved global connectivity counters must be structured')
    registered={k:_count(counters.get(k),k) for k in ('check_requests','actual_geometry_calls','cached_geometry_reuses')}
    observed=dict(check_requests=len(used),actual_geometry_calls=_sum_known(all_calls)['total'],
        cached_geometry_reuses=_sum_known(all_cached)['total'])
    comparisons={}
    for key in registered:
        comparisons[key]=None if registered[key] is None or observed[key] is None else registered[key]==observed[key]
        if comparisons[key] is False:raise ValueError('Saved request receipts disagree with registration: '+key)
    return dict(schema='R69_SHARED_HISTORICAL_BANK_COSTS_V1',registered_case_banks=len(ledgers),
        distinct_candidate_record_keys=len(used),banks=ledgers,
        totals=dict(motion_encoding_calls_inferred=len(used),music_encoding_calls_inferred=len(ledgers),
            geometry_check_requests_inferred=len(used),actual_geometry_calls=_sum_known(all_calls),
            cached_geometry_reuses=_sum_known(all_cached)),registration_counts=registered,
        registration_receipt_agreement=comparisons,
        input_canonical_json_sha256={k:_fingerprint(v) for k,v in
            (('banks',banks),('records',records),('registration',registration))},
        encoding_count_basis='INFERRED_FROM_FROZEN_R67_RUN_MAIN_ENCODE_BANK_NOT_INSTRUMENTED',
        encoding_count_unit='ENCODE_ROTATIONS_AND_ENCODE_MUSIC_RAW_API_INVOCATIONS_NOT_NEURAL_FORWARD_CALLS_BATCHES_FLOPS_OR_TIME',
        encoding_inference='ONE_MOTION_ENCODE_PER_CANDIDATE_AND_ONE_MUSIC_ENCODE_PER_CASE_SHARED_BY_SINGLE64_MULTI64',
        geometry_count_basis='ACTUAL_GEOMETRY_CALLS_THIS_REQUEST_AND_CACHED_GEOMETRY_REUSES_THIS_REQUEST; MISSING_IS_UNKNOWN',
        historical_wall_seconds=None,
        allocation_scope='NON_ADDITIVE_PARENT_BANK_LEDGER_COUNT_ONCE; NEVER_SUM_E1W_SINGLE64_MULTI64_ROW_COUNTS',
        E1W_scope='EXACT_BASELINE_CANDIDATE_REUSE; NO_SEPARATE_FULL_BANK_ENCODING_OR_SEARCH_COST_ALLOCATED',
        independent_sample_scope='BANKS_AND_CANDIDATES_ARE_COST_UNITS_NOT_INDEPENDENT_RECORDINGS',
        current_R69_encoding_calls=0,can_route=False,quality_winner=None)
