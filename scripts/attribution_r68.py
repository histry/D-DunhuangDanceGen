"""Read-only paired attribution from sealed R68 published scalar metadata.

No motion arrays, optimization, new hypothesis selection or quality decision.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

HEX=re.compile(r'^[0-9a-f]{64}$')
METHODS=('E1W','Uniform','A_early','A_late','B_early','B_late')
GROUPS=('touch_target','touch_compensation','touch_declared_edit',
        'cross_declared_edit_boundary','fully_outside_edit_domain')
PAIRINGS={'A_early_minus_late':('A_early','A_late'),'B_early_minus_late':('B_early','B_late'),
          'A_minus_B_early':('A_early','B_early'),'A_minus_B_late':('A_late','B_late')}


def read(path):return json.loads(Path(path).read_bytes().decode('utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def value(doc,*path):
    for key in path:
        if not isinstance(doc,dict) or key not in doc:return None
        doc=doc[key]
    return float(doc) if type(doc) in (int,float) and math.isfinite(doc) else None


def verified_release(directory):
    directory=Path(directory).resolve();manifest=directory/'public_manifest.json';doc=read(manifest)
    if not isinstance(doc.get('files'),list):raise ValueError('Published file list required')
    seen={}
    for item in doc['files']:
        raw=item['path'].replace('\\','/');relative=Path(raw)
        if not raw or raw.startswith('/') or ':' in raw or relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe published manifest member')
        name=relative.as_posix();p=(directory/relative).resolve()
        if (name in seen or not p.is_relative_to(directory) or not p.is_file()
                or not HEX.fullmatch(item['sha256']) or sha(p)!=item['sha256']
                or p.stat().st_size!=item['bytes']):raise ValueError('Published bytes or membership differ: '+name)
        seen[name]=item['sha256']
    required=('controls_records.json','controls_population.json')
    if any(name not in seen for name in required):raise ValueError('Required controls metadata is not published in the seal')
    return directory,dict(public_manifest_sha256=sha(manifest),all_published_file_hashes_verified=True,
        published_files=len(seen),inputs_sha256={name:seen[name] for name in required})


def music_metadata(row):
    cross=row.get('music_cross_invariance');base=dict(independently_rehashed_motion_arrays=False,
        scope='PUBLISHED_IDENTIFIER_STRUCTURE_AND_COHERENCE_ONLY_NOT_INDEPENDENT_ARRAY_OR_AUDIO_VALIDATION')
    if not isinstance(cross,dict):return dict(base,status='NOT_AVAILABLE')
    keys=('motion_R_sha256','q_sha256','shared_same_dynamics_sha256','shared_same_pair_dynamics_sha256',
          'shared_same_bridge_R_sha256','correct_audio_sha256','alternate_audio_sha256')
    structure=all(isinstance(cross.get(k),str) and HEX.fullmatch(cross[k]) for k in keys)
    if not structure:return dict(base,status='INCOMPLETE_OR_MALFORMED',missing_or_invalid_fields=[k for k in keys if not isinstance(cross.get(k),str) or not HEX.fullmatch(cross[k])])
    coherent=(cross['motion_R_sha256']==row.get('R_sha256') and cross['q_sha256']==row.get('q_sha256')
        and cross['shared_same_bridge_R_sha256']==row.get('connection',{}).get('bridge_R_sha256')
        and cross['correct_audio_sha256']==row.get('audio_sha256'))
    return dict(base,status='STRUCTURALLY_COMPLETE_AND_IDENTIFIERS_COHERENT' if coherent else 'INCONSISTENT_PUBLISHED_IDENTIFIERS',
        identifiers= {k:cross[k] for k in keys},source_claim_shared_motion=cross.get('shared_same_motion_array'),
        source_claim_shared_dynamics=cross.get('shared_same_dynamics_record'))


def row_metrics(row,baseline):
    if row.get('status')!='LEGAL':return None,dict(status='NOT_EVALUATED_NO_LEGAL_OUTPUT')
    metrics=row['metrics'];d3=metrics['stencils']['D3'];groups=d3['groups']
    partitions=[groups['partition_'+k]['stencil_count'] for k in ('target_event','compensation','protected','untouched')]
    if any(type(n) is not int or n<0 for n in partitions) or sum(partitions)!=237 or d3['full_stencil_count']!=237:
        raise ValueError('D3 partition must retain all237 full four-frame stencils')
    checks=dict(partition237_verified=True,boundary_contract=metrics.get('boundary_contract'))
    if row['method']!='Uniform':
        f=metrics['footprint']
        if f.get('protected_q_and_R_bytes_exact') is not True or f.get('unchanged_outside_declared_edit_exact') is not True:
            raise ValueError('Strict internal protection statement is missing or false')
        for space in ('FK','SO3'):
            outside=groups['fully_outside_edit_domain']['spaces'][space]['signed_magnitude_delta']
            if outside['count']<1 or outside.get('minimum')!=0 or outside.get('maximum')!=0:
                raise ValueError('Outside-domain '+space+' D3 change must be exactly zero')
        b=row['bridge_invariance'];connection=row['connection']
        if (b.get('boundary_readset_identical') is not True or b.get('actual_bridge_bytes_identical') is not True
                or not isinstance(connection.get('bridge_R_sha256'),str)
                or not HEX.fullmatch(connection['bridge_R_sha256'])
                or b.get('baseline_bridge_R_sha256')!=connection['bridge_R_sha256']
                or not isinstance(baseline,dict)
                or baseline.get('connection',{}).get('bridge_R_sha256')!=connection['bridge_R_sha256']):
            raise ValueError('Strict internal bridge identifiers or readset invariance differ')
        checks.update(strict_internal_published_checks_verified=True,raw_arrays_independently_rehashed=False)
    else:checks['strict_internal_gate_applied']=False
    values={'D3_'+name+'_signed_magnitude_delta_mean':value(groups,name,'spaces','FK','signed_magnitude_delta','mean') for name in GROUPS}
    actual=value(metrics,'whole','actual','FK','D3','p95');ref=value(metrics,'whole','baseline','FK','D3','p95')
    values['whole_D3_p95_difference']=actual-ref if actual is not None and ref is not None else None
    values['maximum_actual_source_clock_change_frames']=value(metrics,'dose','maximum_abs_source_position_change_frames')
    actual=value(metrics,'activity_normalized_oscillation','actual','FK_velocity_6Hz_to_Nyquist_power_fraction')
    ref=value(metrics,'activity_normalized_oscillation','baseline','FK_velocity_6Hz_to_Nyquist_power_fraction')
    values['FK_6Hz_to_Nyquist_power_fraction_difference']=actual-ref if actual is not None and ref is not None else None
    gap=value(row,'connection','metrics','gap_D3');base_gap=value(baseline,'connection','metrics','gap_D3')
    values['G1_gap_D3_difference']=gap-base_gap if gap is not None and base_gap is not None else None
    checks['G1_gap_D3_baseline_source']='SAME_CASE_E1W_CONNECTION' if base_gap is not None else 'NOT_AVAILABLE_NOT_ZERO_FILLED'
    checks['music_cross_metadata']=music_metadata(row)
    return values,checks


def balanced(rows,field):
    units=defaultdict(list);available=[]
    for row in rows:
        v=row['differences'].get(field)
        if v is not None:units[(row['source_uid'],tuple(row['source_interval']))].append(v);available.append(v)
    means=[dict(source_uid=uid,source_interval=list(span),pairs=len(v),mean=sum(v)/len(v)) for (uid,span),v in sorted(units.items())]
    return dict(registered_pairs=len(rows),evaluable_pairs=len(available),missing_pairs=len(rows)-len(available),
        evaluable_source_intervals=len(means),source_interval_means=means,
        source_interval_balanced_mean=sum(v['mean'] for v in means)/len(means) if means else None,
        position_weighted_mean=sum(available)/len(available) if available else None,
        minimum=min(available) if available else None,maximum=max(available) if available else None,ci95=None)


def analyze(directory):
    directory,inputs=verified_release(directory);records=read(directory/'controls_records.json');population=read(directory/'controls_population.json')
    cases=population['cases'];ids={c['key']:c for c in cases}
    if len(ids)!=len(cases) or len(cases)!=12:raise ValueError('Exactly the12 frozen case positions are required')
    intervals={(c['source_uid'],tuple(c['source_interval'])) for c in cases}
    if len(intervals)!=8:raise ValueError('This diagnostic is registered for the8 frozen source intervals')
    indexed={}
    for row in records:
        key=(row['case_key'],row['method'])
        if key in indexed or key[0] not in ids or key[1] not in METHODS:raise ValueError('Duplicated or unregistered case/method row')
        identity=ids[key[0]]
        if row.get('source_uid')!=identity['source_uid'] or row.get('source_interval')!=identity['source_interval']:
            raise ValueError('Published record source identity differs from frozen population')
        indexed[key]=row
    output=[];by_method={}
    for method in METHODS:
        own=[]
        for case in cases:
            raw=indexed.get((case['key'],method));status=raw.get('status','UNKNOWN') if raw else 'MISSING_RECORD'
            values,checks=row_metrics(raw,indexed.get((case['key'],'E1W'))) if raw else (None,dict(status='MISSING_RECORD'))
            own.append(dict(case_key=case['key'],method=method,source_uid=case['source_uid'],source_interval=case['source_interval'],
                status=status,reason=raw.get('reason') if raw else 'REGISTERED_ROW_NOT_AVAILABLE',metrics=values,checks=checks))
        output.extend(own);by_method[method]=dict(registered_cases=12,status_counts=dict(Counter(r['status'] for r in own)),
            legal_cases=sum(r['status']=='LEGAL' for r in own),nonlegal_or_missing_cases=sum(r['status']!='LEGAL' for r in own))
    measured={(r['case_key'],r['method']):r for r in output};comparisons={}
    for name,(left,right) in PAIRINGS.items():
        pairs=[];excluded=[]
        for case in cases:
            a,b=measured[(case['key'],left)],measured[(case['key'],right)]
            if a['status']!='LEGAL' or b['status']!='LEGAL':
                excluded.append(dict(case_key=case['key'],left_status=a['status'],left_reason=a['reason'],right_status=b['status'],right_reason=b['reason']));continue
            fields=sorted(set(a['metrics'])|set(b['metrics']))
            pairs.append(dict(case_key=case['key'],source_uid=case['source_uid'],source_interval=case['source_interval'],
                left=a['metrics'],right=b['metrics'],differences={k:a['metrics'][k]-b['metrics'][k]
                if a['metrics'].get(k) is not None and b['metrics'].get(k) is not None else None for k in fields}))
        comparisons[name]=dict(left=left,right=right,registered_cases=12,common_legal_cases=len(pairs),excluded_cases=excluded,raw_pairs=pairs,
            metrics={field:balanced(pairs,field) for field in sorted({k for row in pairs for k in row['differences']})},
            direction='LEFT_MINUS_RIGHT; LOWER_DYNAMIC_MAGNITUDE_IS_NOT_A_QUALITY_WIN')
    return dict(schema='R68_PUBLISHED_COMMON_CASE_ATTRIBUTION_V1',inputs=inputs,registered_case_positions=12,source_intervals=8,
        methods=by_method,rows=output,comparisons=comparisons,confidence_intervals=None,human_reviewed=False,
        artistic_quality='NOT_EVALUABLE',semantic_stage_truth='UNKNOWN',recording_independence_verified=False,
        scope='EXPOSED_FIXED12_POSITIONS_EIGHT_SOURCE_INTERVALS_DESCRIPTIVE_ONLY_NO_GENERALIZATION_OR_HYPOTHESIS_WIN',
        cross_phase_note='A_AND_B_USE_THEIR_OWN_SAME_MASK_BASELINES; EVENT_POSITION_GEOMETRY_AND_ACTUAL_DOSE_STILL_DIFFER',
        whole_p95_note='WHOLE_P95_DIFFERENCE_IS_NOT_P95_OF_TEMPLATEWISE_MAGNITUDE_DIFFERENCES',can_route=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out)
    if out.exists():raise ValueError('Fresh attribution output required')
    result=analyze(args.release);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream:stream.write((json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf8'))
    print(out)


if __name__=='__main__':main()
