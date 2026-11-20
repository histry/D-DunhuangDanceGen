"""Export R68 sealed scalar evidence, never raw motion or viewing payloads.

Run from the published repository with a fresh --out. Stage commits may differ;
each independent receipt is checked against its own original manifest.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

HEX = re.compile(r'^[0-9a-f]{64}$')
FORBIDDEN = {
    'q','q0','q_injected','tau','time_warp','source_positions','source_grid','target_grid',
    'curve','curves','right_endpoint_frames','half_open_frame_ranges','edited_frames',
    'r','baseline_r','reference_r','bridge_r','full_r','p','j','parents','rotations',
    'coordinates','coords','positions','tokens','music_tokens','motion_tokens',
    'boundary_jets','actual_boundary_jets','boundary_proof','jets','b_matrix',
    'event_mask','edit_mask','support_mask','masks','mask','waveform','samples',
    'audio_samples','audio_bytes','beat_frames','accent_frames','accent_strength',
    'raw_audio','raw_path','audio_path','password','credential','credentials','token',
    'private_key','reviewer_roster_ref','source_proofs','graph','anchors',
}
INTERVALS = {
    'source_interval','target_interval','audio_interval','pair_music_interval',
    'gap_interval_local','evidence_span','target_evidence_interval','target_focus_interval',
    'source_support','target_support_interval','bump_support_interval','support_interval',
    'spectral_band_hz','source_to_target_rates','source_to_target_rate_limits',
    'source_to_target_rate_bounds','inverse_rate_limits','ci95',
}
SCALAR_LISTS = {
    'methods','preview_methods','control_methods','event_types','types','deltas',
    'delta_frames','target_deltas_frames','deltas_frames','rate_limits_dt_ds','offsets','differences',
    'source_uids','required_stages','owned_tmux_sessions_remaining',
    'actual_boundary_frame_indices','available_boundary_frame_indices',
    'g1_actual_boundary_read_indices','g0_actual_boundary_read_indices',
}
METRICS = {
    'FK_D1_mean_mps':('metrics','whole','actual','FK','D1','mean'),
    'FK_D2_p95_mps2':('metrics','whole','actual','FK','D2','p95'),
    'FK_D3_p95_mps3':('metrics','whole','actual','FK','D3','p95'),
    'FK_speed_ratio':('metrics','activity_normalized_oscillation','FK_mean_speed_ratio'),
    'SO3_speed_ratio':('metrics','activity_normalized_oscillation','SO3_mean_speed_ratio'),
    'FK_high_band_fraction':('metrics','activity_normalized_oscillation','actual','FK_velocity_6Hz_to_Nyquist_power_fraction'),
    'max_source_clock_change_frames':('metrics','dose','maximum_abs_source_position_change_frames'),
    'G1_gap_D3_mps3':('connection','metrics','gap_D3'),
    'G1_boundary_D3_mps3':('connection','metrics','boundary_D3'),
}


def read(path):
    return json.loads(Path(path).read_bytes().decode('utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1048576),b''):h.update(value)
    return h.hexdigest()


def write(path,value):
    Path(path).write_bytes((json.dumps(value,ensure_ascii=False,sort_keys=True,
                                      indent=2,allow_nan=False)+'\n').encode('utf8'))


def scalar(value):
    return value is None or isinstance(value,(str,bool,int)) or isinstance(value,float) and math.isfinite(value)


def omitted(key):
    name=str(key).lower()
    return name in FORBIDDEN or name.endswith(('_curve','_curves','_matrix','_matrices'))


def project(value,key=''):
    """Keep scalar evidence, named small metadata lists and structured records.

    Unnamed numeric sequences/matrices are omitted, not recast as measurements.
    Missing records and explicit failure status/reason remain structured rows.
    """
    if omitted(key):return None
    if scalar(value):return value
    if isinstance(value,dict):
        output={}
        for name,item in value.items():
            if omitted(name):continue
            if isinstance(item,list):
                if all(isinstance(v,dict) for v in item):
                    output[name]=[project(v) for v in item]
                elif str(name).lower() in INTERVALS and (len(item)==2 or not item) and all(scalar(v) for v in item):
                    output[name]=list(item)
                elif str(name).lower() in SCALAR_LISTS and all(scalar(v) for v in item):
                    output[name]=list(item)
                # Signal lists have no publication contract and stay private.
            elif isinstance(item,dict):output[name]=project(item,name)
            elif scalar(item):output[name]=item
            else:raise ValueError('Nonfinite or unsupported scalar at '+str(name))
        return output
    if isinstance(value,list) and all(isinstance(v,dict) for v in value):
        return [project(v) for v in value]
    raise ValueError('Uncontracted top-level signal vector or nonfinite scalar')


def hash_manifest(stage):
    stage=Path(stage).resolve();path=stage/'manifest.json';doc=read(path)
    if not isinstance(doc.get('files'),dict) or not isinstance(doc.get('binding'),dict):
        raise ValueError('A sealed R68 file map and source binding are required')
    files=[];seen=set()
    for raw,expected in sorted(doc['files'].items()):
        name=str(raw).replace('\\','/');relative=Path(name)
        if (not name or name.startswith('/') or ':' in name or relative.is_absolute() or '..' in relative.parts
                or not isinstance(expected,str) or not HEX.fullmatch(expected)):
            raise ValueError('Unsafe, duplicated or invalid manifest member')
        name=relative.as_posix()
        if name in seen:raise ValueError('Unsafe, duplicated or invalid manifest member')
        seen.add(name);member=(stage/relative).resolve()
        if (not member.is_relative_to(stage) or not member.is_file() or sha(member)!=expected):
            raise ValueError('Frozen scientific file changed: '+name)
        files.append(dict(path=name,bytes=member.stat().st_size,sha256=expected))
    for name in ('records.json','population.json','registration.json'):
        if name not in seen:raise ValueError('Required science file is outside the seal: '+name)
    return dict(status='ALL_MANIFEST_BYTES_REHASHED_EXACT',manifest_sha256=sha(path),
                binding=project(doc['binding']),files=files,raw_artifact_bytes_copied=False)


def replay(path,manifest):
    path=Path(path);manifest=Path(manifest)
    if not path.is_file():return dict(status='NOT_AVAILABLE',scientific_outputs_exact=None)
    doc=read(path);result=project(doc)
    matched=manifest.is_file() and doc.get('source_manifest_sha256')==sha(manifest)
    exact=doc.get('scientific_outputs_exact') is True and doc.get('differences')==[] and matched
    result.update(status='EXACT' if exact else 'NOT_EXACT_OR_UNBOUND',source_manifest_binding_exact=matched,
                  receipt_sha256=sha(path))
    return result


def lookup(row,path):
    for key in path:
        if not isinstance(row,dict) or key not in row:return None
        row=row[key]
    return float(row) if isinstance(row,(int,float)) and not isinstance(row,bool) and math.isfinite(row) else None


def aggregate(records,population):
    cases=population.get('cases',[]) if isinstance(population,dict) else population
    identities={str(c.get('key',c.get('case_key'))):c for c in cases}
    by_method=defaultdict(list)
    for row in records:by_method[str(row.get('method','UNKNOWN_METHOD'))].append(row)
    methods={}
    for name,rows in sorted(by_method.items()):
        metrics={}
        for label,path in METRICS.items():
            values=[];source=defaultdict(list)
            for row in rows:
                value=lookup(row,path)
                if value is None:continue
                values.append(value)
                identity=identities.get(str(row.get('case_key')),row)
                uid=identity.get('source_uid');span=identity.get('source_interval')
                if isinstance(uid,str) and isinstance(span,list) and len(span)==2:
                    source[(uid,tuple(span))].append(value)
            source_means=[sum(v)/len(v) for v in source.values()]
            metrics[label]=dict(registered_records=len(rows),evaluable_records=len(values),
                missing_records=len(rows)-len(values),position_weighted_mean=sum(values)/len(values) if values else None,
                minimum=min(values) if values else None,maximum=max(values) if values else None,
                evaluable_source_intervals=len(source),
                source_interval_balanced_mean=sum(source_means)/len(source_means) if source_means else None,
                ci95=None)
        connection_counts=Counter('NOT_EVALUATED' if not isinstance(r.get('connection'),dict)
            or r['connection'].get('passes_original_screen') is None else str(r['connection']['passes_original_screen']) for r in rows)
        methods[name]=dict(registered_records=len(rows),status_counts=dict(Counter(str(r.get('status','UNKNOWN')) for r in rows)),
            failure_reason_counts=dict(Counter(str(r['reason']) for r in rows if r.get('reason') is not None)),
            records_with_actual_metrics=sum(isinstance(r.get('metrics'),dict) and bool(r['metrics']) for r in rows),
            G1_original_screen_status_counts=dict(connection_counts),metrics=metrics)
    return dict(registered_records=len(records),registered_cases=len(cases),
        case_keys_with_records=len({r.get('case_key') for r in records}),
        status_counts=dict(Counter(str(r.get('status','UNKNOWN')) for r in records)),methods=methods,
        missing_and_infeasible_not_zero_filled=True,
        scope='DESCRIPTIVE_ALL_REGISTERED_METHOD_ROWS_WITH_EXPLICIT_DENOMINATORS_NO_BEST_SELECTION',
        repeated_music_positions_are_not_independent_source_recordings=True,
        quality_winner=None,can_route=False,artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE')


def export(repo='.',out='reports/r68_release',root='outputs/local_timing_r68_01',
           execution='outputs/local_timing_r68_execution_01'):
    repo=Path(repo).resolve()
    def absolute(path):
        path=Path(path);return path.resolve() if path.is_absolute() else (repo/path).resolve()
    out,root,execution=map(absolute,(out,root,execution))
    if out.exists():raise ValueError('A fresh R68 public directory is required')
    files={};science={};states={};sources=[]
    for mode in ('preview','controls'):
        stage=root/mode
        if not stage.is_dir():states[mode]='NOT_AVAILABLE_NOT_CLAIMED_RUN';continue
        if not (stage/'manifest.json').is_file():states[mode]='INCOMPLETE_NOT_PUBLISHED_AS_FINISHED';continue
        science[mode]=hash_manifest(stage);states[mode]='SEALED'
        records=read(stage/'records.json');population=read(stage/'population.json')
        if not isinstance(records,list) or any(not isinstance(r,dict) for r in records):
            raise ValueError('Every registered record, including failures, must be a structured row')
        files[mode+'_records.json']=project(records)
        files[mode+'_population.json']=project(population)
        files[mode+'_registration.json']=project(read(stage/'registration.json'))
        files[mode+'_summary.json']=aggregate(records,population)
        if (stage/'summary.json').is_file():
            if not any(row['path']=='summary.json' for row in science[mode]['files']):
                raise ValueError('Original scientific summary is outside its stage seal')
            files[mode+'_scientific_summary.json']=project(read(stage/'summary.json'))
        files['independent_'+mode+'.json']=replay(root/('replay_'+mode)/'independent.json',stage/'manifest.json')
        sources.append(dict(path=mode+'/manifest.json',sha256=sha(stage/'manifest.json')))
    if not science:raise ValueError('No sealed R68 scientific stage is available')
    receipts=[]
    if execution.is_dir():
        for path in sorted(execution.rglob('*.json')):
            receipts.append(dict(name=path.relative_to(execution).as_posix(),sha256=sha(path),metadata=project(read(path))))
    for path in sorted((root/'receipts').glob('*.json')):
        receipts.append(dict(name='delivery/'+path.name,sha256=sha(path),metadata=project(read(path))))
    files['execution_and_delivery_receipts.json']=dict(records=receipts,
        scope='STRUCTURED_METADATA_ONLY_NO_LOG_TEXT_OR_MEDIA_PAYLOAD',missing_receipt_is_not_success=True)
    files['scientific_files.json']=dict(stages=science,raw_artifact_bytes_copied=False)
    files['qualification.json']=dict(stages=states,source_commit_by_stage={k:v['binding'].get('commit') for k,v in science.items()},
        independent_by_stage={k:files['independent_'+k+'.json']['status'] for k in science},
        human_reviewed=False,semantic_phase_truth='UNKNOWN',artistic_quality='NOT_EVALUABLE',Native24='NOT_EVALUABLE',
        can_route=False,operational_qualified=False,quality_winner=None,neural_training_steps=0,new_planner_searches=0,
        projection_scope='SCALAR_TREE_NO_Q_ROTATIONS_TOKENS_COORDINATES_JETS_OR_VIEWER_BYTES')
    files['publication_inputs.json']=dict(stage_manifests=sources,publisher_sha256=sha(Path(__file__)),
        publication_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip())
    out.mkdir(parents=True)
    for name,value in files.items():write(out/name,value)
    report=("# R68：来源局部时间干预的标量证据\n\n"
        "本发布逐文件重哈希各阶段封存产物，不复制旋转、位置、逐帧时钟、音频或观看页。"
        "失败与无事件记录保留，不选择表现最好案例，不将缺失填零。\n\n"
        +''.join(f"- {mode}：{state}。\n" for mode,state in states.items())
        +"\n每种方法的登记分母、可评分母、失败原因与动态读数见对应 summary；"
        "位置均值与来源区间均衡均值分列，重复来源与音乐位置不当作独立录制。"
        "自动事件是运动学代理；D3及频率读数均不是艺术质量判决。\n\n"
        "独立复验只有 scientific_outputs_exact=true、差异为空且源manifest SHA一致时记为EXACT。"
        "各阶段可以来自不同代码修订，缺失或未绑定回执不算通过。"
        "本轮没有人工音乐收益、完整世界Root/contact或Native24资格结论，can_route=false。\n")
    (out/'报告.md').write_bytes(report.encode('utf8'))
    write(out/'public_manifest.json',dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(out.iterdir()) if p.is_file()],self_excluded=True,scope='METADATA_SCALARS_ONLY'))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='.')
    parser.add_argument('--out',default='reports/r68_release')
    parser.add_argument('--root',default='outputs/local_timing_r68_01')
    parser.add_argument('--execution',default='outputs/local_timing_r68_execution_01')
    args=parser.parse_args()
    print(export(args.repo,args.out,args.root,args.execution))


if __name__=='__main__':main()
