"""Native-N0/N1/N2 research. No training, generation, threshold activation or editing."""
import argparse,json,sys,subprocess,time,traceback,shutil,hashlib,os
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from dunhuang.io import checked,sha,write
from dunhuang.contracts import motion151
from .native_shadow import LegacyBackend
from .protocols import GroundAndSkeletonContract,digest
from .contact_evidence import load_evidence,conditioned_metrics
from .calibration import reference_report
from .labelled_calibration import load_quality,candidates

def read(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def array_hash(a):return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def input_population(assets,training,prior):
    protocol=read(training/'protocol.json');status=read(training/'status.json')
    if protocol['paired_manifest']!=assets['paired_generation_manifest'] or sha(training/'protocol.json')!=status['protocol_sha256']:
        raise ValueError('Frozen input lineage changed')
    manifest=read(checked(assets['paired_generation_manifest']))
    tasks=read(training/'source_pairs.json')
    from dunhuang.generation.source_pairs import source_pairs
    if tasks!=source_pairs(manifest):raise ValueError('Source task changed')
    selected={p['target_uid'] for p in tasks if p['role']=='train'}
    rows=sorted([r for r in manifest['rows'] if r['window_uid'] in selected],key=lambda r:r['window_uid'])
    if len(rows)!=1468 or any(r['role']!='train' for r in rows):raise ValueError('Registered Train population changed')
    old=prior/'target_audits.jsonl';summary=read(prior/'summary.json')
    if sha(old)!=summary['target_audits_sha256']:raise ValueError('Historical target receipts changed')
    previous={r['window_uid']:r for r in map(json.loads,old.read_text().splitlines())}
    if set(previous)!=selected:raise ValueError('Historical population differs')
    return rows,previous


def generated_pool(experiment,baseline):
    pool=[]
    for name,folder in [('paired300',baseline),('cross',experiment/'cross/mechanism'),('additive',experiment/'additive/mechanism')]:
        for c in read(folder/'generation.json')['cases']:
            path=Path(c['motion']['path']).parent;checked(c['motion']);checked(c['fk'])
            pool.append(dict(name=name+'_'+c['case'],mode=name,domain='aist_historical_val',folder=path,
                             anchor=baseline/('s'+str(c['seed'])+'_m'+str(c['music_index'])+'_e-1')))
    project=Path.cwd()
    for mode in ('cross','additive'):
        for condition in ('original','replaced'):
            pool.append(dict(name=mode+'_dunhuang_'+condition,mode=mode,domain='dunhuang_unpaired',
                folder=experiment/mode/('dunhuang_'+condition),anchor=project/'outputs/v06_01/official150'))
    for mode,name in [('edge','official150'),('temporal','temporal150')]:
        pool.append(dict(name='historical_'+mode,mode=mode,domain='A65_historical',
                         folder=project/'outputs/v06_01'/name,anchor=project/'outputs/v06_01/official150'))
    return pool


def run_train(backend,rows,previous,out,config,contact_refs=None):
    out.mkdir();anchors={};records=[];counts=Counter();ownership=Counter();record_fractions=defaultdict(list);labelled=0
    # Freeze each recording's first admitted target as reference, before evaluating its other windows.
    for r in sorted(rows,key=lambda r:(r['recording_uid'],r['start_frame'])):
        if r['recording_uid'] not in anchors:
            x=np.load(checked(r['motion']));fk=backend.fk(x)
            floor=float(np.percentile(fk[:,[7,8,10,11],1],5))
            anchors[r['recording_uid']]=dict(floor_y_m=floor,reference=r['motion'])
    write(out/'frozen_ground_anchors.json',anchors)
    with (out/'windows.jsonl').open('w',encoding='utf8') as stream:
        for i,r in enumerate(rows,1):
            path=checked(r['motion']);x=np.load(path);motion_sha=r['motion']['sha256']
            evidence=None if contact_refs is None else contact_refs.get(motion_sha)
            labels,label_info=load_evidence(evidence,motion_sha,len(x));labelled+=int(labels is not None)
            a=anchors[r['recording_uid']]
            value,fk=backend.analyse(x,motion_sha,a['floor_y_m'],a['reference']['sha256'],labels=labels,
                                    expected=previous[r['window_uid']]['physical'])
            lineage={k:r[k] for k in ('window_uid','role','recording_uid','choreography_uid','source_group','music_uid','start_frame')}
            record=dict(lineage,motion=r['motion'],legacy=value['legacy'],legacy_contact=value['legacy_contact'],
                        layers=value['layers'],labels=evidence)
            stream.write(json.dumps(record,allow_nan=False)+chr(10))
            metrics={spec['metric']:value['legacy_audit'].get(spec['metric']) for spec in backend.registry['metrics']}
            if not value['legacy_contact']['sample_count']:
                for key in ('foot_skate_mps_p95','foot_skate_mps_max','foot_support_drift_m_p95','foot_support_drift_m_max'):metrics[key]=None
            contact=value['layers']['continuous']['contact'];candidate_metrics=dict(metrics)
            # Foot proposals must use independent static labels, not the speed-conditioned Legacy proxy.
            for key,field,stat in [('foot_skate_mps_p95','independent_static_speed','p95'),
                                   ('foot_skate_mps_max','independent_static_speed','max'),
                                   ('foot_support_drift_m_p95','valid_static_drift','p95'),
                                   ('foot_support_drift_m_max','valid_static_drift','max')]:
                candidate_metrics[key]=contact[field][stat]
            for key in ('foot_contact_height_m_max','foot_penetration_min_m'):candidate_metrics[key]=None
            records.append(dict(lineage,metrics=metrics,candidate_metrics=candidate_metrics))
            counts.update(value['legacy']['reasons']);ownership.update(value['legacy_contact']['exceedance_ownership'])
            n=value['legacy_contact']['exceeding_frame_foot_observations']
            record_fractions[r['recording_uid']].append(value['legacy_contact']['exceedance_ownership']['height_only']/n if n else 0.)
            if i%100==0 or i==len(rows):
                stream.flush();write(out/'progress.json',dict(completed=i,total=len(rows),legacy_replays_exact=i));print('NATIVE_TRAIN',i,flush=True)
    summary=dict(windows=len(rows),recordings=len(anchors),choreographies=len({r['choreography_uid'] for r in rows}),
        connected_source_components=len({r['source_group'] for r in rows}),legacy_passes=sum(previous[r['window_uid']]['physical']['ok'] for r in rows),
        legacy_replay_exact=True,reason_counts=dict(counts),exceeding_frame_foot_ownership=dict(ownership),
        ownership_population='overlapping window-frame-foot observations, not independent source frames',
        recording_balanced_height_only_exceedance_fraction=float(np.mean([np.mean(v) for v in record_fractions.values()])),
        independence_warning='cluster summaries do not prove statistical independence',independent_real_contact_labels=labelled)
    write(out/'summary.json',summary);write(out/'calibration_rows.json',records)
    return records,anchors


def replay_generated(backend,pool,out,assets):
    out.mkdir();results=[];anchors={};before={}
    for c in pool:
        folder=c['folder'];anchor=c['anchor']
        if str(anchor) not in anchors:
            x=np.load(anchor/'native.npy');fk=np.load(anchor/'native_fk.npy')
            anchors[str(anchor)]=dict(value=float(np.percentile(fk[:,[7,8,10,11],1],5)),motion=ref(anchor/'native.npy'))
    write(out/'ground_anchors.json',anchors)
    for i,c in enumerate(pool):
        source=c['folder'];dest=out/c['name'];dest.mkdir()
        for name in ('edge.npy','official_fk.npy','native.npy','native_fk.npy','physical.json'):
            before[str(source/name)]=sha(source/name)
        for name in ('edge.npy','official_fk.npy'):shutil.copyfile(source/name,dest/name)
        q=dict(assets,frames=len(np.load(source/'native.npy')),output=str(dest),anchors=[],beat_frames=[])
        write(dest/'request.json',q)
        with (dest/'native.log').open('w') as log:
            child_env=dict(os.environ)
            package_root=str(Path(__file__).resolve().parents[2])
            child_env['PYTHONPATH']=package_root+os.pathsep+child_env.get('PYTHONPATH','')
            subprocess.run([assets['native_python'],'-m','dunhuang.physical',str(dest/'request.json')],stdout=log,stderr=subprocess.STDOUT,check=True,env=child_env)
        if read(dest/'physical.json')!=read(source/'physical.json') or sha(dest/'native.npy')!=sha(source/'native.npy') or sha(dest/'native_fk.npy')!=sha(source/'native_fk.npy'):
            raise ValueError('Historical full Legacy replay changed: '+c['name'])
        x=np.load(source/'native.npy');fk=np.load(source/'native_fk.npy');a=anchors[str(c['anchor'])]
        value,_=backend.analyse(x,sha(source/'native.npy'),a['value'],a['motion']['sha256'],fk=fk,
                               expected=read(source/'physical.json')['absolute'])
        result=dict(name=c['name'],mode=c['mode'],domain=c['domain'],motion=ref(source/'native.npy'),
                    fk=ref(source/'native_fk.npy'),anchor=a,analysis=value)
        results.append(result);write(dest/'layers.json',value['layers'])
        write(out/'progress.json',dict(completed=i+1,total=len(pool)))
    for name,h in before.items():
        if sha(name)!=h:raise ValueError('Evaluation mutated historical input')
    write(out/'results.json',results)
    return results



def historical_inventory(backend,experiment_out,out):
    records=[];unique={};skipped=[]
    for path in sorted(Path('outputs').rglob('physical.json')):
        folder=path.parent.resolve()
        if folder.is_relative_to(experiment_out.resolve()):continue
        if not all((folder/n).is_file() for n in ('native.npy','native_fk.npy')):continue
        doc=read(path)
        if (doc.get('legacy_commit')!=backend.identity['legacy_commit'] or doc.get('limits')!=backend.limits or
                doc.get('relative_parent_caps') is not False or
                doc.get('audit',{}).get('schema')!=backend.identity['metric_schema']):
            skipped.append(dict(path=str(path),reason='DIFFERENT_OR_UNSEALED_LEGACY_CONTRACT'));continue
        refs={n:ref(folder/n) for n in ('native.npy','native_fk.npy','physical.json')}
        key=digest(dict(motion=refs['native.npy']['sha256'],fk=refs['native_fk.npy']['sha256'],
                        audit=doc['audit'],gate=doc['absolute']))
        records.append(dict(files=refs,identity=key))
        unique.setdefault(key,(folder,doc))
    result=[]
    for key,(folder,doc) in unique.items():
        x=np.load(folder/'native.npy');fk=np.load(folder/'native_fk.npy')
        audit=backend.audit(x,backend.cfg,precomputed_joints=fk);gate=backend.gate(audit)
        if audit!=doc['audit'] or gate!=doc['absolute']:
            raise ValueError('Matching-contract historical audit changed: '+str(folder))
        result.append(dict(identity=key,legacy_replay_exact=True,frames=len(x)))
    for record in records:
        for item in record['files'].values():checked(item)
    write(out,dict(status='ALL_MATCHING_LEGACY_HISTORY_REPLAYED',records=records,
        unique_measurements=result,record_count=len(records),unique_count=len(result),
        skipped=skipped,legacy_identity=backend.identity['id'],inputs_unchanged=True,
        old_edge_arrays_not_reinterpreted=True,new_generations=0,optimizer_calls=0))
    return len(records),len(result)

def annotation_packet(selected,source_ledger,out):
    out.mkdir();ledger={r.get('recording_uid'):r for r in source_ledger if r.get('status')=='VERIFIED_PAIR'}
    entries=[]
    for i,r in enumerate(selected):
        states=out/(str(i)+'_states.npy');np.save(states,np.full((150,4),-1,np.int8))
        original=ledger[r['recording_uid']]
        entry=dict(schema='independent_contact_labels_v1',validated=False,motion_sha256=r['motion']['sha256'],
            frames=150,fps=30,foot_joint_indices=[7,8,10,11],source='manual_video_annotation',
            independent_of_evaluation_metric=True,reviewer=None,labels=ref(states),
            original_video=original['video'],official_video_url=original['video_url'],
            video_start_seconds=r['start_frame']/30,window_uid=r['window_uid'],
            instruction='Review independently; UNKNOWN=-1, swing=0, static=1, intentional_slide=2. Do not label from metric speed or Legacy outcome.')
        write(out/(str(i)+'_evidence.json'),entry);entries.append(entry)
    write(out/'index.json',dict(status='UNREVIEWED_ANNOTATION_PACKET',records=entries,validated=False,
        quality_labels='not provided; no false-reject estimate or released thresholds'))
    return len(entries)


def main():
    ap=argparse.ArgumentParser()
    for key in ('assets','benchmark-assets','training','prior-target-audit','prior-experiment','baseline-mechanism','source-ledger','out'):ap.add_argument('--'+key,required=True)
    ap.add_argument('--config',default='configs/native_calibration_v2.json')
    a=ap.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    assets=read(a.assets);mesh_assets=read(a.benchmark_assets);config=read(a.config)
    if config['mode']!='SHADOW' or config['operational_qualified'] or not config['legacy_gate_unchanged']:
        raise ValueError('Only unqualified Shadow evaluation is authorized by this protocol')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','diff','HEAD','--'],text=True):raise ValueError('Dirty tracked code')
    write(out/'protocol.json',dict(schema='native_reliability_experiment_v2',code_commit=commit,
        assets=ref(a.assets),mesh_assets=ref(a.benchmark_assets),config=ref(a.config),source_ledger=ref(a.source_ledger),
        prior_training_protocol=ref(Path(a.training)/'protocol.json'),training_steps=0,optimizer_calls=0,
        new_generations=0,legacy_gate_unchanged=True,mode='SHADOW',
        missing_evidence_is_unknown=True))
    try:
        backend=LegacyBackend(assets)
        if backend.registry!=config['legacy_metric_provenance']:raise ValueError('Registered Legacy limits/definitions changed')
        write(out/'legacy_identity.json',backend.identity);write(out/'threshold_provenance.json',backend.registry)
        historical_counts=historical_inventory(backend,out,out/'historical_inventory.json')
        rows,previous=input_population(assets,Path(a.training),Path(a.prior_target_audit))
        contact_refs=config['external_evidence']['contact_labels']
        if contact_refs is not None:contact_refs=read(checked(contact_refs))['by_motion_sha256']
        records,anchors=run_train(backend,rows,previous,out/'train',config,contact_refs)
        bootstrap=config['bootstrap']
        write(out/'calibration.json',reference_report(records,backend.registry,bootstrap['iterations'],bootstrap['seed']))
        quality=load_quality(config['external_evidence']['quality_labels'],{r['window_uid']:r for r in rows})
        write(out/'labelled_candidates.json',candidates(records,backend.registry,quality,bootstrap['iterations'],bootstrap['seed']))
        pool=generated_pool(Path(a.prior_experiment),Path(a.baseline_mechanism))
        generated=replay_generated(backend,pool,out/'generated',assets)
        selected=[]
        for group in sorted({r['source_group'] for r in rows}):
            seen=set()
            for row in sorted(rows,key=lambda r:(r['recording_uid'],r['start_frame'])):
                if row['source_group']==group and row['recording_uid'] not in seen:
                    selected.append(row);seen.add(row['recording_uid'])
                    if len(seen)==config['sensitivity']['recordings_per_source_component']:break
        annotation_packet(selected,read(a.source_ledger),out/'annotation_review')
        from .native_sensitivity import sensitivity
        sensitivity(backend,selected,generated,mesh_assets,config,out/'sensitivity')
        from .native_perturbations import controlled
        controlled(backend,mesh_assets,config,out/'controlled')
        write(out/'status.json',dict(status='COMPLETED_NATIVE_RELIABILITY_SHADOW_AUDIT',
            train_windows=len(rows),recordings=len(anchors),generated_replays=len(generated),historical_inventory_counts=historical_counts,
            legacy_results_unchanged=True,operational_qualified=False,mode='SHADOW',
            training_steps=0,optimizer_calls=0,new_generations=0,seconds=time.monotonic()-start))
        print('NATIVE_RELIABILITY_COMPLETED',flush=True)
    except Exception:
        write(out/'status.json',dict(status='FAILED',error=traceback.format_exc(),legacy_gate_unchanged=True))
        raise

if __name__=='__main__':main()
