"""Bounded A/B/C training on sealed clocks; final weights, no Test or Val selection."""
import argparse,json,subprocess,time,hashlib,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dunhuang.io import checked,sha,write
from .clock_data import prepare,encode,metadata,read,ref
from .clock_metrics import energy_scores,clock_report,retrieval,grouped,correlation
from .transport import ClockNullTransport,position_contrastive,SCHEMA
from .model import RetrievalModel
from .sampling import group_schedule


def digest(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def paired_scores(head,mode,bank,query,candidate,all_shifts=True):
    m=bank['mt'][query];mv=bank['mv'][query]
    d=bank['dt'][candidate];dv=bank['dv'][candidate]
    if not all_shifts:d=d[:,2:3];dv=dv[:,2:3]
    times=d.shape[1];n=len(query)
    m=m[:,None].expand(-1,times,-1,-1).flatten(0,1);mv=mv[:,None].expand(-1,times,-1).flatten(0,1)
    d=d.flatten(0,1);dv=dv.flatten(0,1)
    if mode=='A':
        x=torch.cat((m,d,m-d,m*d),-1);valid=mv&dv
        value=(head(x).squeeze(-1)*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        return value.reshape(n,times),None
    clock=torch.arange(m.shape[1],device=m.device)/15
    result=head(m,d,mv,dv,clock,clock)
    return result['score'].reshape(n,times),result


def evaluate(head,mode,bank,ids,rows,wrong):
    scores=[];other=[];diagnostics=[]
    with torch.no_grad():
        for start in range(0,len(ids),24):
            ix=ids[start:start+24]
            values,result=paired_scores(head,mode,bank,ix,ix)
            score,_=paired_scores(head,mode,bank,ix,[wrong[i] for i in ix],False)
            scores.append(values.cpu().numpy());other.append(score[:,0].cpu().numpy())
            if result is not None:
                select=torch.arange(len(ix),device=bank['mt'].device)*5+2
                clock=torch.arange(60,device=select.device)/15
                p=result['plan'][select];distance=(clock[:,None]-clock[None,:]).abs()
                for j in range(len(ix)):
                    mass=p[j].sum()
                    diagnostics.append(dict(coverage=float(result['coverage'][select[j]]),
                        music_null=float(result['music_null_fraction'][select[j]]),
                        motion_null=float(result['motion_null_fraction'][select[j]]),
                        marginal_residual=float(result['marginal_residual'][select[j]]),
                        monotonic_violations=int(result['monotonic_violations'][select[j]]),
                        clock_consistency_abs_seconds=float((p[j]*distance).sum()/mass) if mass>1e-6 else None))
    values=np.concatenate(scores);wrong_values=np.concatenate(other)
    selected=[rows[i] for i in ids];report=clock_report(values,selected,wrong_values)
    # Same-song clock retrieval. Inputs contain candidate-relative clocks only.
    matrix=np.full((len(ids),len(ids)),-1e4,np.float32);pairs=[]
    for i,q in enumerate(ids):
        for j,c in enumerate(ids):
            if rows[q]['music_uid']==rows[c]['music_uid']:pairs.append((i,j,q,c))
    with torch.no_grad():
        for start in range(0,len(pairs),64):
            part=pairs[start:start+64]
            s,_=paired_scores(head,mode,bank,[x[2] for x in part],[x[3] for x in part],False)
            for pair,value in zip(part,s[:,0].cpu().numpy()):matrix[pair[0],pair[1]]=value
    report['same_song_clock_retrieval']=retrieval(matrix,selected,'local')
    if diagnostics:
        report['transport']={key:dict(mean=float(np.mean([r[key] for r in diagnostics if r[key] is not None])),
            min=float(min(r[key] for r in diagnostics if r[key] is not None)),
            max=float(max(r[key] for r in diagnostics if r[key] is not None))) for key in diagnostics[0]}
        report['transport']['clock_localisation_scope']='band-prior-biased clock consistency, not independently labelled action-peak localisation'
    return report,dict(scores=values,wrong_scores=wrong_values,local_retrieval=matrix,diagnostics=diagnostics)


def run(assets,global_ref,sync_ref,config,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise ValueError('Registered experiment requires CUDA')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','diff','HEAD','--'],text=True):raise ValueError('Dirty tracked code')
    protocol=dict(schema='clock_partial_alignment_experiment_v1',code_commit=commit,
        config=config,global_checkpoint=global_ref,sync_checkpoint=sync_ref,
        paired_manifest=assets['paired_generation_manifest'],test_evaluations=0,
        selection='fixed last step; no Val model selection or additional search',
        time_map_schema='event_time_warp_motion74_v1',operational_qualified=False)
    write(out/'protocol.json',protocol)
    rows,arrays=prepare(assets,global_ref,sync_ref,out/'data')
    base=RetrievalModel().cuda().eval()
    payload=torch.load(checked(global_ref),map_location='cuda',weights_only=True);base.load_state_dict(payload['state'])
    frozen_before=digest(base)
    raw_bank=encode(base,arrays,out/'encodings.npz','cuda')
    bank={k:torch.from_numpy(v).cuda() for k,v in raw_bank.items() if k!='tcn'}
    train=[i for i,r in enumerate(rows) if r['role']=='train'];val=[i for i,r in enumerate(rows) if r['role']=='val']
    # Harder unrelated-pair controls are fixed by a simple energy proxy before neural scoring.
    em=arrays['energy_music'][:,::2];ed=arrays['energy_motion'][:,15:135:2]
    wrong={}
    for i in val:
        pool=[j for j in val if rows[j]['source_group']!=rows[i]['source_group'] and rows[j]['music_uid']!=rows[i]['music_uid']
            and rows[j]['choreography_uid']!=rows[i]['choreography_uid'] and rows[j]['start_frame']==rows[i]['start_frame']]
        if not pool:raise ValueError('No registered same-position unrelated control')
        values=correlation(em[i],ed[pool]);wrong[i]=pool[int(np.argmax(values))]
    write(out/'wrong_controls.json',dict(pairs=wrong,definition='same source start; distinct music, choreography, component; highest energy correlation'))
    energy=energy_scores(arrays)
    baseline=clock_report(energy[val],[rows[i] for i in val],np.array([correlation(em[i],ed[wrong[i]]) for i in val]))
    global_scores=(bank['mg'][val]@bank['dg'][val].T).cpu().numpy()
    write(out/'baselines.json',dict(energy=baseline,frozen_global=retrieval(global_scores,[rows[i] for i in val],'song')))
    groups={}
    for i in train:groups.setdefault(rows[i]['source_group'],[]).append(i)
    schedule=group_schedule(rows,config['steps'],config['batch'],config['seed'])
    np.save(out/'training_schedule.npy',np.asarray(schedule))
    results={};initializations={}
    for mode in ('A','B','C'):
        folder=out/mode;folder.mkdir()
        torch.manual_seed(config['seed'])
        if mode=='A':
            head=copy.deepcopy(base.sync)
            old=torch.load(checked(sync_ref),map_location='cuda',weights_only=True)
            head.load_state_dict({k.removeprefix('sync.'):v for k,v in old['state'].items() if k.startswith('sync.')})
        else:head=ClockNullTransport(band_seconds=config['band_seconds'],epsilon=config['epsilon'],iterations=config['sinkhorn_iterations']).cuda()
        initializations[mode]=dict(digest=digest(head),parameters=sum(p.numel() for p in head.parameters()))
        optimizer=torch.optim.AdamW(head.parameters(),lr=config['learning_rate']);history=[]
        for step,ids in enumerate(schedule,1):
            scores,result=paired_scores(head,mode,bank,ids,ids)
            loss=torch.nn.functional.cross_entropy(scores/config['score_temperature'],torch.full((len(ids),),2,device='cuda',dtype=torch.long))
            if result is not None:loss=loss+config['monotonic_weight']*result['monotonic_penalty'].mean()
            if mode=='C':
                take=torch.arange(len(ids),device='cuda')*5+2
                aligned={k:(v[take] if isinstance(v,torch.Tensor) and len(v)==len(ids)*5 else v) for k,v in result.items()}
                loss=loss+config['position_weight']*position_contrastive(aligned,[rows[i]['source_group'] for i in ids])
            if not torch.isfinite(loss):raise ValueError('Nonfinite alignment loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            if step%100==0 or step==config['steps']:
                history.append(dict(step=step,loss=float(loss.detach())))
                write(folder/'history.json',history);print('CLOCK_TRAIN',mode,step,float(loss.detach()),flush=True)
        checkpoint=dict(schema=SCHEMA,mode=mode,state=head.state_dict(),config=config,
            global_checkpoint_sha256=global_ref['sha256'],norm=payload['norm'],
            protocol_sha256=sha(out/'protocol.json'),step=config['steps'],
            qualified=False,operational_qualified=False,execution_time_map_schema=protocol['time_map_schema'])
        torch.save(checkpoint,folder/'head.pt')
        head.eval();metrics,measurements=evaluate(head,mode,bank,val,rows,wrong)
        np.savez_compressed(folder/'measurements.npz',**{k:v for k,v in measurements.items() if k!='diagnostics'})
        write(folder/'transport_diagnostics.json',measurements['diagnostics'])
        write(folder/'validation.json',metrics);results[mode]=metrics
        print('CLOCK_EVALUATED',mode,flush=True)
    assert initializations['B']==initializations['C']
    assert digest(base)==frozen_before
    # Fixed pre-pooling phase diagnostic. No pooling replacement is selected.
    tcn=raw_bank['tcn'][val];stride=tcn[:,::2];odd=tcn[:,1::2]
    average=(stride+odd)/2
    anti=np.pad(tcn,((0,0),(1,1),(0,0)),mode='edge')
    anti=(anti[:,:-2]+2*anti[:,1:-1]+anti[:,2:])/4
    diagnostics=dict(stride_even_odd_relative_rms=float(np.sqrt(np.mean((stride-odd)**2))/max(np.sqrt(np.mean(tcn**2)),1e-12)),
        two_frame_average_even_relative_rms=float(np.sqrt(np.mean((average-stride)**2))),
        antialias_even_odd_relative_rms=float(np.sqrt(np.mean((anti[:,::2]-anti[:,1::2])**2))/max(np.sqrt(np.mean(anti**2)),1e-12)),
        pooling_replaced=False,learned_attention_added=False,scope='observed one-frame phase sensitivity; not proof of semantic peak preservation')
    write(out/'phase_diagnostic.json',diagnostics)
    write(out/'initializations.json',initializations)
    write(out/'result.json',dict(status='COMPLETED_CLOCK_ALIGNMENT_SHADOW',results=results,baselines=read(out/'baselines.json'),
        phase_diagnostic=diagnostics,rows=len(rows),train_rows=len(train),validation_rows=len(val),
        train_components=len(groups),validation_components=len({rows[i]['source_group'] for i in val}),
        initializations=initializations,encoder_digest_before=frozen_before,encoder_digest_after=digest(base),
        training_steps_per_arm=config['steps'],test_evaluations=0,operational_qualified=False,
        qualification_blockers=['NO_NEW_BLIND_EVALUATION','NO_INDEPENDENT_ACTION_PEAK_LABELS','ONLY_TWO_VAL_COMPONENTS'],
        seconds=time.monotonic()-start))
    print('CLOCK_ALIGNMENT_COMPLETED',flush=True)


def main():
    ap=argparse.ArgumentParser()
    for key in ('assets','global-checkpoint','sync-checkpoint','config','out'):ap.add_argument('--'+key,required=True)
    a=ap.parse_args();run(read(a.assets),ref(a.global_checkpoint),ref(a.sync_checkpoint),read(a.config),a.out)
if __name__=='__main__':main()
