"""Full SDMR framework first, then one preregistered modality removal."""

import sys

import json

import subprocess

from pathlib import Path

import numpy as np

import torch

ROOT=Path(__file__).resolve().parents[1]

sys.path.insert(0,str(ROOT/'src'))

from dunhuang.io import write,sha

from dunhuang.alignment.dataset import prepare,load

from dunhuang.alignment.train import fit

from dunhuang.alignment.evaluate import retrieval

from dunhuang.alignment.semantics import annotate



def main(out):

    out.mkdir(parents=True,exist_ok=False)

    legacy=ROOT/'third_party/danceRAG'

    assets=Path('/home/disk/lsm/storage/danceRAG/assets/datasets')

    protocol=dict(seed=20261111,global_steps=5000,sync_steps=2000,val_interval=250,actual_shift_frames=[-15,-8,0,8,15],input_fps=30,

        selection='validation MRR for global, group synchronization win for sync',

        primary='full 50D/74D TCN4 Transformer2 width128 embedding256',

        minimal_ablation='inference removal of acoustic32 after full framework, no test-based selection',

        expert_preferences='unavailable; filename semantics used only on motion side',

        historical_test='development regression, not a new blind test',

        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())

    write(out/'protocol.json',protocol)

    manifest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else prepare(ROOT/'inputs/sdmr_population.json',out/'data',legacy,assets)

    rows,arrays,norm=load(manifest)
    write(out/'data_audit.json',dict(manifest_sha256=sha(manifest),windows=len(rows),
        counts={role:{domain:sum(r['role']==role and r['domain']==domain for r in rows) for domain in ('finedance','aistpp')} for role in ('train','val','test')},
        group_counts={role:len({r['group'] for r in rows if r['role']==role}) for role in ('train','val','test')},
        normalization_role='train',global_inputs_no_identity_features=True))


    write(out/'normalization.json',norm)

    model,result=fit(rows,arrays,norm,out,protocol['global_steps'],protocol['sync_steps'])

    # The complete model is selected before any ablation is evaluated.

    test=[i for i,r in enumerate(rows) if r['role']=='test']

    ablated=dict(arrays)

    ablated['music']=arrays['music'].copy();ablated['mm']=arrays['mm'].copy()

    ablated['music'][:,:,18:]=0;ablated['mm'][:,:,18:]=False

    write(out/'minimal_ablation.json',dict(scope='inference feature removal; not retrained architecture ablation',

        full=result['retrieval_test'],without_acoustic32=retrieval(model,rows,ablated,test,'cuda')))

    sys.path.insert(0,str(legacy))

    from motion.retrieval_generation import decode_continuous_source

    from evaluation.paired_alignment_observations_v3 import motion_from_edge151

    catalog_file=Path('/home/disk/lsm/storage/danceRAG/outputs/unpaired_data_protocol_20261015_ZY1C32/catalog/motion_catalog.json')

    catalog=json.loads(catalog_file.read_text());events=[];embeddings=[];labels=[]
    smpl_root=Path('/home/disk/lsm/storage/danceRAG/assets/motion/smpl_official_14')
    write(out/'filename_labels.json',dict(records=[dict(annotation=annotate(file.stem),filename=file.name,sha256=sha(file)) for file in sorted(smpl_root.glob('*.npz'))],
        policy='user-authorized coarse filename labels, no event-level gesture or music preference inferred'))


    categories=['飞天','莲花','飘带','琵琶','冥想','三十六式','鼓舞','UNKNOWN']

    source_cache={}

    for event in catalog['events']:

        uid=event['source_uid']

        source=next(r for r in catalog['source_records'] if r['source_uid']==uid)

        if uid not in source_cache:

            raw=Path('/home/disk/lsm/storage/danceRAG/assets/motion/smpl_official_14')/Path(source['raw_receipt']['path']).name

            if sha(raw)!=source['source_sha256']:raise ValueError('ChangE source changed')

            n=source['raw_frames']

            motion,_=decode_continuous_source(raw,source['discovery_metadata'],(0,n),known_rows=np.ones(n,bool))

            values,mask,_=motion_from_edge151(motion)

            source_cache[uid]=(values,mask,motion)

        values,mask,source_motion=source_cache[uid]

        a,b=event['core_frame_start'],event['core_frame_end']

        length=min(b-a,240)

        m=(values[a:a+length]-norm['motion']['mean'])/norm['motion']['std']*mask[a:a+length]

        with torch.no_grad():

            embedding=model.motion(torch.as_tensor(m,device='cuda').float()[None],torch.as_tensor(mask[a:a+length],device='cuda')[None])[0]

        semantic=annotate(uid)

        events.append(dict(event_uid=event['event_uid'],source_uid=uid,raw_interval=[a,b],semantic=semantic,

            source_sha256=source['source_sha256'],feature_frames=length))

        embeddings.append(embedding.cpu().numpy()[0]);labels.append(categories.index(semantic['category']))

    write(out/'events.json',dict(events=events,categories=categories,catalog_sha256=sha(catalog_file)))

    x=torch.as_tensor(np.stack(embeddings),device='cuda');y=torch.as_tensor(labels,device='cuda')

    # Filename labels describe source recordings; this is target-domain descriptive adaptation.

    optimizer=torch.optim.AdamW(model.semantic.parameters(),lr=.003)

    for step in range(500):

        logits=model.semantic(x);loss=torch.nn.functional.cross_entropy(logits,y)

        optimizer.zero_grad();loss.backward();optimizer.step()

    write(out/'semantic_fit.json',dict(events=len(events),sources=len(source_cache),steps=500,

        category_counts={c:labels.count(i) for i,c in enumerate(categories)},

        fitting_accuracy=float((model.semantic(x).argmax(1)==y).float().mean()),

        heldout_generalization='NOT_ESTABLISHED',supervision='SMPL filename category, recording-level',

        expert_preference_training_steps=0))

    torch.save(dict(state=model.semantic.state_dict(),categories=categories),out/'semantic_head.pt')

    np.save(out/'event_embeddings.npy',np.stack(embeddings))

    result['ablation_status']='completed_after_full_framework'

    result['event_semantics']='filename coarse labels, no expert music preference'

    write(out/'result.json',result)

    from dunhuang.alignment.select import select
    select(model,result,events,np.stack(embeddings),source_cache,out,ROOT)
    print('SDMR_COMPLETE',result['qualified'],flush=True)

if __name__=='__main__':main(Path(sys.argv[1]).resolve())

