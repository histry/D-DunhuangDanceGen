import json
from pathlib import Path
import numpy as np
from dunhuang.io import sha,write,pinned
POPULATION_SHA='8c0c9dc9725cea5797e90f3dd463a7783f825fed43bb374bdbcb37d2286eef04'
def prepare(population,output,legacy,assets):
    import sys
    import subprocess
    commit=subprocess.check_output(['git','-C',str(legacy),'rev-parse','HEAD'],text=True).strip()
    sys.path.insert(0,str(legacy))
    from training.paired_alignment_data_v4 import _native_motion
    from evaluation.paired_alignment_observations_v3 import full_audio_observations,CONTRACT
    if sha(population)!=POPULATION_SHA:raise ValueError('Population identity mismatch')
    output.mkdir(parents=True,exist_ok=False)
    rows=json.loads(population.read_text())['records']
    roles={}
    for row in rows:
        role=row['purpose_role']
        if role not in ('train','val','test'):continue
        group=row['group_uid']
        if group in roles and roles[group]!=role:raise ValueError('Group leakage')
        roles[group]=role
    cache={};manifest=[];blocked=[]
    for index,original in enumerate(rows):
        if original['purpose_role'] not in ('train','val','test'):continue
        row=dict(original)
        for name in ('motion_asset','audio_asset'):
            ref=dict(row[name]);tail=ref['path'].split('/assets/datasets/')[1]
            ref['path']=str(assets/tail);row[name]=ref
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Raw source mismatch '+ref['path'])
        key=row['audio_asset']['sha256']
        try:
            if key not in cache:
                music,mm,receipt=full_audio_observations(row['audio_asset']['path'])
                cache[key]=(music,mm)
                write(output/(key+'.audio.json'),receipt)
            music,mm=cache[key]
            motion,dm,_=_native_motion(row,lambda key:None)
        except Exception as exc:
            # Preserve explicit blocked sources without manufacturing observations.
            blocked.append(dict(pair=row['pair_uid'],error=repr(exc)))
            write(output/'blocked.json',blocked)
            continue
        length=min(len(music),len(motion))
        # Full continuous support plus symmetric half-second halo, no circular shifts.
        if length<150:continue
        starts=np.unique(np.linspace(15,length-135,4).astype(int)).tolist()
        for start in starts:
            file=output/('pair_%05d_%05d.npz'%(index,start))
            np.savez_compressed(file,music=music[start:start+120],music_mask=mm[start:start+120],
                motion=motion[start-15:start+135],motion_mask=dm[start-15:start+135])
            manifest.append(dict(path=str(file.resolve()),sha256=sha(file),role=row['purpose_role'],
                group=row['group_uid'],pair=row['pair_uid'],domain=row['dataset_id'],
                audio_sha256=key,motion_sha256=row['motion_asset']['sha256'],start_frame=start,frames=120,
                pairing_evidence=row['pairing_evidence']))
        if index%25==0:print('DATA',index,len(manifest),len(blocked),flush=True)
    write(output/'manifest.json',dict(rows=manifest,blocked=blocked,feature_contract=CONTRACT,
        population_sha256=POPULATION_SHA,extractor_commit=commit,roles=roles,heldout_status='previously exposed development regression',
        window_policy='four evenly spaced 4s windows with actual +/-0.5s motion halo'))
    return output/'manifest.json'

def load(manifest):
    from collections import Counter
    doc=json.loads(manifest.read_text());rows=doc['rows']
    music=[];motion=[];mm=[];dm=[]
    for row in rows:
        path=manifest.parent/Path(row['path'].replace(chr(92),'/')).name
        if sha(path)!=row['sha256']:raise ValueError('Cached features changed')
        with np.load(path) as z:
            music.append(z['music']);motion.append(z['motion']);mm.append(z['music_mask']);dm.append(z['motion_mask'])
    arrays={k:np.stack(v) for k,v in zip(('music','motion','mm','dm'),(music,motion,mm,dm))}
    norm={}
    train=[i for i,r in enumerate(rows) if r['role']=='train']
    counts=Counter(rows[i]['group'] for i in train)
    for name,mask in [('music','mm'),('motion','dm')]:
        total=np.zeros(arrays[name].shape[-1]);square=total.copy();mass=total.copy()
        for i in train:
            x=arrays[name][i];m=arrays[mask][i]
            if name=='motion':x=x[15:135];m=m[15:135]
            w=1/(counts[rows[i]['group']]*len(x))
            total+=(x*m).sum(0)*w;square+=(x*x*m).sum(0)*w;mass+=m.sum(0)*w
        mean=total/np.maximum(mass,1e-8)
        std=np.sqrt(np.maximum(square/np.maximum(mass,1e-8)-mean**2,1e-4))
        std[mass==0]=1
        arrays[name]=(((arrays[name]-mean)/std)*arrays[mask]).astype(np.float32)
        norm[name]=dict(mean=mean.tolist(),std=std.tolist())
    return rows,arrays,norm
