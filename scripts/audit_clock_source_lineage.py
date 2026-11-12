"""Archive-role and audio-oracle audit; metadata checks are not new Test evaluation."""
import argparse,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dunhuang.io import sha,write

def read(p):return json.loads(Path(p).read_text())
def run(current,reference,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    current=Path(current);reference=Path(reference);new=read(current/'manifest.json');old=read(reference/'manifest.json')
    overlap={}
    for label,oldkey,newkey in [('audio','audio_sha256','audio'),('raw_motion','motion_sha256','raw_motion')]:
        previous={}
        for r in old['rows']:previous.setdefault(r[oldkey],set()).add(r['role'])
        roles={}
        for r in new['rows']:
            key=r[newkey]['sha256']
            if key in roles and roles[key]!=r['role']:raise ValueError('Current source crosses roles')
            roles[key]=r['role']
        counts=Counter()
        for identity,role in roles.items():
            prior=previous.get(identity)
            counts[(role,','.join(sorted(prior)) if prior else 'not_found')]+=1
        overlap[label]={a+'->'+b:n for (a,b),n in counts.items()}
    matches=[]
    for r in old['rows']:
        if r['role']=='test':continue
        cache=current/'audio'/(r['audio_sha256']+'.npz')
        source=reference/Path(r['path'].replace(chr(92),'/')).name
        if not cache.exists() or not source.exists():continue
        if sha(source)!=r['sha256']:raise ValueError('Frozen reference feature file changed')
        with np.load(cache) as z:values=z['values'];mask=z['mask']
        start=r['start_frame']
        if start+120>len(values):continue
        with np.load(source) as z:oldvalues=z['music'];oldmask=z['music_mask']
        matches.append(dict(max_abs=float(abs(values[start:start+120]-oldvalues).max()),
            masks_equal=bool(np.array_equal(mask[start:start+120],oldmask)),
            bitwise_equal=bool(np.array_equal(values[start:start+120],oldvalues)),
            source_feature_sha256=r['sha256'],new_audio_cache_sha256=sha(cache)))
        if len(matches)==32:break
    def versions(paths):
        return sorted({read(p).get('librosa_version','unrecorded') for p in paths})
    result=dict(schema='clock_source_lineage_audit_v1',current_manifest_sha256=sha(current/'manifest.json'),
        frozen_reference_manifest_sha256=sha(reference/'manifest.json'),source_role_overlap=overlap,
        validation_sources_in_prior_train=overlap['audio'].get('val->train',0)+overlap['raw_motion'].get('val->train',0),
        old_audio_librosa_versions=versions(reference.glob('*.audio.json')),
        new_audio_librosa_versions=versions((current/'audio').glob('*.json')),
        audio_oracle=dict(status='MEASURED_SHARED_WINDOWS' if matches else 'NOT_EVALUABLE_NO_SHARED_CACHE',
            windows=len(matches),selection='first32 overlapping non-Test cached windows in frozen archive order',
            max_abs=max((r['max_abs'] for r in matches),default=None),
            masks_all_equal=all(r['masks_equal'] for r in matches) if matches else None,
            bitwise_equal_windows=sum(r['bitwise_equal'] for r in matches),details=matches),
        full_model_blindness_proven=False,test_model_evaluations=0,used_to_change_split=False,
        scope='known identity/archive roles only; prior Val checkpoint selection and unknown EDGE pretraining exposure remain')
    write(out,result);print('SOURCE_LINEAGE_AUDITED',len(matches),result['validation_sources_in_prior_train'])
def main():
    p=argparse.ArgumentParser()
    for k in ('current','reference','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.current,a.reference,a.out)
if __name__=='__main__':main()
