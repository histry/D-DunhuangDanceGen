"""Frozen-score metadata controls and music-group summaries; never selects or updates weights."""
import argparse,collections
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha

def run(root,out):
    root=Path(root);d=root/'content_data';manifest=read(d/'manifest.json');rows=manifest['rows'];a=np.load(root/'content_model/evaluation.npz');b=np.load(root/'content_replay/evaluation.npz')
    for k in a.files:assert np.array_equal(a[k],b[k]),k
    assert read(root/'content_model/result.json')['population_sha256']==sha(d/'manifest.json')
    assert sha(d/'normalization.json')==manifest['normalizer_sha256']
    val=[rows[i] for i in a['val_indices']];p=a['positive'];n=a['negative'];N=len(val)
    center=np.array([np.mean(r['audio_frame_interval']) for r in val]);duration=np.array([r['frames'] for r in val])
    scores=dict(learned=a['scores'],clock_only=-abs(center[:,None]-center),duration_only=-abs(duration[:,None]-duration))
    results={}
    for task in ('all_verified_val','same_music_time'):
        result={}
        for name,score in scores.items():
            groups=collections.defaultdict(list);recordings=collections.defaultdict(list);ties=[];uniform=[]
            for i,r in enumerate(val):
                valid=p[i]|n[i]
                if task=='same_music_time':valid&=np.array([r['audio_sha256']==x['audio_sha256'] for x in val])
                if not (p[i]&valid).any() or not (n[i]&valid).any():continue
                ids=np.flatnonzero(valid);best=ids[np.argmax(score[i,ids])];correct=bool(p[i,best])
                groups[r['source_group']].append(correct);recordings[r['recording_uid']].append(correct)
                top=ids[score[i,ids]==score[i,best]];ties.append(len(top));uniform.append(float(p[i,top].mean()))
            result[name]=dict(query_micro=float(np.mean([v for g in groups.values() for v in g])),music_group_macro=float(np.mean([np.mean(g) for g in groups.values()])),
                              recording_macro=float(np.mean([np.mean(g) for g in recordings.values()])),
                              per_music_group={k:dict(queries=len(g),recall_at_1=float(np.mean(g))) for k,g in sorted(groups.items())},
                              mean_top_ties=float(np.mean(ties)),uniform_top_tie_expected_recall=float(np.mean(uniform)),tie_rule='FIRST_FROZEN_CANDIDATE_INDEX; UNIFORM_EXPECTATION_ALSO_REPORTED')
        results[task]=result
    write(out,dict(results=results,evaluation_replay_exact=True,optimizer_steps=0,weight_sha256=sha(root/'content_model/model.pt'),
        population_sha256=sha(d/'manifest.json'),controls_use_metadata_not_artistic_truth=True,
        clock_only_scope='KNOWN_SOURCE_CLOCK_DIAGNOSTIC_NOT_AVAILABLE_AS_TARGET_UNPAIRED_CORRECT_PAIR_ORACLE',
        new_training_controls=0,matched_energy_control=False,shuffled_training_control=False,independent_new_songs=False,
        cultural_transfer_proven=False,can_route=False,operational_qualified=False))
    print('R37_CONTENT_DIAGNOSTICS',results,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
