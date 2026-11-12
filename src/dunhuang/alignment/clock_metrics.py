"""Identity-aware clock discrimination; source-clock labels are not expert beat labels."""
import numpy as np
import torch

SHIFTS=(-15,-8,0,8,15)

def correlation(m,d):
    m=m-m.mean(-1,keepdims=True);d=d-d.mean(-1,keepdims=True)
    return (m*d).sum(-1)/np.maximum(np.linalg.norm(m,axis=-1)*np.linalg.norm(d,axis=-1),1e-12)

def energy_scores(arrays):
    m=arrays['energy_music'][:,::2]
    d=np.stack([arrays['energy_motion'][:,15+s:135+s:2] for s in SHIFTS],1)
    return correlation(m[:,None,:],d)

def grouped(values,rows,iterations=300):
    result={}
    for key in ('recording_uid','choreography_uid','source_group'):
        groups={}
        for r,value in zip(rows,values):groups.setdefault(r[key],[]).append(float(value))
        means=np.array([np.mean(groups[k]) for k in sorted(groups)])
        rng=np.random.default_rng(20261117)
        draws=[rng.choice(means,len(means),replace=True).mean() for _ in range(iterations)]
        result[key]=dict(groups=len(means),balanced_mean=float(means.mean()),
            bootstrap95=np.percentile(draws,[2.5,97.5]).tolist() if len(means)>1 else None)
    return result

def clock_report(scores,rows,wrong_scores=None):
    positive=scores[:,2,None];negative=scores[:,[0,1,3,4]]
    win=((positive>negative)+.5*(positive==negative)).mean(-1)
    result=dict(queries=len(rows),aligned_vs_shift=grouped(win,rows),
        offset_choice_accuracy=float((scores.argmax(-1)==2).mean()),chance_offset=.2,
        per_shift={str(s/30):float(np.mean((scores[:,2]>scores[:,i])+.5*(scores[:,2]==scores[:,i]))) for i,s in enumerate(SHIFTS) if s},
        bootstrap_scope='whole known association groups; only two Val components, coverage unestablished',
        labels='official paired source clock; not independent music-onset/action-peak correspondence',
        fresh_blind_test=False)
    if wrong_scores is not None:
        wins=(scores[:,2]>wrong_scores)+.5*(scores[:,2]==wrong_scores)
        result['unrelated_pair_control']=grouped(wins,rows)
        result['unrelated_pair_definition']='different known music/choreography/component; does not mean another dance is illegitimate'
    return result

def retrieval(scores,rows,task):
    values=[];chance=[]
    for i,row in enumerate(rows):
        pool=np.arange(len(rows)) if task=='song' else np.array([j for j,r in enumerate(rows) if r['music_uid']==row['music_uid']])
        positive=np.array([rows[j]['music_uid']==row['music_uid'] and (task=='song' or rows[j]['start_frame']==row['start_frame']) for j in pool])
        order=np.argsort(-scores[i,pool],kind='stable');rank=int(np.flatnonzero(positive[order])[0])+1
        values.append((rank<=1,rank<=5,1/rank));chance.append(positive.mean())
    v=np.array(values)
    return dict(recall1=float(v[:,0].mean()),recall5=float(v[:,1].mean()),mrr=float(v[:,2].mean()),
        recall1_grouped=grouped(v[:,0],rows),chance_positive_fraction=float(np.mean(chance)),
        positives='same canonical music identity'+(' and same original source time; all captures accepted' if task=='local' else '; all captures accepted'),
        task=task)
