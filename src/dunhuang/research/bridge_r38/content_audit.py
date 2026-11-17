"""Independent CPU readout of sealed embeddings; no model, optimizer or data fitting."""
import sys
from pathlib import Path
import numpy as np
from dunhuang.research.complete_r36.core import read,write,sha
root=Path(sys.argv[1]);a=np.load(root/'content/evaluation.npz');old=read(root/'content/result.json');records={}
for mode in ('correct','shuffle','shift'):
    M=a[mode+'_music'];D=a[mode+'_motion'];score=a[mode+'_scores'];error=float(abs(M@D.T-score).max())
    if error>5e-6:raise ValueError('Embeddings do not reproduce scores')
    p=a['positive'];n=a['negative'];hits=[]
    for i in range(len(M)):
        valid=p[i]|n[i]
        if not p[i].any() or not n[i].any():continue
        indices=np.flatnonzero(valid);hits.append(bool(p[i,indices[np.argmax(score[i,valid])]]))
    recall=float(np.mean(hits))
    if recall!=old['metrics'][mode]['all']['recall_at_1']:raise ValueError('Metric readout differs')
    if not np.array_equal(D,a['correct_motion']):raise ValueError('Music intervention changed motion embeddings')
    records[mode]=dict(recall_at_1=recall,queries=len(hits),score_matmul_max_error=error,music_embedding_rms_change=float(np.sqrt(np.mean((M-a['correct_music'])**2))))
write(root/'independent_content_audit.json',dict(records=records,optimizer_steps=0,arrays_sha256=sha(root/'content/evaluation.npz'),scope='CPU_EMBEDDING_SCORE_AND_ALL_POOL_RANK_READOUT_NOT_TRAINING_REPLAY'))
print('CONTENT_CPU_READOUT_VERIFIED',len(a['val_indices']))
