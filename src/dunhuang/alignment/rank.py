import json
from pathlib import Path
import torch
from dunhuang.io import checked,sha

def authorize(checkpoint,report):
    receipt=json.loads(Path(report).read_text())
    if sha(checkpoint)!=receipt['retrieval_checkpoint_sha256']:raise ValueError('Checkpoint identity differs')
    return dict(enabled=bool(receipt['qualified']),fallback_reason=None if receipt['qualified'] else 'heldout_sync_or_retrieval_not_qualified',
                preference_status=receipt['preference_status'],checkpoint_sha256=sha(checkpoint))

def rank(model,music,music_mask,candidates,k=5):
    if not candidates:raise ValueError('No feasible candidates')
    with torch.no_grad():
        query=model.music(music,music_mask)[0]
        scores=[]
        for candidate in candidates:
            if not candidate['support_receipt']['supported']:continue
            motion=checked(candidate['features'])
            import numpy as np
            with np.load(motion) as z:
                values=torch.as_tensor(z['motion'],device=query.device).float()[None]
                mask=torch.as_tensor(z['mask'],device=query.device)[None]
            vector=model.motion(values,mask)[0]
            scores.append(dict(event_uid=candidate['event_uid'],source_uid=candidate['source_uid'],
                score=float((query*vector).sum()),source=candidate['source'],support_receipt=candidate['support_receipt']))
    return sorted(scores,key=lambda row:(-row['score'],row['event_uid']))[:k]
