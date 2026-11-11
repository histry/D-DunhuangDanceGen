import json
from pathlib import Path
import torch
from dunhuang.io import checked,sha

def authorize(checkpoint,report):
    receipt=json.loads(Path(report).read_text())
    if sha(checkpoint)!=receipt['retrieval_checkpoint_sha256']:raise ValueError('Checkpoint identity differs')
    return dict(enabled=bool(receipt['qualified']) and receipt.get('qualification_basis')=='validation_only',fallback_reason=None if receipt['qualified'] and receipt.get('qualification_basis')=='validation_only' else 'missing_validation_qualification',
                preference_status=receipt['preference_status'],checkpoint_sha256=sha(checkpoint))

def rank(model,music,music_mask,candidates,norm,k=5):
    if not candidates:raise ValueError('No feasible candidates')
    with torch.no_grad():
        mean=torch.tensor(norm['music']['mean'],device=music.device)
        std=torch.tensor(norm['music']['std'],device=music.device)
        query=model.music((music-mean)/std*music_mask,music_mask)[0]
        scores=[]
        for candidate in candidates:
            if not candidate['support_receipt']['supported']:continue
            motion=checked(candidate['features'])
            import numpy as np
            with np.load(motion) as z:
                values=torch.as_tensor(z['motion'],device=query.device).float()[None]
                mask=torch.as_tensor(z['mask'],device=query.device)[None]
            mean=torch.tensor(norm['motion']['mean'],device=query.device)
            std=torch.tensor(norm['motion']['std'],device=query.device)
            vector=model.motion((values-mean)/std*mask,mask)[0]
            scores.append(dict(event_uid=candidate['event_uid'],source_uid=candidate['source_uid'],
                score=float((query*vector).sum()),source=candidate['source'],support_receipt=candidate['support_receipt']))
    return sorted(scores,key=lambda row:(-row['score'],row['event_uid']))[:k]


def synchronized_scores(global_model,sync_model,music,music_mask,motion,motion_mask):
    """Scores from matching encoders; synchronization head may be a separate checkpoint."""
    for branch in ('music','motion'):
        for name,value in getattr(global_model,branch).state_dict().items():
            if not torch.equal(value,getattr(sync_model,branch).state_dict()[name]):
                raise ValueError('Global and synchronization encoder identities differ')
    with torch.no_grad():
        m,mt,mv=global_model.music(music,music_mask)
        d,dt,dv=global_model.motion(motion,motion_mask)
        if mt.shape[1]!=dt.shape[1]:raise ValueError('Phase-aligned sequences must share physical grid')
        return (m*d).sum(-1),sync_model.sync_score(mt,dt,mv&dv)
