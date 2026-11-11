"""Expert preference adaptation with explicit abstentions and frozen encoders."""
import torch
from torch.nn import functional as F

def fit(model,records,steps=300):
    eligible=[r for r in records if r.get('provenance')=='expert_comparison' and r.get('winner') in ('a','b') and r.get('agreement',0)>=.67]
    if not eligible:
        return dict(status='NO_AGREED_EXPERT_PREFERENCES',steps=0)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.preference.parameters():parameter.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.preference.parameters(),lr=.001)
    device=next(model.parameters()).device
    for _ in range(steps):
        loss=0.
        for row in eligible:
            music=torch.as_tensor(row['music_embedding'],device=device)
            a=torch.as_tensor(row['a_embedding'],device=device)
            b=torch.as_tensor(row['b_embedding'],device=device)
            difference=model.preference(torch.cat((music,a)))-model.preference(torch.cat((music,b)))
            sign=1 if row['winner']=='a' else -1
            loss=loss+F.softplus(-sign*difference).mean()
        optimizer.zero_grad();(loss/len(eligible)).backward();optimizer.step()
    return dict(status='FIT_AGREED_EXPERT_PREFERENCES',steps=steps,records=len(eligible),qualification='requires independent grouped evaluation')
