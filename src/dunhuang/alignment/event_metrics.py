"""Independent labelled metrics and clearly named source-clock proxy retrieval."""
import numpy as np
from .event_targets import validate_anchors

def multi_positive_loss(scores,positive,valid,temperature=.1):
    import torch
    if scores.shape!=positive.shape or valid.shape!=positive.shape or temperature<=0: raise ValueError("Candidate layouts differ")
    if not positive.dtype==valid.dtype==torch.bool or torch.any(positive&~valid): raise ValueError("Invalid positive mask")
    if not positive.any(-1).all() or not (valid&~positive).any(-1).all(): raise ValueError("A positive and a known negative required per query")
    z=scores/temperature
    return (torch.logsumexp(z.masked_fill(~valid,-1e4),-1)-torch.logsumexp(z.masked_fill(~positive,-1e4),-1)).mean()

def source_clock_masks(query,candidates,tolerance_frames=1):
    """Known captures of the same song/time are positives. Other times require non-overlap.
    Same component ambiguous interpretations are excluded, never forced negatives.
    """
    positive=[];valid=[]
    for r in candidates:
        same=r["music_uid"]==query["music_uid"]
        delta=abs(r["start_frame"]-query["start_frame"])
        p=same and delta<=tolerance_frames
        negative=(same and delta>=120) or (r["source_group"]!=query["source_group"] and
            r["music_uid"]!=query["music_uid"] and r["choreography_uid"]!=query["choreography_uid"])
        positive.append(p);valid.append(p or negative)
    return np.asarray(positive,bool),np.asarray(valid,bool)

def proxy_retrieval(scores,positive,valid):
    ranks=[];wins=[]
    for s,p,v in zip(scores,positive,valid):
        if not p.any() or not (v&~p).any(): continue
        best=np.max(s[p]);wrong=s[v&~p]
        # Pessimistic ties prevent constant/clock-only scores claiming perfect retrieval.
        ranks.append(int(1+np.sum(wrong>=best-1e-7)))
        wins.append(float(np.mean((best>wrong+1e-7)+.5*(np.abs(best-wrong)<=1e-7))))
    if not ranks: return dict(status="NOT_EVALUABLE",queries=0,R1=None,R5=None,negative_pair_win_rate=None)
    return dict(status="SOURCE_CLOCK_PROXY_ONLY",queries=len(ranks),R1=float(np.mean(np.asarray(ranks)<=1)),
        R5=float(np.mean(np.asarray(ranks)<=5)),negative_pair_win_rate=float(np.mean(wins)))

def independent_metrics(packets,rows,predictions,tolerance=.1):
    if not packets:
        return dict(status="NOT_EVALUABLE_NO_INDEPENDENT_ANCHORS",validated=False,EventRecall1=None,EventRecall5=None,EventMAE_seconds=None,
            TypedEventAccuracy=None,NULL_precision=None,NULL_recall=None,risk_coverage=None)
    errors=[];typed=[];null_true=[];null_pred=[];conf=[];correct=[]
    for packet in packets:
        row=rows[packet["window_uid"]];anchors=validate_anchors(packet,row)
        pred=predictions.get(packet["window_uid"],[])
        if len(pred)!=len(anchors): raise ValueError("One prediction or explicit abstention required per anchor")
        for a,p in zip(anchors,pred):
            if a["ambiguous"]: continue
            isnull=p.get("is_null")
            if type(isnull) is not bool: raise ValueError("Explicit predicted NULL required")
            null_true.append(a["is_null"]);null_pred.append(isnull)
            good=a["is_null"]==isnull
            if not a["is_null"] and not isnull:
                e=abs(a["motion_seconds"]-p["motion_seconds"]);errors.append(e)
                t=(a["event_type"]==p["event_type"] and a["body_part"]==p["body_part"]);typed.append(t);good=good and e<=tolerance and t
            elif not a["is_null"]: typed.append(False)
            c=float(p["confidence"])
            if not np.isfinite(c) or not 0<=c<=1: raise ValueError("Finite unit confidence required")
            conf.append(c);correct.append(good)
    t=np.asarray(null_true);p=np.asarray(null_pred);tp=int((t&p).sum())
    coverage=[]
    if conf:
        # Accept confidence ties together; do not optimise a threshold on this evaluation population.
        for threshold in sorted(set(conf),reverse=True):
            keep=np.asarray(conf)>=threshold
            coverage.append(dict(threshold=threshold,coverage=float(keep.mean()),risk=float(1-np.asarray(correct)[keep].mean())))
    return dict(status="INDEPENDENT_ANCHORS_EVALUATED",validated=True,anchors=len(conf),
        EventMAE_seconds=float(np.mean(errors)) if errors else None,
        EventMAE_population="non-NULL correctly accepted matches; NULL misses included in risk and typed accuracy",
        TypedEventAccuracy=float(np.mean(typed)) if typed else None,
        NULL_precision=tp/int(p.sum()) if p.sum() else None,NULL_recall=tp/int(t.sum()) if t.sum() else None,
        risk_coverage=coverage)
