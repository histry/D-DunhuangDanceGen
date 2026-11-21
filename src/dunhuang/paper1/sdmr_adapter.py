"""Frozen R41/R67 scoring on actual physical time, without a second latent Warp."""
import numpy as np
from dunhuang.research.sdmr_warp_rank.encoder import FrozenEncoder
from dunhuang.research.sdmr_warp_rank.readout import score_bank

def score_executed_bank_shadow(net,music,masks,rotations):
    # Both songs use the identical absolute frame interval, no circular shift.
    packs={k:net.encode_music_raw(music[k],masks[k]) for k in music}
    motions=[net.encode_rotations(R) for R in rotations]
    support=np.logical_and.reduce([x[1] for x in packs.values()]+[x[1] for x in motions])
    frames=len(rotations[0]);centres=np.arange(6,frames-6,dtype=float)
    details={};scores={}
    for name,(tokens,valid,proof) in packs.items():
        d=score_bank(tokens,[v[0] for v in motions],valid,[v[1] for v in motions],centres=centres,fixed_support=support)
        details[name]=dict(readout=d,music_encoding=proof)
        # A-source evidence is a global average of these same frozen local content
        # tokens. It is a declared readout, not the original separately trained head.
        def norm(v):return v/max(float(np.linalg.norm(v)),1e-8)
        global_scores=[float(norm(tokens[support].mean(0))@norm(v[0][support].mean(0))) if support.any() else None for v in motions]
        scores[name]=dict(d['scores'],GlobalLocalTokenMean=global_scores)
    return dict(scores=scores,details=details,motion_encoding=[v[2] for v in motions],common_support=support.tolist(),
                net_binding=net.binding,network_digest=net.check(),can_route=False,latent_warp_executed=False,
                global_readout_scope='L2_MEAN_FROZEN_LOCAL_TOKENS_NOT_ORIGINAL_TRAINED_GLOBAL_HEAD')

def recall_shadow(score_result,rows):
    ids=[i for i,r in enumerate(rows) if r['origin']=='E1W']
    values=score_result['scores']['actual']['GlobalLocalTokenMean']
    return dict(candidate_uids=[rows[i]['uid'] for i in ids],scores=[values[i] for i in ids],
                ranking=[rows[i]['uid'] for i in sorted(ids,key=lambda i:(values[i] is None,-(values[i] or 0),rows[i]['uid']))],
                candidates_removed=0,can_route=False)
