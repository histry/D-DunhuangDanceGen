"""Independent modality-local crops; original recording clock is used only in labels."""
import numpy as np


def crop_origins(seed,step,count,max_origin=30):
    if count<1 or max_origin<1:raise ValueError('Independent crop population')
    return np.random.default_rng(seed+step).integers(0,max_origin+1,(count,2))


def clock_labels(length,music_origin,motion_origin,margin=30):
    if length<=2*margin or abs(music_origin-motion_origin)>margin:raise ValueError('Common interior support')
    music_index=np.arange(margin,length-margin);motion_index=music_index+music_origin-motion_origin
    if (motion_index<0).any() or (motion_index>=length).any():raise ValueError('Invalid clock label')
    return music_index,motion_index


def offset_scores(music,motion,offsets,margin=30):
    """Equal-support candidate scores, never absolute-origin or length features."""
    index=np.arange(margin,len(music)-margin)
    if not len(index) or max(abs(np.asarray(offsets)))>margin:raise ValueError('Common offset support')
    return np.array([(music[index]*motion[index+d]).sum(-1).mean() for d in offsets])


def tie_accuracy(scores,truth):
    scores=np.asarray(scores);ties=np.isclose(scores,scores.max(-1,keepdims=True),atol=1e-7,rtol=0)
    return np.take_along_axis(ties,np.asarray(truth)[...,None],axis=-1)[...,0]/ties.sum(-1)
