"""Frozen catalog selection, with explicit source identity and deterministic ranking."""
import numpy as np
from .io import checked
from .contracts import motion151

def retrieve(catalog, query=None, limit=3):
    rows = catalog['entries']
    if not rows or limit < 1:
        raise ValueError('Nonempty catalog and positive limit required')
    if query is None:
        ranked = sorted(rows, key=lambda r: r['id'])
    else:
        query = np.asarray(query, dtype=float)
        if query.ndim != 1 or not np.isfinite(query).all():
            raise ValueError('Finite one-dimensional retrieval query required')
        def distance(row):
            value = np.asarray(row['descriptor'], dtype=float)
            if value.shape != query.shape or not np.isfinite(value).all():
                raise ValueError('Descriptor/query contract mismatch')
            return float(np.linalg.norm(value-query)), row['id']
        ranked = sorted(rows, key=distance)
    result = []
    for row in ranked[:limit]:
        motion = motion151(np.load(checked(row['motion']), allow_pickle=False))
        frame = row['source_frame']
        if type(frame) is not int or not 0 <= frame < len(motion):
            raise ValueError('Catalog source frame out of range')
        result.append(dict(id=row['id'], pose=motion[frame], source_frame=frame, source=row['motion']))
    return result

def select_supported_phrase(candidates, density, previous_exit=None):
    """Rank supported clips by activity percentile and angular entry compatibility."""
    if not candidates or not np.isfinite(density) or not 0<=density<=1:
        raise ValueError('Candidates and music density in [0,1] required')
    activities=np.array([c['activity_rad_s'] for c in candidates])
    def score(candidate):
        percentile=float(np.mean(activities<candidate['activity_rad_s']))
        distance=0.
        if previous_exit is not None:
            relative=previous_exit.swapaxes(-1,-2)@candidate['entry_rotation']
            distance=float(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)).mean()/np.pi)
        return abs(percentile-density)+0.25*distance,percentile,distance
    selected=min(candidates,key=lambda c:(score(c)[0],c['source_uid'],c['raw_center']))
    total,percentile,entry=score(selected)
    return selected,dict(score=total,activity_percentile=percentile,entry_distance_over_pi=entry,density=density)
