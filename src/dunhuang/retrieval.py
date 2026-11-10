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
