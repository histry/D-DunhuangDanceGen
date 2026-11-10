import numpy as np
from .contracts import MotionAdapter, motion151

def anchor_windows(frames, anchors):
    """Map global sparse full-joint poses consistently into every covering window.
    Root and contacts are deliberately unconditioned in the first style experiment.
    """
    count = max(1, (max(0, frames-150)+74)//75+1)
    values = np.zeros((count,150,151), np.float32)
    masks = np.zeros_like(values, dtype=bool)
    for anchor in anchors:
        t = anchor['frame']
        if type(t) is not int or not 0 <= t < frames:
            raise ValueError('Anchor outside the actual music clock')
        pose = motion151(np.asarray(anchor['pose'], np.float32).reshape(1,151))
        edge = MotionAdapter.to_edge(pose)[0]
        joints = anchor.get('joints', list(range(1,22)))
        if not joints or len(set(joints)) != len(joints) or any(type(j) is not int or not 0 <= j < 24 for j in joints):
            raise ValueError('Explicit unique SMPL joint indices required')
        channels = np.concatenate([np.arange(7+6*j,13+6*j) for j in joints])
        for i in range(count):
            local = t-75*i
            if 0 <= local < 150:
                prior = masks[i,local,channels]
                if np.any(prior & (np.abs(values[i,local,channels]-edge[channels]) > 1e-6)):
                    raise ValueError('Conflicting anchor constraints')
                masks[i,local,channels] = True
                values[i,local,channels] = edge[channels]
    return values, masks

def normalize_constraints(normalizer, values, masks):
    import torch
    v = torch.as_tensor(values).clone()
    m = torch.as_tensor(masks)
    s = normalizer.scaler
    affine = v*s.scale_.cpu()+s.min_.cpu()
    if torch.any(affine[m].abs() > 1+32*torch.finfo(v.dtype).eps):
        raise ValueError('Anchor outside frozen checkpoint normalization support')
    result = normalizer.normalize(v.clone())
    recovered = normalizer.unnormalize(result.clone())
    if m.any() and float((recovered[m]-v[m]).abs().max()) > 1e-4:
        raise ValueError('Anchor normalization roundtrip failed')
    return result, m


def constraint_support(normalizer, values, masks):
    import torch
    x=torch.as_tensor(values)
    affine=x*normalizer.scaler.scale_.cpu()+normalizer.scaler.min_.cpu()
    outside=(affine.abs()>1+32*torch.finfo(x.dtype).eps)&torch.as_tensor(masks)
    joints=[]
    for j in range(24):
        count=int(outside[:,:,7+6*j:13+6*j].sum())
        if count:joints.append(dict(joint=j,unsupported_coordinates=count))
    return dict(supported=not bool(outside.any()),unsupported_coordinates=int(outside.sum()),joints=joints,
        frozen_normalizer=True,poses_clipped=False)

def source_neighborhoods(frames, anchors, radius):
    """Bind each music tick to an actual 30 Hz source frame at unit time scale."""
    from .io import checked
    if type(radius) is not int or radius < 1:
        raise ValueError('Positive integer trajectory radius required')
    rows, weights = [], []
    occupied = set()
    for anchor in anchors:
        motion = motion151(np.load(checked(anchor['source']), allow_pickle=False))
        center, tick = anchor['source_frame'], anchor['frame']
        if min(center, tick) < radius or center+radius >= len(motion) or tick+radius >= frames:
            raise ValueError('Insufficient source or music context')
        if not np.array_equal(motion[center], np.asarray(anchor['pose'], dtype=np.float32)):
            raise ValueError('Source center differs from frozen target')
        for offset in range(-radius, radius+1):
            frame = tick+offset
            if frame in occupied:
                raise ValueError('Intersecting temporal anchor neighborhoods')
            occupied.add(frame)
            rows.append(dict(frame=frame, pose=motion[center+offset], joints=anchor['joints']))
            weights.append(float(np.cos(np.pi*offset/(2*radius))**2) if abs(offset)<radius else 0.)
    return rows, weights
