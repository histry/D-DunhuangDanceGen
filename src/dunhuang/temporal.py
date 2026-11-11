"""Geodesic guidance for verified, contiguous local SMPL joint trajectories."""
import json
import numpy as np
import torch
from .conditioning import anchor_windows, normalize_constraints, source_neighborhoods, constraint_support
from .io import write, checked


def prepare(request, normalizer, output, device):
    from pytorch3d.transforms import (rotation_6d_to_matrix,
        matrix_to_axis_angle, axis_angle_to_matrix)
    radius = request['trajectory_radius']
    strength = float(request['trajectory_strength'])
    if not 0 < strength < 1:
        raise ValueError('Temporal strength must be strictly between zero and one')
    if request.get('planned_time_warp', False):
        if 'event_plan' not in request or 'audio' not in request:
            raise ValueError('Planned temporal guidance requires a hash-bound Event Plan and audio')
        from .planning.event_plan import validate
        from .planning.phase_guidance import mapped_neighborhoods
        plan = validate(json.loads(checked(request['event_plan']).read_text()),
                        request['audio']['sha256'], request['frames'])
        rows, weights = mapped_neighborhoods(plan, request['frames'], radius)
        write(output/'phase_guidance.json',
              dict(schema='phase_guidance_v1', source='verified_event_plan',
                   music_clock_frames=request['frames'], events=len(plan['events']),
                   conditioned_frames=sum(float(w)>0 for w in weights),
                   mapping='inverse_monotone_source_music_slerp',
                   frozen_normalizer_unchanged=True, native_qualification='UNKNOWN'))
    else:
        # Historical B2 and V0.5 replay use their original unit-speed prompts.
        rows, weights = source_neighborhoods(request['frames'], request['anchors'], radius)
    raw, mask = anchor_windows(request['frames'], rows)
    write(output/'normalization_preflight.json', constraint_support(normalizer, raw, mask))
    normalize_constraints(normalizer, raw, mask)
    index = np.nonzero(mask[:,:,7:].reshape(len(raw),150,24,6).all(-1))
    window, tick, joint = [torch.as_tensor(a,device=device) for a in index]
    global_weights = {row['frame']: w for row,w in zip(rows, weights)}
    envelope = torch.tensor([global_weights[int(i*75+t)] for i,t in zip(index[0],index[1])],
                            dtype=torch.float32,device=device)
    channels = 7+6*joint[:,None]+torch.arange(6,device=device)
    selection = (window[:,None],tick[:,None],channels)
    goal = rotation_6d_to_matrix(torch.as_tensor(raw,device=device)[selection])
    scale = normalizer.scaler.scale_.to(device)[channels]
    bias = normalizer.scaler.min_.to(device)[channels]
    def guide(predicted, progress):
        original = predicted[selection]
        rotation = rotation_6d_to_matrix((original-bias)/scale)
        difference = matrix_to_axis_angle(rotation.transpose(-1,-2)@goal)
        weight = strength * progress**2 * envelope
        updated = rotation @ axis_angle_to_matrix(difference*weight[:,None])
        raw_vectors = ((original-bias)/scale).reshape(-1,2,3)
        encoded = (raw_vectors @ (rotation.transpose(-1,-2) @ updated)).reshape(-1,6)*scale+bias
        if not torch.isfinite(encoded).all():
            raise ValueError('Nonfinite temporal rotation')
        # Keep unsupported proposals at the model prediction, joint by joint.
        allowed = (encoded.abs().amax(-1)<=1+32*torch.finfo(encoded.dtype).eps) & (weight>0)
        result = predicted.clone()
        result[selection] = torch.where(allowed[:,None],encoded,original)
        return result
    return guide
