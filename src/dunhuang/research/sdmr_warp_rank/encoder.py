"""CPU-only frozen R41 local content readout; raw and normalized music differ."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read, sha
from dunhuang.research.event_alignment_r49.common import Frozen, sealed
from dunhuang.research.paired_event_r41.model import EventContent
from dunhuang.research.source_tasks.common import forbid_optimizers
from dunhuang.research.warp_coupling_r53.common import array_hash


CHECKPOINT_SHA = '212148e7cd86b482cc7734035725b1165edcd7a1c35e03fa513f6982f440e772'
NORMALIZER_SHA = '80a83d6f7a5f1bca62d3298cba206a871188f5c6b1519ab4a897672c7c986e35'


def _fingerprint(net):
    h = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        h.update(name.encode('utf8'))
        h.update(array_hash(value.detach().cpu().numpy()).encode('ascii'))
    return h.hexdigest()


def _load_cpu():
    forbid_optimizers()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path('outputs/paired_event_r41_01')
    # Preserve the R49 wrapper's sealed inputs and exact local branches while
    # avoiding its automatic CUDA device choice.
    f = Frozen.__new__(Frozen)
    f.device = 'cpu'
    f.manifest_path = sealed(root, 'data/manifest.json')
    f.manifest = read(f.manifest_path)
    if f.manifest['normalizer']['sha256'] != NORMALIZER_SHA:
        raise ValueError('Frozen R37 Train normalization identity changed')
    f.normalizer = read(checked(f.manifest['normalizer']))['statistics']
    f.checkpoint = sealed(root, 'main/model.pt')
    if sha(f.checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Frozen R41 model identity changed')
    with np.load(sealed(root, 'data/skeleton.npz'), allow_pickle=False) as z:
        f.joints, f.parents = z['J'].copy(), z['parents'].copy()
    f.net = EventContent(96).cpu().eval().requires_grad_(False)
    f.net.load_state_dict(torch.load(f.checkpoint, map_location='cpu', weights_only=True)['state'])
    if sum(p.numel() for p in f.net.parameters()) != 522632:
        raise ValueError('Frozen R41 architecture parameter count changed')
    return f


class FrozenEncoder:
    """Return ``(tokens[T-12,96], valid[T-12], provenance)``.

    ``_fixture`` is exclusively for synthetic unit tests; its proof is marked
    TEST_FIXTURE and cannot be mistaken for actual checkpoint inference.
    """
    def __init__(self, *, _fixture=None):
        self.f = _load_cpu() if _fixture is None else _fixture
        self.fixture = _fixture is not None
        self.f.net.cpu().eval().requires_grad_(False)
        self.frozen, self.net = self.f, self.f.net
        self.joints, self.parents = self.f.joints, self.f.parents
        self.binding = self.f.binding()
        if (self.binding['checkpoint_sha256'] != CHECKPOINT_SHA
                or self.binding['normalizer_sha256'] != NORMALIZER_SHA
                or self.binding['device'] != 'cpu'):
            raise ValueError('Exact frozen checkpoint, Train normalizer and CPU required')
        self.before = _fingerprint(self.f.net)

    def weight_digest(self):
        return self.check()

    def check(self):
        current = _fingerprint(self.f.net)
        if current != self.before:
            raise ValueError('Frozen model weights changed during inference')
        if self.f.net.training or any(p.requires_grad or p.grad is not None for p in self.f.net.parameters()):
            raise ValueError('Frozen eval-only model must not acquire gradients')
        return current

    def _finish(self, tokens, valid, length, proof):
        tokens, valid = np.asarray(tokens), np.asarray(valid, bool)
        if tokens.shape != (length-12, 96) or valid.shape != (length-12,) or not np.isfinite(tokens).all():
            raise ValueError('Actual complete-sequence13-frame content token support required')
        proof.update(self.binding, model_digest_before=self.before, model_digest_after=self.check(),
            token_frame_centers=list(range(6, length-6)), input_frames=length,
            token_frames=length-12, token_radius_frames=6,
            valid_tokens=int(valid.sum()), token_sha256=array_hash(tokens), valid_sha256=array_hash(valid),
            test_fixture=self.fixture, training_updates=0, optimizer_calls=0, can_route=False)
        return tokens.copy(), valid.copy(), proof

    @torch.inference_mode()
    def encode_music_normalized(self, values, valid):
        """R59 M is already normalized; never pass it through normalize again."""
        self.check()
        values = np.asarray(values, np.float32)
        valid = np.asarray(valid, bool)
        if values.ndim != 2 or values.shape[1] != 50 or valid.shape != values.shape[:1] or len(values) < 13 or not np.isfinite(values).all():
            raise ValueError('Finite prenormalized50D music and frame mask required')
        x = torch.as_tensor(values)[None]
        x = self.f.net.music(x)
        x = self.f.net.local_music_norm(self.f.net.local_music(x.transpose(1,2)).transpose(1,2))
        tokens = F.normalize(x, dim=-1)[0].cpu().numpy()
        support = np.lib.stride_tricks.sliding_window_view(valid,13).all(-1)
        return self._finish(tokens, support, len(values), dict(kind='MUSIC',
            input_sha256=array_hash(values), input_valid_sha256=array_hash(valid),
            normalization='R59_ALREADY_NORMALIZED_NO_ADDITIONAL_NORMALIZATION', additional_normalizations=0))

    @torch.inference_mode()
    def encode_music_raw(self, values, mask):
        self.check()
        values, mask = np.asarray(values, np.float32), np.asarray(mask, bool)
        if values.ndim != 2 or values.shape[1] != 50 or values.shape != mask.shape or len(values) < 13 or not np.isfinite(values[mask]).all():
            raise ValueError('Raw50D music and channel observations required')
        tokens, valid = self.f.music_tokens(values, mask)
        return self._finish(tokens, valid, len(values), dict(kind='MUSIC',
            input_sha256=array_hash(values), input_mask_sha256=array_hash(mask),
            normalization='RAW50_FROZEN_TRAIN_NORMALIZER_ONCE', additional_normalizations=1))

    @torch.inference_mode()
    def encode_rotations(self, rotations):
        self.check()
        rotations = np.asarray(rotations, float)
        if rotations.ndim != 4 or rotations.shape[1:] != (22,3,3) or len(rotations) < 13 or not np.isfinite(rotations).all():
            raise ValueError('Real full22-joint SO(3) sequence required')
        if not np.allclose(rotations.swapaxes(-1,-2)@rotations, np.eye(3), atol=1e-5, rtol=0) or not np.allclose(np.linalg.det(rotations),1,atol=1e-5,rtol=0):
            raise ValueError('Proper SO(3) rotation matrices required')
        values, mask = self.f.motion(rotations)  # observed features, derivative reset, Train normalization once
        values, mask = np.asarray(values, np.float32), np.asarray(mask, bool)
        if values.shape != (len(rotations),274) or mask.shape != values.shape or mask[0,198:272].any():
            raise ValueError('Warped observations need explicit unknown initial dynamics')
        # Clock/duration metadata must stay outside content even when the
        # upstream observation function includes those columns.
        mask = mask.copy()
        mask[:,-2:] = False
        values = np.where(mask,values,0).astype(np.float32)
        tokens, valid = self.f.motion_tokens(values,mask)
        return self._finish(tokens,valid,len(rotations),dict(kind='MOTION',
            rotation_sha256=array_hash(rotations), normalized_observations_sha256=array_hash(values),
            content_observation_mask_sha256=array_hash(mask), observed_values=int(mask.sum()),
            clock_duration_channels_observed=False, first_frame_dynamics_observed=False,
            normalization='ACTUAL_SO3_WARP_REDERIVED274_FROZEN_TRAIN_NORMALIZER_ONCE', additional_normalizations=1,
            input_observation_contract='132ROW6D_66HEADING_INVARIANT_FK_66OMEGA_8AGG_2MASKED_CLOCK',
            derivatives='RECOMPUTED_AFTER_ACTUAL_SO3_WARP_FIRST198_272_UNKNOWN'))
