"""Frozen inputs and common observation support for the R49 comparisons."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from dunhuang.io import checked
from dunhuang.research.complete_r36.core import read, sha
from dunhuang.research.paired_event_r41.model import EventContent, RADIUS
from dunhuang.research.phase_r45.audio import extract
from dunhuang.research.phase_discrimination_r48.study import time_grid
from dunhuang.research.transition_r37.content import observations

R41 = Path("outputs/paired_event_r41_01")
R48 = Path("outputs/event_feasibility_r48_01")
BASELINE = "540bb27b4bf14c0d353c587dc8e9f96ea9228a54"
OFFSETS = (-15, -8, 0, 8, 15)
SEED = 49049
MULTI_WEIGHT = 0.1
WARP_WEIGHT = 0.05


def sealed(root, relative):
    manifest = read(root / "artifacts_manifest.json")
    entry = next(f for f in manifest["files"] if f["path"] == str(relative))
    return checked(dict(path=str(root / relative), sha256=entry["sha256"]))


class Frozen:
    def __init__(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.manifest_path = sealed(R41, "data/manifest.json")
        self.manifest = read(self.manifest_path)
        self.normalizer = read(checked(self.manifest["normalizer"]))["statistics"]
        self.checkpoint = sealed(R41, "main/model.pt")
        skeleton = np.load(sealed(R41, "data/skeleton.npz"), allow_pickle=False)
        self.joints, self.parents = skeleton["J"], skeleton["parents"]
        self.net = EventContent(96).to(self.device).eval().requires_grad_(False)
        self.net.load_state_dict(torch.load(self.checkpoint, weights_only=True, map_location=self.device)["state"])

    def normalize(self, values, mask, name):
        norm = self.normalizer[name]
        return np.where(mask, (values - norm["mean"]) / norm["std"], 0).astype(np.float32)

    def motion(self, rotation):
        values, mask = observations(rotation, self.joints, self.parents)
        mask[0, 198:272] = False
        return self.normalize(values, mask, "motion"), mask

    @torch.inference_mode()
    def music_tokens(self, values, mask):
        x = torch.as_tensor(self.normalize(values, mask, "music"), device=self.device)[None]
        x = self.net.music(x)
        x = self.net.local_music_norm(self.net.local_music(x.transpose(1, 2)).transpose(1, 2))
        valid = np.lib.stride_tricks.sliding_window_view(mask.any(-1), 13).all(-1)
        return torch.nn.functional.normalize(x, dim=-1)[0].cpu().numpy(), valid

    @torch.inference_mode()
    def motion_tokens(self, values, mask):
        mask = mask.copy()
        mask[:, -2:] = False
        x = torch.as_tensor(np.concatenate((np.where(mask, values, 0), mask), axis=-1), device=self.device)[None]
        x = self.net.motion(x)
        x = self.net.local_motion_norm(self.net.local_motion(x.transpose(1, 2)).transpose(1, 2))
        valid = np.lib.stride_tricks.sliding_window_view(mask.any(-1), 13).all(-1)
        return torch.nn.functional.normalize(x, dim=-1)[0].cpu().numpy(), valid

    def binding(self):
        return dict(checkpoint_sha256=sha(self.checkpoint), manifest_sha256=sha(self.manifest_path),
                    normalizer_sha256=self.manifest["normalizer"]["sha256"], device=self.device)


def audio_signals(path):
    signal = extract(path, read("src/dunhuang/research/phase_r45/config.json"))
    grid = time_grid(len(signal["onset"]), signal["sample_rate"], signal["hop"])
    drift = float(np.max(np.abs(grid - np.arange(len(grid)))))
    if drift > 1e-8:
        raise ValueError("Audio feature frames are not on the registered 30 Hz clock")
    wave, sr = sf.read(path, always_2d=True)
    mono = wave.mean(1)
    energy = []
    for t in range(len(grid)):
        frame = mono[round(t*sr/30):round((t+1)*sr/30)]
        energy.append(float(np.sqrt(np.mean(frame**2))) if len(frame) else 0.0)
    energy = np.array(energy)
    signal["energy"] = (energy / max(float(np.quantile(energy, .95)), 1e-10)).clip(0, 1)
    signal["grid_drift_frames"] = drift
    return signal


def interpolate(values, valid, q, radius=0):
    """Interpolate only when both contributing samples have genuine support."""
    q = np.asarray(q, dtype=float) - radius
    if not len(q) or not np.isfinite(q).all() or q.min() < 0 or q.max() > len(values)-1:
        return None
    a = np.floor(q).astype(int)
    b = np.ceil(q).astype(int)
    if not (valid[a] & valid[b]).all():
        return None
    frac = (q-a)[:, None]
    return values[a]*(1-frac)+values[b]*frac


def cosine(a, b):
    norm = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    return float(np.mean(np.sum(a*b, axis=-1)/np.maximum(norm, 1e-12)))


def multiscale(energy, activity):
    correlations = []
    for stride in (1, 4, 12):
        n = len(energy)//stride
        if n < 2:
            continue
        a = energy[:n*stride].reshape(n, stride).mean(1)
        b = activity[:n*stride].reshape(n, stride).mean(1)
        a, b = a-a.mean(), b-b.mean()
        correlations.append(float(a@b / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12)))
    return float(np.mean(correlations)) if correlations else 0.
