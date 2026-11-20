# Independent music-event candidate branch

This branch runs fixed official Beat This! and All-In-One checkpoints on complete audio and keeps acoustic candidates separate from learned observations. It does not change Planner, E1-W, Fixed48, G1, SDMR or Native qualification.

The catalog retains source audio hashes, original seconds and sample positions, floating 30 FPS positions, rounded indices and separate support flags. Model probability is not calibrated accuracy. Cross-model agreement is consensus, functional sections are not independently confirmed phrases, and absent phrase evidence stays UNKNOWN.

Execution order (from repository root, deployed music_events_01 environment):

```bash
export PYTHONPATH=src PYTHONUTF8=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python -m dunhuang.research.music_events.infer
python -m dunhuang.research.music_events.view
# Actual browser acceptance must bind receipts/preview_browser_acceptance.json to main/manifest.json.
python -m dunhuang.research.music_events.infer --replay
python -m dunhuang.research.music_events.paired
python -m dunhuang.research.music_events.saved_eval
```

Use tmux and preserve process exit codes. `infer --replay` is a fresh full-model replay and reports exact scientific-file differences without hiding floating-point differences. `paired --prepare` runs in the historical motion environment to retain the sealed source and legacy-audio contracts; `paired` runs in the isolated music environment only after preview acceptance. The six predeclared Train audio groups are historically exposed diagnostics, not a new generalization benchmark. Missing sources are retained without replacement.

`deploy.sh` uses a separate venv over the existing judge environment and installs overrides only into that venv. The process-local NATTEN compatibility aliases implement the deprecated official API over the current native operations; official model source/weights remain unchanged. Legacy trusted official checkpoints require the explicit PyTorch compatibility flag above. Do not use it for untrusted checkpoints. Model locks, dependency revisions, timing, GPU-memory scope and failed deployment logs remain in private output receipts.

The viewer uses one original-audio clock and offers seek, stepping, event overlays, optional clicks and an exportable candidate-review form. Its visible automatic candidates make these exports assisted reviews, not independent blind annotations. Public evidence excludes audio, weights, stems and activation arrays. Human music quality, phrase accuracy and Native24 remain NOT_EVALUABLE without suitable independent evidence.
