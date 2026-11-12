# Research and baseline boundaries

Production entrypoints remain scripts/generate.py, planning/worker.py and edge_backend.py. Frozen EDGE, Native-Legacy-v1 and the default legacy Adapter visibility are unchanged.

New experiment implementations live under src/dunhuang/research/source_identifiability:
- g2: same-architecture Adapter retraining, exact-source key supervision and corrupted-identity negative control.
- a2: canonical audible-clock coverage and separately registered source-association challenge.

Run these as Python modules with PYTHONPATH=src, not duplicated Python wrappers. Runtime mathematical operators (audio clocks, source-following geodesics and event-condition consumption) are shared and tested separately from the study drivers.

Historical generation trainers have moved from generation/{train,event_transfer,root_experiment}.py into research/legacy/{paired_training,event_transfer,root_experiment}.py. All active callers and their subprocess module names have been updated. generation/targets.py retains only target-coordinate conversion and parameter hashing shared by data preparation and experiments. Historical scientific artifacts and prior failed studies remain available; their exact original code can also be checked out by recorded Git SHA.

Four redundant four-line script wrappers were removed; their launchers now use python -m. No dataset, checkpoint or scientific receipt was deleted. No compatibility layer imports old training drivers into the production generation package.

Observed-trained G2 weights require a matching research runtime. Legacy consumers reject these checkpoints instead of silently applying legacy visibility. The new experiment checkpoints are unqualified and never replace the default generator.

## A2-V3 / G2-M source necessity

The study in `src/dunhuang/research/necessity` reuses frozen operators and runtimes without changing existing source or configuration files. Its separate `configs/source_necessity_v1.json` registers Train-only candidate support, two source-time tasks, exact self-recording training targets, a matched 2x2 supervision/time-step experiment and an occurrence-preserving source-shuffle control.

Task qualification is limited to exposed development mechanism research; it does not promote generic Data V3 training, independent semantic evaluation, default retrieval, generation or G3. Main runs and fresh optimizer-free replays use separate new output roots. See [protocol](../docs/source_necessity_v1.md).
