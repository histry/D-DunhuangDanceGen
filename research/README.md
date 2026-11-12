# Research and baseline boundaries

Production entrypoints remain scripts/generate.py, planning/worker.py and edge_backend.py. Frozen EDGE, Native-Legacy-v1 and the default legacy Adapter visibility are unchanged.

New experiment implementations live under src/dunhuang/research/source_identifiability:
- g2: same-architecture Adapter retraining, exact-source key supervision and corrupted-identity negative control.
- a2: canonical audible-clock coverage and separately registered source-association challenge.

Run these as Python modules with PYTHONPATH=src, not duplicated Python wrappers. Runtime mathematical operators (audio clocks, source-following geodesics and event-condition consumption) are shared and tested separately from the study drivers.

Historical generation trainers have moved from generation/{train,event_transfer,root_experiment}.py into research/legacy/{paired_training,event_transfer,root_experiment}.py. All active callers and their subprocess module names have been updated. generation/targets.py retains only target-coordinate conversion and parameter hashing shared by data preparation and experiments. Historical scientific artifacts and prior failed studies remain available; their exact original code can also be checked out by recorded Git SHA.

Four redundant four-line script wrappers were removed; their launchers now use python -m. No dataset, checkpoint or scientific receipt was deleted. No compatibility layer imports old training drivers into the production generation package.
