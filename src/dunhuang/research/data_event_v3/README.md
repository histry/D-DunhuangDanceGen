# Data V3 / ChangE E1 isolated study

Entry: python -m dunhuang.research.data_event_v3.study (see docs/data_event_v3.md).
This package is not imported by pipeline.py or the default generation planner.
registry.py creates versioned task role ledgers; study.py independently decodes raw
solo sources and compares E0/E1 on frozen Core boundaries and real music slots.

Reusable deterministic admission / identity / phase operations live in data/.
No optimizer, learned encoder, statistics fitting, source selection by Native,
or generation is performed. The V2 event catalog remains a non-routing shadow.
Old training / split heuristics stay in research/legacy for historical replay.
