# SDMR-E2 P2 fixed comparison

The original P1 Train gate passed. P1 exposed Val has already been reported; it is
not used for any P2 setting, checkpoint selection or budget change. All final
240-step checkpoints are reported, including failures. No Test tensors are read.

NoRank, Rank and ShuffledRank share the same frozen R41 initialization (522632
parameters), Train normalizer, energy-matched fixed candidate lists, schedule,
AdamW lr 0.0003, clip 1, and 240 updates. At each step length cycles 90/150/240;
up to four distinct Train music groups contribute one query each. Same-length
complete real subwindows are used, never padded. R41 auxiliary losses retain
global 1, clock .1, local token .1, full-parent observed reconstruction .1 and
equivariance .05. Their original real parent-context crops are preserved.

Rank adds masked stride-3 diagonal multi-positive logsumexp loss, temperature .1,
weight 1. NoRank computes the same branches with rank weight 0. ShuffledRank
permutes the registered positive mask independently of the input, preserving
positive counts; these are counterfactual training labels, not artistic truth.
This controls model calls/updates, not a claim of perfectly identical GPU FLOPs.

The fixed schedule is generated from seed 43043 before the first update and
saved per arm. No dynamic hard mining or Val selection. Snapshots before/after
steps 0/120/239 and full shared-input gradient vectors for all six objectives
are preserved. A fresh process replays those losses/gradients with zero optimizer
calls and verifies actual weight changes. All three final checkpoints are evaluated
on the same 433 tasks with actual, both-zero and both feature-swap controls.
Global and Diagonal R@1/R@5/MRR are reported by music group. Frozen P1 Duration,
Energy and partial-path comparisons remain the baseline; they are not retuned.

Scope remains physical source-clock subwindow ranking, not semantic event truth.
Benchmark forced positive/negative candidate construction is an offline diagnostic;
it is not deployable target-domain selection. Related candidate pools are not
independent replications. Native ChangE shadow labels remain UNKNOWN and routing
stays disabled. Module II's failed training gate is unrelated and stays failed.
