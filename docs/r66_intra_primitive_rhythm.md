# R66-B: fixed Core internal rhythm

This research sidecar compares the same source interval at the same240 actual
target frame centers. Uniform, frozen E1-W and Rhythm form the first visible
delivery. ShuffledRhythm is then run with the identical algorithm and fixed
search budget. The previously selected event identities, music intervals and
48-frame transition budget never change. Transitions are not regenerated here.
Changed Core endpoint velocities do not inherit earlier Bridge qualification.

The twelve identity-selected positions cover eight unique source intervals and
three historically exposed music tracks. Four positions repeat a source with
different music. They are not twelve independent recordings. No new Planner,
SDMR, Diffusion or neural training is invoked.

Rhythm uses a bounded finite beam over source phase anchors and nearby audio
beat/accent candidates. The source-to-target PCHIP is monotone and C1; C2
smoothness is not guaranteed. Its derivative must remain within [.75,1.33]
across analytic per-piece extrema and a dense grid. Actual source positions are
obtained by numerical inversion of that curve, not piecewise-linear inversion
of its control points. Both full source support and the exact240-frame budget
are enforced. The inverse position speed has the reciprocal limits.

The fixed objective adds no learned weights: half beat plus half accent reward,
minus .05 normalized rate deviation and .05 normalized rate curvature. These
are registered engineering settings, not validated perceptual trade-offs.
The shuffled control shifts local proxy events by120 frames modulo240; it is
not a label of artistically incorrect dance. Uniform is an explicit candidate
and remains separately reported; missing evidence or illegal maps are retained
instead of being silently called a successful optimization.

Evaluate all outputs against the same true music clock. Mapped source-peak
distance measures the optimized technical task. Re-detected output peaks,
body22 FK and SO3 dynamics, geometry range, path length and endpoint jets are
additional diagnostics, still not independent artistic truth. Zero resampling
reconstruction error merely checks sampler conformance. Root/contact/Native24,
semantic stage correctness and human quality remain NOT_EVALUABLE.

Publish code, synchronize its SHA and use tmux:

```
PYTHONPATH=src python scripts/rhythm_r66.py main
# Deliver and inspect the main candidate before subsequent controls.
PYTHONPATH=src python scripts/rhythm_r66.py controls
PYTHONPATH=src python scripts/rhythm_r66.py verify --out outputs/rhythm_r66_01/replay_main
PYTHONPATH=src python scripts/rhythm_r66.py verify --replay-mode controls --out outputs/rhythm_r66_01/replay_controls
```

R66-A is a separate paired-clock validity experiment. Neither its outcome nor
the optimized R66-B proxy reward grants a new Planner operational qualification.
R65 independent review records and its frozen anonymous package remain intact.
