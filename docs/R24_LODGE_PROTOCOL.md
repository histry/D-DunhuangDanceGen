# R24: ChangE reference control becomes the main research route
EDGE and R23 stay frozen comparisons. No default generator swap; retain original Lodge9+0/checkpoint.
Reference-only compiler and experiment live in research/lodge_reference_r24. No copied Lodge Global planner.

## Fixed population and conditions
Use sealed R8 pair0 (female_FeiTian / male_pipa_1), two source-disjoint full ChangE Cores with a common legal E1-W duration96. Do not select by Lodge outputs. Require mean upper8 source distinction≥0.15rad before sampling; otherwise stop, never replace the fixed pair automatically.
Use first256 frames of original dunhuangwu2/3 music from sealed R8 metadata. Native35D features are re-extracted from real audio, not EDGE4800D.
Two fixed seeds24041/24042 × two musics × no reference/A/B =12 trajectories.
All six conditions per seed use the same native inpaint_soft_ddim and original guidance ramp. No-reference is a zero mask, never a different sampler.
Official soft sampler requires batch>1. Use one six-row batch per seed; intercept only state-shaped randn/randn_like to broadcast the same seeded Gaussian draw across six rows at every iteration. This common-random-number control changes noise coupling, not marginal sampling equations, model or schedule. Record50 draws and their SHA. Repeat in an isolated fresh process, compare all arrays exactly.

## Full Core and phase compilation
Both full source Cores are mapped using their frozen E1-W maps to96 target frames inside[64,160); exact source durations come from sealed plans and must be reported, not inferred from labels.
22 fitted body rotations are observed; source Root translation, contacts, terminal hand rotations and fingers are unknown.
Only true first22 rotations enter cues. No Root or C4 mask. Outside the Core the source boundary poses are held for display; unobserved values are zero-masked before sampling.
Choose strongest non-truncated onset/speed-peak/direction-turn/root-turn proxy from each event, form the union of their ±1-frame windows and the two Core endpoint frames. Same cue times/mask for A/B; fewer than half Core frames prompted. Full source phases and order remain available, but phase numbers are not a newly trained neural condition.
Report mapping/axis validity and Normalizer roundtrip loss, especially prompted rotations. Do not call proxy phases human semantic truth.
FineDance skeleton/body geometry is preserved in actual sampling. Native24 remains NOT_EVALUABLE; no fabricated terminal rotations/fingers.

## Metrics and finite decision
Report source retention separately on cue and unprompted Core, to both candidate references, including upper8. These are retimed-reference retention proxies, not paired target errors for the new music.
For each seed/music require both references to be closer to their assigned event (whole22 and upper8), improve unprompted assigned error by5% relative to zero-mask baseline, and keep common22 Jerk P95 within1.1×baseline. Four groups, no outcome-driven cue retuning.
Record music-induced change under identical noise; nonzero response alone is not perceptual music coordination.
Report common22 world Jerk/activity and root-relative velocity spectral energy≤3Hz/>6Hz using a Hann window. This metric-only decomposition does not smooth videos or establish naturalness.
If zero-training control is insufficient, preserve failure evidence and design the next separately registered global/local reference adapter; do not fabricate paired ChangE–new-music supervision or choose a training budget after viewing results.
No actual-history, future Root, tracker distillation or whole-song runs this round. Human review is not a prerequisite.

Original native sampler and architecture are pinned in inputs/lodge_r22/runtime_manifest.json.
Source reference: [Lodge official](https://github.com/li-ronghui/LODGE/tree/f38642ad67bc3402444f5f914010a911e5bd5c89).
