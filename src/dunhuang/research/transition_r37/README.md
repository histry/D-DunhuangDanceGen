# R37: source-content memory and one shared transition

This isolated research module leaves R35/R36, the official EDGE baseline and SDMR routing untouched. Keep all raw sources and the original 499 atomic units. R36's exact 1,955-unit catalog and music boundary candidates are frozen inputs.

## First deliver viewable effects

`bash scripts/run_r37_first.sh outputs/transition_r37_01` runs source-interval-memory planning, a fresh exact planning replay, fixed-parameter postprocessing of already-saved R36 motion states, a fresh postprocessing replay, and the first viewer. Source pages contain full recordings and multiple candidate jumps, stop at the end by default, and offer explicit optional looping. Planning pages expose used original intervals, new-content frames, backward jumps and acoustic repetition evidence. Repetition evidence is not a cultural or artistic correctness label.

`bash scripts/run_r37_bridge.sh outputs/transition_r37_01` then consumes one boundary from the actual new plan and the fixed old failed song1 boundary. It compares discontinuous-reference R36-like editing, a nonlearned SO(3) quintic Hermite baseline and bounded single-Lodge modification of that shared geometric transition. The two fixed seeds are 35041 and 35042. Each reverse process has fifty updates from t=119. Source residual influence is smoothly suppressed in the common transition; the body outside it is explicitly protected. Unknown world Root and contact remain unknown. The model does not train a new bridge or automatically acquire phase semantics. This is mechanism-inspired editing, not a reproduction of DoubleTake or CondMDI, and not an isolated Context-weight ablation.

Same-state x0 proposals and their projected counterparts are retained for the identical four-frame boundary stencil on all fifty steps, both seeds. Never attribute a low core error to learned control when the core is copied or constrained. There is no complete two-event or whole-song qualification: a 256-frame computation window only contains portions of longer semantic units.

## Smoothing contract

Root is processed in Euclidean position/velocity state; rotation KF/RTS uses a right-error relinearized orientation/velocity approximation, with diagonal axis covariance. It is not an exact invariant Lie-group EKF. Residual RTS works in spatial parent-joint coordinates on each continuous source event independently, with SO(3) reconstruction, a C2 correction envelope and the original budget reapplied. It cannot fix discontinuities already in the reference. All parameters are fixed in config before output inspection; no visual or metric parameter sweep is allowed in this run. Time steps are physical 1/30-second motion frames, not diffusion iterations. Saved decoded motion is processed before fixed-skeleton FK; the viewer never secretly smooths XYZ. C4 is retained as a model output, not contact truth. Check source configuration, amplitude and peak timing alongside jerk, not jerk alone.

## SDMR interface scope

The shared versioned clock reader accepts declared actual-frame-centre maps and legacy exclusive-end maps, rejects unsupported versions, and drives content extraction, plan preview and generation. Derivatives are recomputed after SO(3) resampling. The new 274D observed-content interface retains 132D local rotations, 66D root-orientation-relative FK, 74D dynamic observations, progress and actual duration, with channel masks. A small dual encoder reconstructs pose and geometry from the exact tokens used for retrieval; it has no raw-pose decoder bypass. Local, song-level, ambiguous-overlap and unverified relationships stay distinct. Multi-positive training requires actual certified clock records. ChangE has no fabricated music pairings.

This release tests input identifiability, shared-token gradients and label contracts; it does not train new SDMR weights. The next training population, variable-unit batches, Train-only standardization, matched candidate evaluation and paired-vs-shuffled selection must be registered together. Random new training on the repeatedly exposed Val groups would not establish independent music-content benefit. SDMR `can_route=false` remains explicit.

## Validation and interpretation

Fresh-process replay compares all saved arrays bitwise and recomputes FK/budgets and region metrics in CPU code. Preserve failed packaging/execution receipts rather than overwrite outputs. Full Native24 remains `NOT_EVALUABLE_22_OF24`. No new optimizer updates, whole-song generations or additional training ablations are authorized by passing these software checks alone.

Methods referenced for design, not reproduced: [PriorMDM/DoubleTake official code](https://github.com/priorMDM/priorMDM), [Bayesian Filtering and Smoothing](https://www.cambridge.org/core/books/bayesian-filtering-and-smoothing/F88740E8D25010CF3119A5CA379FA37A), [SoulNet paper](https://arxiv.org/abs/2507.14915), [TMR](https://arxiv.org/abs/2305.00976). None proves this project's target-domain quality without its own results.
