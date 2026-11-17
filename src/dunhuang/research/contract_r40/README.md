# R40: execution-contract research

The formal EDGE path, frozen R37 plans and R38/R39 baselines remain unchanged. R40 adapts only the existing Context branch, initialized from the R38 trained checkpoint. All bridge examples still explicitly use the existing ChangE Source branch; this is not dataset-aware routing.

Registered main comparison: Frozen R39, 240-update Extra (R38 conditional/raw-pose objective and executor), and 240-update Matched (actual CFG, R39 continuous Root, executed-body objective, real target D1/D2/D3). Both trained arms refresh their own actual sampler states at 0 and 120, with identical cases, seeds, batch order and update counts. CFG changes compute cost, which must be reported. This is a combined contract intervention, not component-wise causal attribution. Extra is deliberately deferred until the first Matched viewer exists.

Only real continuous paired motion is a target. Hidden gap truth is removed before Source feature/derivative/global-token encoding and Context construction. No RTS output is a training target. Root and body budgets and RTS scales remain frozen from R39. Val is historically exposed, not new blind evaluation. Cross-source examples have no unique continuous truth; 22 joints do not establish Native24, true contact or cultural semantics. No whole-song run is admitted by interface availability.

Watchable delivery precedes additional ablations. Actual source/final arrays, code SHA, checkpoints, refresh caches, gradients, seeds and receipts must be retained. Independent replay prohibits optimizer construction and reconstructs selected training gradients plus both cache refreshes and final sampler trajectories.
