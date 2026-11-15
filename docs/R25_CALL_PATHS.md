# R25 real data and neural call paths
scripts/run_r25.sh → regression → data + fresh raw/audio/feature replay → zero-head graph check → event/phase actual training + optimizer-free saved-update replay → trained graph interventions →124 sampler outputs + fresh sampler replay → metric replay.
ReferenceAdapter.set_reference: validity-masked joint9D + flag → shared joint MLP/joint ID →5 body parts →64D sequence →2 temporal convs/global pooled token → per-layer K/V.
Actual Lodge decoder stack layers1/4/7 forward hooks: hidden512 → Q64 (+phase query) →global/local reference attention →zero-initialized output512 →present-reference residual. No reference means identity; no x_t reference writeback.
Training uses actual model.forward with known diffusion t; the frozen no-reference teacher is computed with matched music-dropout state. Reference dropout is a separate explicit validity change.
Sampling uses actual official guided_forward and inpaint_soft_ddim; all constraint masks zero for trained adapters, keeping R24 sparse outputs as independent comparisons. The same full reference memory is available to conditional/unconditional MUSIC branches.
Stage proxies/target clocks are source-derived condition channels; no unpaired new-music target labels or expert semantics are invented.
