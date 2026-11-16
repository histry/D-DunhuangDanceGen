# R36: complete-source proposals and bounded reference editing

Research entry points live here; the default EDGE path, SDMR routing and frozen R35 code stay unchanged. The new source units are kinematic proposals, not expert-labelled complete Dunhuang actions.

1. `python -m dunhuang.research.complete_r36.study --out outputs/complete_r36_01/planning`
2. Repeat in a fresh process with `--replay-from` and a fresh destination.
3. `python -m dunhuang.research.complete_r36.view --root outputs/complete_r36_01 --out outputs/complete_r36_01/viewer_plan`

The library preserves the old 499 entries and original assets. It builds longer same-recording units without crossing quality-rejected gaps, protects long low-speed phases, keeps geometric families and retains variants. The joint planner chooses source unit and variable-duration music interval together. Music structure is audio-only and heuristic; SDMR does not route. Internal phase anchors obey ordered bounded time maps. Source and music support are half-open intervals, but map knots use actual frame centers, including the last real frame; there is no invented endpoint hold or source extrapolation.

The viewer uses one audio clock for all panels, supports stepping and timeline seeking, and distinguishes source arrangements from generated motions. Reference XYZ is displayed on a shared 22-joint rig with unknown world Root/contact. Offline playback needs a browser supporting DecompressionStream; a loopback-only read-only server can be used for QA.

This is inspired by multi-level motion organization and source-conditioned editing, not a reproduction of those papers: [HACA](https://www.f-zhou.com/tc.html), [ChoreoMaster](https://cg.cs.tsinghua.edu.cn/papers/SIG-2021-ChoreoMaster.pdf), [Lodge](https://arxiv.org/html/2403.10518v1), [MotionFix](https://motionfix.is.tue.mpg.de/).

No new unseen-song or cultural-semantic qualification is claimed. The three user-specified songs are historically exposed development cases. Full Native24 remains not evaluable for the 22-joint source contract. Additional ablations and whole-song generation remain zero until the actual short-window effects have been inspected.

`train` makes one fixed 160-update main candidate from the frozen R35 Context checkpoint. Diffusion time, sampled/actual state, source, provenance, missing side and gap are independently drawn from a Cartesian design. Generated-context values are real saved model outputs; planned context is a supplied same-recording reference, not fabricated cross-source truth. All observed ChangE body frames are supervised, including the read and edge regions; unknown Root/C4 use a teacher regularizer and are not zero-filled targets. The initial paired cache is refreshed using the candidate's planned context, while ChangE actual states remain the recorded parent cache. This limitation is explicit; it is not fully on-policy training.

`generate` compares the exact stored R35 Frozen/Context arrays with one corrected-Context plus source-editing candidate on the same three pre-registered windows and two seeds. It uses the same row-noise stream but a low-noise start at t=119, and fifty DDIM updates. Rotations are edited as R_ref Exp(delta), bounded on every x0 prediction; the declared read-only neighbor body is preserved explicitly. Root/C4 are initialized from a model proposal and updated jointly, never labelled source truth. Source retention is partly enforced, not proof of learned editing or music responsiveness. This combined main effect is not a causal ablation. Independent replay re-computes training gradients with zero optimizer steps, then reproduces generated arrays in a fresh process.
