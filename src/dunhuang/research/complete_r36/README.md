# R36: complete-source proposals and bounded reference editing

Research entry points live here; the default EDGE path, SDMR routing and frozen R35 code stay unchanged. The new source units are kinematic proposals, not expert-labelled complete Dunhuang actions.

1. `python -m dunhuang.research.complete_r36.study --out outputs/complete_r36_01/planning`
2. Repeat in a fresh process with `--replay-from` and a fresh destination.
3. `python -m dunhuang.research.complete_r36.view --root outputs/complete_r36_01 --out outputs/complete_r36_01/viewer_plan`

The library preserves the old 499 entries and original assets. It builds longer same-recording units without crossing quality-rejected gaps, protects long low-speed phases, keeps geometric families and retains variants. The joint planner chooses source unit and variable-duration music interval together. Music structure is audio-only and heuristic; SDMR does not route. Internal phase anchors obey ordered bounded time maps. Source and music support are half-open intervals, but map knots use actual frame centers, including the last real frame; there is no invented endpoint hold or source extrapolation.

The viewer uses one audio clock for all panels, supports stepping and timeline seeking, and distinguishes source arrangements from generated motions. Reference XYZ is displayed on a shared 22-joint rig with unknown world Root/contact. Offline playback needs a browser supporting DecompressionStream; a loopback-only read-only server can be used for QA.

This is inspired by multi-level motion organization and source-conditioned editing, not a reproduction of those papers: [HACA](https://www.f-zhou.com/tc.html), [ChoreoMaster](https://cg.cs.tsinghua.edu.cn/papers/SIG-2021-ChoreoMaster.pdf), [Lodge](https://arxiv.org/html/2403.10518v1), [MotionFix](https://motionfix.is.tue.mpg.de/).

No new unseen-song or cultural-semantic qualification is claimed. The three user-specified songs are historically exposed development cases. Full Native24 remains not evaluable for the 22-joint source contract. Additional ablations and whole-song generation remain zero until the actual short-window effects have been inspected.
