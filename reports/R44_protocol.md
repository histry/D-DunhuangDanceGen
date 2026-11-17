# R44 preregistered zero-training study

Baseline: 5e0506c2f29ca6fb8600620c60affed641cab34f. Formal Planner/EDGE/Native and all parent outputs remain frozen. Research code is isolated under reperformance_r44 and retrieval_audit_r44. No optimizer calls, new network samples, Test tensors or whole-song generation in A/B. No automated label becomes human truth. Native24 remains NOT_EVALUABLE for 22-joint / unknown world Root and contact.

## Primary visible result before controls

Same 22 R43 cases and seeds 35041/35042. Geometry is deterministic: its duplicate seed tracks are not independent trials. G0 saved Geometry; G1 original E1-W piecewise clock; G2 C2 quintic clock; G4 saved R43 Core failure. G1/G2 use the same SO(3) RotationSpline and Root CubicSpline interpolants, with exact integer-anchor restoration. This interpolant change is explicit. Root is the inherited model prior, not ChangE world truth. Core intervals [8,99),[158,248); outside, including Bridge [104,152), byte-exact saved Geometry. No hidden evaluation target is used to derive clocks or phases. Peak proxies come from visible Geometry Core only; no semantic phase claims. No extra motion is inserted into a static source.

Audio beats: pinned librosa 0.11.0 runtime, 22050 Hz, hop735; phase proposals reuse E1-W. G2 phase displacement <=6 actual frames, integer knots, source/music derivative in [1/1.33,1/.75] certified at all polynomial extrema. Each knot uses d1=1,d2=0. Continuous angular rate/acceleration is not a jerk guarantee. Report actual 30Hz FK Core, boundary and gap D1/D2/D3, activity/range, pose-anchor error, phase order and protected exactness, including meditation.

Primary viewer uses saved arrays and one audio clock; fixed camera/scale, seek/pause/rate/single-frame/clock-map inspection. All 22 cases, not metric-selected best cases. First return and inspect this effect before later controls. Then G3 fixed map + swapped playback must produce identical motion; WrongMapB rebuilds legal maps from alternate real music; NoMusicTime is identity. Music response in this experiment is explicit retiming, NOT new learned music consumption. RMS waveform energy peaks are a separate, imperfect timing proxy, not real event correspondence.

## A: population, cost and hubness

First audit frozen R43 cost units and actual Native23x24 path coverage/ties/source availability. 1-sim match vs two 0.3 drops has local indifference sim=.4, not a full-path necessary/sufficient rule. Preserve strict one-to-one matching; do not call it full Drop-DTW.

Use the immutable 10,107-row R43 index. Select two raw recordings per all 48 Train music groups and a metadata-middle window at each 90/150/240 length; preserve original Val tasks unchanged. No score-based sampling, no Test. Enumerate positive/negative/UNKNOWN and missing support; same-raw exclusion and role isolation checked. Labels indicate official physical audio-clock correspondence, not artistic preference. Shared choreography is reported and never called unseen-choreography generalization.

Train groups deterministically hash-split 24 calibration / 24 within-Train probe before scoring. Calibration: per-group negative diagonal-token similarity 95th percentile, then median across calibration groups; set drop=(1-threshold)/2 clipped [.05,.75]. No coverage maximization or Val selection. AUC/TPR/FPR on probe determines whether calibration is informative; otherwise remains diagnostic. Original Global/Diagonal/Offset/Partial remain frozen. Train-bank hubness diagnostic uses k=10 negative cross-music calibration queries and subtracts candidate bank affinity; not a full BeatDance reproduction. All ties counted fractionally, report availability-normalized source concentration. Native scores have no target-domain accuracy ground truth.

## Conditional training

Do not train a new neural branch just because legal maps exist. Necessary Train-only screen: >=3/4 ChangE Train cases preserve activity ratio [.85,1.15], Core D3 <=1.05 Geometry; protected/gap exact; paired Train music timing proxy improvement over both WrongMapB and NoMusicTime has music-group bootstrap 95% lower bound >0. A positive screen alone does not prove semantic music expression: independent real paired clock/feature compatibility must also be present before neural training. Missing credible benefit stops at A/B; do not launch large training or songs. Any follow-on training needs its own fixed protocol before execution.

## Evidence

Publish code before each server stage. tmux launcher, exact SHA, source hashes, success/failure receipts; fresh-process numeric replay with zero optimizer calls; parent manifests unchanged. Preserve failed runs. Clean only temporary helpers created for R44. Public reports exclude private raw arrays/music/checkpoints; locally return this run's viewer files explicitly requested by user.

Mechanism references: SciPy RotationSpline and BPoly.from_derivatives official docs; BeatDance ICMR2024 official https://github.com/XulongT/BeatDance ; Drop-DTW NeurIPS2021 official https://github.com/SamsungLabs/Drop-DTW . No claim to reproduce either network or full algorithm.

CPU审计/视频的可选依赖可通过 `pip install -e ".[r44,test]"` 安装；默认生产依赖不引入Numba。特征提取和官方FK仍使用已核验的EDGE/Lodge运行时，音频检测仍使用资产清单中的librosa0.11.0，不应据此改动已封存实验环境。
