# R23 actual call paths
scripts/run_r23.sh → local_r23.common.inputs authenticates R22 receipts/data/model seals.
- gradients → original branch_r22.objective.build → saved pre values → autograd.grad dot products; no optimizer.
- sample → original branch_r22.engine.load + context_r21.sampler.sample → FixedGuidance reads both live network outputs, returns clamp(unc+w(cond−unc)). Original CFG2 path is evaluated but its output is discarded for CFG1; same two main-net forwards. Actual-state snapshots are separate from q_sample.
- represent → frozen mechanism_selected → full-target least squares → fixed shared continuous decoder → actual Native/FK; no diffusion or prior learning.
- external → strict original Lodge model → normalization and wiring interventions; terminal position retarget separate from incomplete output admission.
- physical_tracking → source/G2 paired input and runtime inventory only; no verified PHC runtime is assumed.
- evaluate → all64 conditions → original Native, full-source retention, correct hard-domain templates.
- scripts/verify_r23.py → fresh-process arrays/results, all88 physical reports and144 bundles, frozen paths.
All code is under research; formal planner/EDGE/sampler and historical baselines do not import local_r23.
