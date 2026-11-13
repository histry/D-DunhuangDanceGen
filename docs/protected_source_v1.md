# Protected source control, event information and E1-W Shadow integration

Baseline 55b96a7. All new logic is isolated in research/protected_guidance,
research/event_information and research/planner_integration. Existing src/config
files, frozen EDGE, sampling, Native and default Planner remain byte-identical.

## R5-P0 registration
Reuse exactly four prior Train source pairs, four exposed Val pairs, three times
999/499/19, the same q(X_A) Train noise and actual saved C2-A Val trajectory states.
Use only provided observed reference poses for guidance. Other/hidden poses are
evaluation-only. The reference loss, legal identity map and 1% condition L2 budget
remain unchanged. Condition perturbations may shrink within this fixed upper bound;
never renormalize a constrained candidate back to the bound.

Build at most six orthonormal Krylov directions using JVP/VJP of current predicted
Root, predicted contact and non-target rotation tangent readouts. The 76,800-column
dense Jacobian is never built. Solve a convex nearest-B1/B2 coefficient problem in
this finite subspace using fixed SciPy SLSQP maxiter200/ftol1e-10. Exact coefficient
constraints are root RMS, contact infinity, mean non-target tangent norm and
condition L2 budget. A pre-registered 0.9 linear safety fraction tightens the local
bounds; it does not relax the original nonlinear protection thresholds.
No relinearization, amplitude search or Val parameter selection is permitted.
Unsuccessful solves are UNKNOWN and produce zero control; they do not prove
infeasibility of the entire condition or latent space.

Compare B1, B2, PB1, PB2, norm-matched shrunk B1/B2 and norm-matched random PB1/PB2
controls. Maximum 384 case/time/source readouts and96 protected solves.
Check projected directions with JVP/VJP, three central-difference scales and their
independent basis linear-combination response. Root<=.01m, contact max<=.05,
non-target mean<=.01rad, activity [.9,1.1], acceleration/jerk proxy ratio<=1.1
remain the exact actual-forward acceptance gates.
Report joint source/directional/protection rates, subspace norm loss, linear-model
mismatch and solver unknowns. All-case Train qualification still requires24/24,
numeric trust and superiority to matched random. Fixed Val t19 has20 Native audits:
four common baselines plus PB1/PB2 for eight source conditions; independent replay
repeats the same cases and checks60 motion/FK arrays. These are single-step x0
predictions, never full generated dances. Full-chain guidance and mapping training
are not registered in this round.

## A2-P0 registration
Reuse canonical 99,552 real motion source frames, current roles, known six Val
music groups and raw labels. Freshly decode all registered AIST raw sources and
require exact equality to their sealed native arrays; use decode's actual upper8
observation mask. Keep old74D normalization immutable. Bind new pose48 and phase
statistics to Train full recordings once each, with independent masks.

F0 uses old dynamic74 and hold-centre +/-2 labels. F1 uses the same input with the
actual half-open detected hold interval. Their target definitions differ: an AP
difference is NOT an architecture gain. Speed-peak centre targets stay unchanged.
F2 adds observed upper8 pose48; F3 additionally adds observed elapsed/future times
to full-source motion-onset/direction-turn events (clipped3s). Hold and speed-peak
events themselves are excluded from feature channels to avoid feeding targets.
F4_pose/F4_time permute the corresponding blocks within source recording.

This first audit fits no encoder or classifier weights. Use a fixed5-nearest-
neighbour reference bank, uniform24 frames per Train recording,8 Train queries per
recording, and all fixed Val frames. Train queries leave the entire music group
out. Distances average squared Train-standardized differences only over pairwise
observed support, with equal block weights. Tie-break by frozen bank order.
Music-group AP/F1 and1000 paired group bootstraps are descriptive exposed-DEV
readouts. Shared choreography and unknown prior exposure remain public.
A mathematical static-pose witness demonstrates74D non-injectivity for pose
identity while explicitly showing BOTH static poses are holds. It cannot imply
pose information is necessary to detect stillness. No static-identity proxy is
advertised as real semantic correspondence. No new SDMR training is registered.

## Planner-P0 registration
Use the same499 event IDs, eight sources, four actual music timelines and49 slots.
Compare E0 legacy, E1 legacy and E1-W. Candidate admission must match the frozen
E1-W eligibility arrays, without changing Core, source anchor or rate limits.
Minimal research beam selection is checked against actual default E0 planning.
The selected E1-W events are then consumed by the real Planner's explicit fixed-
event research call. The default caller remains unchanged and heuristic.

Save every candidate UID, source/hash, exact map, final Event Plan SHA and semantic
identity; build actual native and scaler-conditioned prompt arrays, verify
independent source sampling, overlap identity, phase order, upper8-only masks and
UNKNOWN Root/contact support. Learned retrieval is not promoted and no EDGE
generation or Native calls are made by this integration test.

## Execution and evidence
Push code/config/tests before SSH execution. Use fresh study/receipt roots under
tmux with scripts/run_protected_source.sh. Full regression, EDGE checks and all
three fresh-process replays must finish before sealing:
scripts/export_protected_source.py --out STUDY --receipt RECEIPT --seal.
Failed/interrupted attempts stay intact. Publish aggregates, exact execution SHA,
hashes and receipts; private arrays, identifiers and raw data stay off Git.
Credentials, SSH clients and loopback GitHub relay stay in RAM and close in finally;
no SSH helper files are required. Existing modified reference-repository documents
must remain untouched. Operational qualification stays false.
