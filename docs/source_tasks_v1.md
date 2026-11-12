# Source Tasks v1: saved source response, paired CF and content identification

Baseline acce669 is frozen except the explicit pipeline guard: a Dunhuang request cannot supply legacy learned_retrieval anchors which the Planner would overwrite. Whole-song retrieval must use its Planner alignment entrance. This rejects an ambiguous configuration before file access; valid default generation remains unchanged. E1-W integration, online learned ranking, G3 and physical optimization are not promoted.

## Registered data and stopping rules

All new study drivers are isolated in research/source_tasks. Existing necessity/source_identifiability/transfer_audit code and Native definitions remain unchanged. Frozen source_necessity_03 public hashes bind the exact Data V3 Train/Val source manifest, original masks, actual 151D/Jukebox data and four previously exposed evaluation music/source pairs. No Test tensors are read. Unknown work aliases and pretrained exposure remain UNKNOWN. New qualification only authorizes the explicitly exposed development study; generic Data V3 qualification and operational_qualified remain false.

P0 is read-only: confirm all52 previous native arrays still agree with their independent replay, compute actual A/B upper8 SO(3) output separation and divide by the true reference separation on common provided observations. Also report full-observed/hidden regions and directional source preferences separately. A large response is not necessarily a correct response.

P1 enumerates independent recording pairs with the same verified basic choreography, original audio SHA, physical clock and actual Jukebox bytes/hash. Each target remains its own real recording. Common provided reference difference must be >=.05rad. Require >=30 Train pairs /15 Train recordings /3 Train music groups and >=4 Val music /8 Val recordings; all four previously fixed evaluation cases must qualify. Validate existing raw/target/observed/Jukebox/clock hashes again.

## Common high-noise support

Use the official pinned EDGE beta operator AND its float32 torch cumulative product. Verify the resulting alpha buffer against the actual runtime before training.

For each real normalized pair compute the total conditional Gaussian mean KL:
KL_t = alpha_t * ||X_A-X_B||^2 / (2*(1-alpha_t)).
Only actual50-step DDIM times >=899 with SNR<=.02 and KL<=.5 are eligible. Require >=2 eligible times per admitted TRAIN pair. Val is measured by free generation, never by this corrupted-target task, so its original fixed cases require source/clock/music/observability qualification rather than training-corruption support. Record its noise diagnostic without filtering or replacing the fixed cases. Never change the numerical thresholds.

The first preflight implementation wrongly applied training-corruption eligibility to Val as well. It was stopped before training and preserved; a role-scope regression test corrects this without changing KL/SNR thresholds, sources, cases or budgets.

A shared state is sampled by choosing H in {A,B} using a fixed random schedule and drawing q(x_t|X_H) with the official forward operator. Both model branches receive the identical realized tensor and identical music. This is a mixture of two real-target forward distributions; it is not an arbitrary pure-noise input. Conditional KL is a registered support diagnostic, not proof that the task has the original denoising conditional distribution.

## G2-CF fixed comparison

Each arm has 400 AdamW updates, batch2 (one A/B pair), lr1e-4, gradient clip1, original357424-parameter Cross Adapter, same saved initialization, pair order, supported timestep and Gaussian noise. The backbone is frozen and its digest/gradients checked on close. Fixed last-step checkpoints; no Val selection.

C0: M3-style self-denoising objective on the SAME new paired population and supported timesteps. A/B each use their own q state with the same Gaussian. This is a matched control, not the historical M3 weight.
C1: two real targets with one shared q-mixture state; no CF objective.
C2: C1 plus bidirectional CF preference hinge.
C3: C2 with A/B reference geometry exchanged while targets stay original. Both sources occur exactly once in each batch; the source marginal is exactly preserved.

Keep MSE +.1 velocity +.02 rotation +.05 source reconstruction and weight2 following on real observed stage masks. CF uses squared SO(3) distances, margin=min(.25*mean source squared-angle difference,.01rad^2), weight2. CF reads only common provided upper8 observations; unknown reference storage is sanitized before rotation decoding. Record a finite nonzero first-step CF adapter gradient for C2/C3. No hard copying, new network, root/contact pseudo-truth or NULL-centering branch.

Evaluation uses the same frozen4 Val cases and old seed20261113: bypass, C0-C3 each A/B, sparse hard keys A/B =44 actual generations, with unmodified Native and old directional>.001rad, gain>0, activity>=.9 criterion. Source-response ratios and hidden-frame diagnostics remain auxiliary. Fresh process loads fixed models and repeats all44 generations; compare132 normalized/native/FK arrays and44 physical JSONs exactly, with optimizer constructors forbidden.

## A2 content task, independent of the G2 gate

Reconstruct50D audio at exact fractional source clock and74D motion derivatives computed on full recordings before cropping real120-frame cores. Reuse independently raw-rebuilt audio/native sources and verify their hashes. Bind a NEW normalizer artifact to existing Train-only full-source statistics and new source registry identity; inherit no old encoder weights or old normalizer.

Freeze both existing A2-V3 candidate matrices and all nuisance/clock scores. Their selectors use complete paired nuisance data and remain OFFLINE research tasks, never an online query algorithm.

Compare fresh 15Hz, 30Hz, shared-weight dual rate, and dual rate with music/source association permutation. All have exactly equal parameter counts and identical initial state. Dual calls the same Transformer twice, so more computation is explicitly not equal FLOPs. Deterministic dropout0 for all; physical sinusoidal positions use seconds, not sample index.

Each arm400 AdamW updates/batch4/lr1e-4/clip1: alternate the two tasks,200 updates each, source-group-balanced schedules. For shuffle, exactly preserve sampled music occurrence marginals while prohibiting same music/choreography/recording associations. Score global256D cosine plus .5 aligned local128D cosine on common120-frame support; train candidate CE at temperature.07. No semantic/preference heads or artificial event ground truth.

Save fixed last-step checkpoints with task/normalizer/code binding. Evaluate only identical Val candidates. Report paired music-group deltaR1/deltaMRR against frozen nuisance and clock controls,1000 group bootstraps, seed20261113. Keep all4 model results; no best-on-Val checkpoint. A same-clock different-music feature-swap diagnostic uses exactly the comparable query subset and reports its own population; it is not energy preserving. Only6 known Val music groups exist, so no blind generalization claim. Positive offline score gains do not promote online retrieval.

## Execution and release

Push source/config/tests/protocol to GitHub first; server fetch verifies exact commit before:
tmux new-session -d -s ddg-source-tasks 'bash scripts/run_source_tasks.sh outputs/source_tasks_01 outputs/source_tasks_01_receipt > outputs/source_tasks_01.log 2>&1'

Run regression and EDGE supplement, P0 plus fresh recompute, P1 plus fresh qualification replay, independent content observations replay, qualified CF and content training, then frozen-model replays. A closed G2 gate does not close the independent A2 task. Trap launcher exit code and UTC times. Seal aggregate release only after exit_status=0. Preserve all failed attempts and private identities/weights on server. No disk SSH helper is needed; close in-memory connections/relay and remove only helpers created this run.

Literature boundary: the supplied CVF OpenDance/CAV-MAE pages returned403 in this check, and PubMed42685165 provided no verifiable body. These studies are independently specified mechanisms, not claimed reproductions of those papers.
