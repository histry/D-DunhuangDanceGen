# R66: actual paired source-clock validity

R66 is an isolated research sidecar. It reads the sealed R59 registration and
raw assets; it never changes R52/R64/R65, invokes a Planner, changes proxy
weights, or trains a network. Fixed48 remains the operational baseline.

The population, audio detector, motion peak extractor, five shifts, RMS/tempo
donor rule, tie handling, cluster unit and missingness handling are frozen in
the code commit before scoring. All 338 original queries remain registered.
Each comparison uses the same inner 120 motion frames. Peaks are extracted
before cropping from the complete original recording. Raw audio is read at
motion_frame + official_source_offset + delta. Positive delta reads later
audio. There is no cyclic wrap, padded support, or injected-motion recovery
sign reversal.

The exact R64 Gaussian beat/accent reward is evaluated, together with its two
components, raw RMS versus angular-speed correlation and a density-only tied
control. No feature, threshold, score direction or population is selected from
the observed outcome. Wrong-song donors are same-domain/same-role and selected
by metadata hashes, then by local RMS and median inter-beat interval only.
Failure to find a supported donor remains missing, with no candidate expansion.
Wrong-song discrimination is only a diagnostic: different music is not proof
of artistic incompatibility.

Report window -> raw source -> music group balanced results, cluster-bootstrap
intervals, ties and all missing/registered denominators. Train, historically
exposed AIST Val and FineDance official Genre Train diagnostics are separate.
FineDance common-origin evidence is not waveform-calibrated AIST clock truth.
No new music, unseen choreography or pretraining-independence claim is made.
Source clock truth does not establish per-event semantic or artistic truth.

Run after publishing code and SHA-bound server synchronization:

```
PYTHONPATH=src python scripts/timing_validity_r66.py run
PYTHONPATH=src python scripts/timing_validity_r66.py verify --out outputs/timing_validity_r66_replay_01
```

Use separate tmux processes and preserve exit-status receipts. Replay rechecks
raw hashes, re-extracts audio clocks and full-record motion peaks, then compares
all scientific rows/statistics/clocks exactly. The three short listening
examples are chosen by identity before scores; they contain actual paired
source motion and shifted audio, not newly generated choreography. Playback
does not count as independent quality evaluation.
