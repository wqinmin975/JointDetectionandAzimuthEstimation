# Response to Reviewers

**Manuscript:** sensors-4688906 — *Joint Detection and Azimuth Estimation of UAVs Using MEMS Microphone Arrays and Multi-Task Deep Learning: From Simulation to Real-World Recordings*

**Decision:** Major revision. We thank all three reviewers for a careful and constructive assessment. We have revised the manuscript substantially and, importantly, we have re-examined the underlying code and experiments rather than only editing text. This process uncovered several genuine defects that we have corrected; the reported results have been regenerated accordingly and every number in the revised manuscript is produced by the released code. Below we address each point. A summary of the most consequential changes is given first.

---

## Summary of the major changes

1. **We found and fixed a real implementation defect in the "dynamic uncertainty weighting."** In the original code the two log-variance parameters were constructed *after* the optimizer and were therefore never updated; the nominal dynamic scheme was in fact static equal weighting with a global 0.5 scale. We moved the parameters into the optimizer, verified that they now update (final log-variances −2.76 and +0.94), and reran all experiments. The repaired dynamic rule still does **not** improve peak azimuth accuracy over fixed equal weighting (7.50° vs 5.62° at −5 dB). We therefore no longer claim that dynamic weighting is more accurate; its value is stated honestly as removing the manual weight search. The manuscript no longer claims that joint training is the primary source of the accuracy gain.

2. **We added properly trained single-task references.** The previous "detection-only"/"localization-only" rows used an *untrained* output head. We added Solo-Det and Solo-Loc networks trained individually from the same initialization. With these correct references, single-task localization (5.95°) matches or slightly exceeds the joint model (7.50°), and single-task detection matches it exactly (Pd 1.00 / Pfa 0.00). We report this openly.

3. **We corrected the training/evaluation mismatch and the delay model.** The regression loss is now a masked 180°-periodic circular L1 (negative samples no longer receive an artificial 0° target), and the simulated inter-microphone delays are now fractional (frequency-domain phase ramp) instead of rounded to integer samples, consistent with the continuous-delay CRB model.

4. **We added a paired benchmark and an honest SOTA comparison.** GCC-PHAT, SRP-PHAT, MUSIC and the network are now evaluated on *identical* Monte Carlo samples (same SNR, same signal, same noise). Paired t-tests and a non-paired Welch test are reported. MUSIC is included as a diagnostic subspace baseline; it is in fact the most accurate method under the idealized single-tone simulation, and we state this explicitly and discuss the conditions under which the comparison is and is not representative.

5. **We redefined the real-data task as 180° azimuth-axis estimation and disclosed the folded-error limitation.** The reported 18.05° is a 180°-folded error; unfolded to the full 360° circle it is approximately 78–103°, because 68 of 163 recordings have true azimuths in (180°,360°] while predictions lie entirely in [0,180). We now present this openly: the folded numbers characterize only the 180° axis, while full-circle capability is reported separately. In a v3-architecture re-run, training with unfurled 360° labels and a 360° circular loss reduces full-circle MAE from 138.4° to 31.9° (predictions span [84°,267°]), at the cost of a higher folded error (24.4° versus 19.09° in that re-run); this result remains in-sample.

6. **We clarified the 6-channel vs 28-channel architecture.** These are two separately trained networks, not a weight-reused adapter; the description in Fig. 1, the methods and the contribution list now state this precisely.

All line references below are to the revised manuscript.

---

## Reviewer 1

**R1-1. Azimuth-error metric / 180° folding physical consistency.**
We agree the metric must be explicit. For the 6-element ULA at broadside elevation the source direction is parameterized by azimuth over [0°,180°); we now define the reported error as a **180°-folded azimuth-axis error**, min(|Δ| mod 180°, 180° − |Δ| mod 180°), and state its chance level (45° folded, 90° on the full circle) in Sec. "Metrics". We do not claim the ULA distinguishes front from back. The training loss was previously linear L1 in degrees while evaluation used the circular metric — a genuine inconsistency. We replaced training with a masked 180°-periodic circular L1, so training and evaluation now use the same geometry. The 5.28° figure was from the old protocol; the corrected protocol gives 7.50° at −5 dB (Table I). No dataset result relies on equating 0° and 180° without disclosure: the real-data folded/unfolded distinction is now stated explicitly (see R1-2 and our response to Reviewer 2, Major 2).

**R1-2. Front–back ambiguity for the 3-D IDMT and WSU arrays must follow from geometry.**
We agree and thank the reviewer for this important point. We replaced the imprecise "mirror ambiguity about the array plane" wording. More importantly, we audited the actual predictions: of the 163 IDMT segments, 68 have true azimuth in (180°,360°], yet every prediction lies in [0,180). The reported 18.05° is a folded error; the unfolded full-circle error is ≈78–103° depending on the segment-level aggregation convention. We now disclose this in a dedicated "Folded-error disclosure" paragraph, state that the task is **180° azimuth-axis estimation** rather than full-circle DOA, and give a concrete example (the segment group Phantom4-static-flight-02 has a folded MAE of 9.76° but an unfolded error near 171°, i.e., an almost complete front/back reversal). We no longer infer full-circle ability from the folded metric; instead, a companion v3-architecture re-run with unfurled 360° labels reduces full-circle MAE from 138.4° to 31.9° (the 68 back-half segments have 37.3° full-circle error), while folded MAE rises from 19.09° to 24.4°—an honest trade-off that is still an in-sample estimate. We also note the model could in principle output values outside [0,180) — its final layer has no clamp — so this is a consequence of the folded training targets, not a hard output constraint.

**R1-3. DOA regression loss included drone-absent samples with artificial 0° target.**
Confirmed and fixed. Negative samples were assigned label 0° and entered the regression loss. The regression term is now masked to drone-present samples only (masked circular L1), and we verified the mask numerically. The single-task and all multi-task variants were rerun under the corrected loss.

**R1-4. Ablation did not compare against properly trained single-task networks.**
Confirmed and fixed. The old detection-only/localization-only rows used untrained output heads. We added Solo-Det and Solo-Loc trained from the same seed and initialization, and report them as the correct references. The result is stated honestly: with proper single-task training, localization is 5.95° versus 7.50° for the joint model, and detection is identical (Pd 1.00 / Pfa 0.00). We removed the claim that joint training is necessary or the main source of gain; the untrained-head rows are retained only as diagnostic ablations and clearly labelled as such.

**R1-5. Synthetic scenario too simple (single AM tone, Gaussian noise); MEMS non-idealities absent.**
This is a valid limitation. We have (a) strengthened the honesty statement in the Introduction and Conclusion that the main results come from idealized simulation and third-party public recordings, not from measurements with our own MEMS hardware, with all MEMS-specific impairments (self-noise, sensitivity mismatch, channel failure, geometry error) explicitly listed as future work; and (b) retained the real-corpus experiments as the evidence for conditions beyond the ideal simulation. We did not fabricate a broadband/non-stationary simulation study; we now state the scope of the evidence plainly rather than implying broader robustness.

**R1-6. Delay implementation vs CRB consistency; fractional delays needed.**
Confirmed and fixed. Delays were previously rounded to integer samples with circular roll. We implemented linear fractional delays via a zero-padded frequency-domain phase ramp and applied them consistently in all simulated experiments. We also changed the CRB wording: Eq. (3) is now described as a narrowband broadside RMSE-type reference, not a lower bound on angle-averaged MAE — the CRB bounds the variance/RMSE of unbiased estimators at a single angle, and the AM signal is treated as narrowband (sideband power small). We do not claim the estimator "approaches" the bound.

**R1-7. Classical and network baselines must share SNRs, samples and Monte Carlo realizations.**
We added a fully paired benchmark: for every SNR, one set of 600 trials generates a single (azimuth, signal, noise) realization shared by the network, GCC-PHAT, SRP-PHAT and MUSIC. At −5 dB the paired mean differences are 36.90° (network vs GCC, t=−32.8, p≈1.2e−135) and 42.22° (network vs SRP, t=−35.0, p≈7.2e−147), with 95% CIs excluding zero. We also report a non-paired Welch test for the Table I comparison (diff 38.90°, 95% CI [31.91°,45.89°], p≈2e−16). We are explicit about which test is paired and which is not.

**R1-8. IDMT fine-tuning on the same 163 recordings; need grouped holdout.**
The in-sample nature of the 18.05° figure was already disclosed in five places in the original manuscript; we retained and strengthened that disclosure. We verified that the evaluation azimuths come from the onboard GPS/pose CSV accompanying each recording — data that would be available at deployment — and are used only as evaluation ground truth and fine-tuning targets, never as network input, so this is not input leakage. We added a per-flight error decomposition (8 flights; flight prefix and segment counts given in the revised results and in the released CSV). We state plainly that a held-out number requires re-fine-tuning within a flight-grouped split and is left as future work; we do not present 18.05° as a generalization estimate.

**R1-9. Real-world joint detection not demonstrated.**
We agree. The WSU experiment evaluated only classical energy and SRP-power detectors; the learned detection head was never run on real background negatives, and IDMT has no negatives. We have rewritten the relevant claims so that WSU supports only "classical-baseline external validation and classical detection ROC", and we state that real false-alarm performance of the learned detector remains unvalidated. We include Hanley–McNeil confidence intervals for the WSU AUCs (test1 energy 0.669, CI [0.605,0.733]; test1 SRP 0.528, CI [0.451,0.605], not significant; test2/combined ≈0.5) with an explicit caveat that 0.1 s adjacent frames are strongly correlated and the 54 test1 background frames lie within 5.3 s of one recording, so the nominal CIs are optimistic. We do not claim robust real detection performance.

**R1-10. Geometry-aware frontend: how mean-pooled coordinate encoding preserves GCC-channel/pair association.**
We revised the methods description. The 28 GCC-PHAT channels correspond one-to-one to the 8·7/2 = 28 microphone pairs (i<j, lexicographic), and each channel is convolved along the lag axis so a per-pair TDOA peak is learned before any pooling. The mean-pooled coordinate vector is a global *condition* appended to the fused features, not a replacement for the per-pair channels; the pair-to-channel association is therefore carried by the input channels themselves, while the geometry vector provides an array-shape conditioning signal. We clarified this distinction in the text and added preprocessing details (coordinate units, pair ordering, lag sign convention, resampling/anti-aliasing, per-window normalization).

**R1-11. Dynamic weighting similar to fixed; advantage is avoiding tuning.**
We agree and now state this directly. After repairing the implementation, dynamic weighting (7.50°) did not beat fixed λ=1.0 (5.62°); its stated value is that it needs no manual λ search. The contribution list and conclusion were rewritten accordingly.

**R1-12. Use "MEMS microphone array" carefully.**
We agree. The title retains MEMS as the sensing modality of interest, but the Introduction and Conclusion now explicitly state that the reported results are from ideal simulation and third-party public recordings, not from our own MEMS hardware, and that hardware-level validation is future work.

---

## Reviewer 2

### Major comments

**R2-M1. Network architecture vs simulation-to-real pipeline (6 channels vs 28×61 GCC-PHAT).**
The reviewer is correct that the text was unclear. There is no adapter and no reuse of the 6-channel weights: we use **two separate networks**. The simulation/Table I uses JointDetDOANetRaw (Conv1d(6,32) on raw 6×2048 waveforms). The real-data branch uses a separately constructed JointDetDOANet with a 28×61 GCC-PHAT input and Conv1d(28,32), pretrained on corrected simulation and then fine-tuned on IDMT. Their first layers have different shapes and cannot share weights. We rewrote Fig. 1 and its caption, the methods, the contribution list and the conclusion to state this precisely, and removed the "same backbone reused / zero-shot transfer of the same weights" wording. The "zero-shot" experiment means the GCC network trained on corrected simulation was applied to real recordings before fine-tuning — not that the 6-channel network was applied to 28-channel inputs.

**R2-M2. Training loss (linear L1 in degrees, [0°,180°)) vs circular evaluation.**
Confirmed and fixed. Training now uses a masked 180°-periodic circular L1, matching the circular evaluation metric; we also state what happens to out-of-range predictions (the folded metric wraps them; the head itself is unbounded).

**R2-M3. IDMT fine-tuning is not independent validation.**
Addressed in detail under R1-8. We retained the in-sample disclosure, added a per-flight decomposition, stated the grouped-split requirement, and clarify that the azimuth ground truth is deployment-available sensor data and not an input leak.

**R2-M4. Detection branch not tested on real background recordings.**
Addressed under R1-9. We scaled down the claim: real detection of the learned head is not demonstrated; WSU results concern classical detectors only.

**R2-M5. Synthetic scenario too narrow for robustness claims.**
Addressed under R1-5. We narrowed the claims and explicitly list the missing conditions (wind, reverberation, varying rotor speed, multiple UAV models, Doppler, non-stationary noise, MEMS non-idealities) as future work.

**R2-M6. Classical benchmark unbalanced and incomplete.**
We addressed this on three fronts: (i) a fully paired benchmark with identical samples and paired t-tests (R1-7); (ii) a non-paired Welch test on the Table I numbers (diff ≈38.9°, CI [31.9°,45.9°], p≈2e−16) to show the gap is not an artefact of independent draws; (iii) MUSIC as a subspace method not based on PHAT whitening, included as a diagnostic comparison. We also note that in our model GCC/SRP are near chance at essentially all SNRs because the 800 Hz carrier period (20 samples) is much longer than the maximum inter-element delay (~3.7 samples), producing a narrowband TDOA ambiguity — refined PHAT/SRP with sub-sample peak interpolation does not change this, so the failure is a property of the narrowband signal model. We state that this diagnostic limits how far the classical comparison should be generalized. Detection operating points are now discussed separately from localization, and Pfa is reported for both classical and learned detectors.

**R2-M7. CRB comparison mixes incomparable quantities.**
We agree on all three sub-points and revised the text. Eq. (3) is now labelled a narrowband broadside RMSE-type reference at θ=90°, not a bound on angle-averaged MAE. We state that the CRB constrains variance/RMSE of unbiased estimators, that it diverges at endfire so a full angular average is unbounded, and that the single-carrier expression is a narrowband approximation to the AM waveform (sideband power small). We removed "approaching the bound" language.

### Minor comments

- **Lreg masking for noise-only samples:** fixed (masked circular L1).
- **Fractional-delay method:** frequency-domain phase ramp with zero padding; used consistently in all simulated experiments; stated in Methods.
- **Fig. 3 per-model counts and uncertainty:** segment counts are Mavic 32, Phantom4 115, Align 16; these are now stated, with the caution that the 16-alignment result rests on few recordings.
- **WSU AUC confidence intervals and temporal correlation:** Hanley–McNeil CIs added with the frame-correlation caveat (R1-9).
- **Preprocessing details:** added (coordinate units, 28-channel pair ordering, lag sign, 48→16 kHz polyphase anti-alias filtering, per-window normalization).
- **Code for IDMT fine-tuning and WSU processing, splits and seeds:** the released repository contains the real-data evaluation and fine-tuning routines (in `drone_acoustic_joint.py`, `--mode real`) and the WSU classical evaluation script (`wsu_eval.py`). The fixed seeds and the segment-level organization of the IDMT corpus are documented in the README and in the code; the 163 recordings are used as one 30-window-per-segment set, and we state explicitly that a flight-grouped split is required for a generalization estimate.
- **Computational cost for the 28×61 frontend:** the two networks have different costs. The six-channel raw-waveform network has 36,078 parameters and about 11.47 M MAC (22.9 MFLOPs) per forward pass, as reported in the manuscript. The separately trained 28-channel GCC-PHAT network has 24,618 parameters and about 0.314 M MAC (0.629 MFLOPs) for its forward pass, i.e. it is substantially smaller. We stress that this figure covers the learned network only; the GCC-PHAT frontend is a fixed, non-parametric FFT-based preprocessing stage with its own (non-learned) computational cost, which is not included in the 0.629 MFLOPs. We have added this clarification to the manuscript rather than reusing the 6-channel figure for both networks.

### Questions for the authors

Answered respectively in R2-M1 (tensor shapes and first-layer configurations; the simulated training data is raw waveforms, not GCC features), R2-M2 (circular loss adopted), R2-M3/R1-8 (grouped holdout rationale), R2-M4/R1-9 (real negative detection), and R2-M7 (CRB derivation and assumptions).

### Recommendation

We have addressed the input-dimension discrepancy through an explicit two-network description, disclosed the in-sample nature of the real-data result and clarified the detection evidence, and corrected the angular loss, false-alarm statistics and CRB comparison.

---

## Reviewer 3

**R3-1. Add a geometry schematic of the testbed.**
Added a TikZ schematic in the system-model section showing the 6-element ULA (d=0.08 m, element indices, azimuth definition, broadside) and the 8-element cubic array with coordinate axes, with a caption stating it is schematic and that experiments use the measured coordinates.

**R3-2. Conduct a real field trial.**
We thank the reviewer but a new field trial with our own hardware is beyond the scope of this revision. We have instead (a) used two public real-world corpora (Fraunhofer IDMT and WSU), (b) disclosed precisely what they do and do not validate, and (c) stated field deployment with the proposed MEMS hardware explicitly as future work. We have narrowed the claims accordingly rather than implying a field validation we did not perform.

**R3-3. No direct comparison with a modern UAV localization algorithm.**
We added MUSIC as a subspace-based DOA reference evaluated on the same paired samples. We also explain why learned DOA methods (DOA-Net, SS-CNN, transformer estimators) are discussed but not re-evaluated: they require different inputs (phase spectra / spatial pseudo-spectra) and separate training budgets, so a direct comparison would not be like-for-like. MUSIC is presented as a diagnostic comparison and its superior accuracy under the ideal narrowband single-source assumption is reported honestly.

**R3-4. Figure 2 text-to-plot proportions improper.**
We adjusted the subplot width/height in Fig. 2 for balanced proportions; no data or curves were changed.

---

## Supplementary refinement round (point-by-point)

This additional round addresses a detailed 12-section refinement list. We group the response into textual/clarity changes and the new experiments that the list requested, and we state the resulting manuscript location for each.

**A. Abstract.** We now state explicitly that the accuracy gain comes from the learned end-to-end representation, not from multi-task joint training (no measurable gain) or dynamic weighting (whose only benefit is removing manual tuning). The 18.1° figure is qualified in the abstract as a 180°-folded axis error, not a full-circle bearing and not a generalization result, and the Monte-Carlo simulation detail was trimmed to make room for the real-data findings and limitations.

**B. Introduction.** We added the acoustic pain points (wind, traffic, reverberation) that motivate the WSU noise experiment; listed the four MEMS non-idealities (gain mismatch, channel failure, calibration error, element self-noise) and identified them as the main simulation-to-hardware gap; sharpened the research gap (few acoustic multi-task DOA works, even fewer that jointly output presence probability and a continuous bearing, and a general absence of properly trained single-task controls); regrouped the contributions into six categories and separated performance contributions from engineering-tool contributions (dynamic weighting = no manual tuning, NP threshold = a calibration strategy, neither an accuracy gain).

**C. Related work.** Section 2.1 now explains, ahead of the results, why PHAT-based TDOA collapses for the narrowband tone: the 800 Hz carrier period (20 samples) greatly exceeds the maximum inter-element delay (~3.7 samples), producing a narrowband TDOA ambiguity — this is a property of the signal model, not of noise weighting. Section 2.2 notes that learned DOA methods overwhelmingly assume a present source and that UAV acoustic deep learning mostly performs detection only. Section 2.3 compares Kendall uncertainty weighting with GradNorm, explains our choice (closed-form, interpretable effective weight), and notes that our finding — dynamic weighting does not beat fixed equal weighting — is relevant to similar acoustic multi-task studies. A new Section 2.4 surveys UAV acoustic multi-task learning and its general lack of strict single-task ablations.

**D. Signal model and setup.** The CRB caveat (RMSE-type reference, broadside only, not a lower bound on MAE) was moved to immediately after Eq. (3). We state that the single-tone simulation is a controlled simplification that induces a domain shift, anticipate the large zero-shot gap, confirm that train/validation/test are disjoint with SNR randomly sampled in training and evaluated per-SNR at test, and added a footnote that the early TDOA-delay bug was fixed and all results use corrected code.

**E. Network and figures.** Fig. 2 now separates the two networks with a dashed frame and bold labels, enlarges the text, and states boldly in the caption that no weights are shared. We added a physical interpretation of λ_t (effective fraction of the classification gradient), a practical note on gradient clipping when a task loss approaches zero, and the derivation of the 61-lag window (±30 lags around zero; the ±30 window is a conservative frontend choice rather than the physical maximum, which is ~10.5 samples for the 0.225 m cube). Fig. 3(a) now carries a caption noting the vertical axis starts at 0.98 while the actual Pd range is [0,1]; Fig. 5 ROC legend was reflowed and error bars (±1 std from the per-frame CSV) were added to the bar chart; Tables 2–4 footers identify the 180°-folded MAE.

**F. New experiment — geometry-encoding ablation (list item 5.4 / 12.1).** We previously stated this module was not ablated. We now ran a controlled ablation (same seed, data and initialization, n=600 at −5 dB) on the 28-channel GCC network: removing the geometry encoding raises MAE from 2.25°±1.79° to 2.95°±2.30°, a paired difference of +0.698° (95% CI [0.526°,0.870°], paired t=7.96, p<10⁻¹⁴, 5000-label permutation p<2×10⁻⁴). The module therefore contributes an approximately 31% error reduction. The 6-element raw-waveform network has no geometry encoder, so the ablation applies only to the 28-channel network. Reported in Section 4.4.

**G. New experiment — 360° full-circle retraining (list item 6.3.1 / 12.2).** We retrained the real-data head with unfurled 0–360° labels and a 360°-periodic circular loss. On the same 163 segments, full-circle MAE falls from 138.4° (folded-label model) to 31.9° (paired t=−18.05, p≈1.3×10⁻⁴⁰; 95% CI [−118.2°,−94.9°]); predictions now span [84°,267°], and the 68 back-half segments reach 37.3° full-circle error. The honest cost is a higher folded error (19.09°→24.4°), i.e., a folded-vs-full-circle trade-off, and the result remains in-sample. The manuscript now reports folded and full-circle results side by side rather than withdrawing the full-circle claim. Reported in Sections 7 and 9.

**H. New experiment — MEMS impairment simulation (list item 12.3).** Using the cached model with paired replay, Holm-corrected paired tests show significant degradation from single-channel failure (+0.44° to +2.07°, worst at the reference element, 1.65°→3.73°), geometry calibration error (2/5/10 mm → +0.146°/+0.646°/+1.736°), and 10 dB self-noise (+0.191°); 20 dB self-noise (p=0.67) and constant channel gain are not significant. For constant per-channel gain (σ=0.05/0.10/0.20) the three conditions give an identical 1.653470° MAE (Δ≈4×10⁻¹⁰°): GCC-PHAT's phase whitening and normalization make the frontend mathematically insensitive to a frequency-flat, time-invariant channel gain. We state this as a frontend invariance, not as network robustness, and note that frequency-selective/time-varying gain, channel phase mismatch and ADC saturation were not tested and cannot be extrapolated to. Reported in Sections 7 and 9.

**I. New experiment — real background detection (list item 6.3.2 / 12.4).** IDMT contains no negative recordings (verified: 163/163 segments are drone-present), so we constructed a cross-domain test pairing IDMT positives (4890 windows) with WSU no_drone ambient negatives (861 windows). Ranking is near-perfect (AUC 0.998, Hanley–McNeil 95% CI [0.9974,0.9991]) but the default γ=0.5 gives Pfa=0.541; after NP recalibration, Pfa=0.05→Pd=0.9939 and Pfa=0.01→Pd=0.9843 (Wilson CIs reported). We label this strictly cross-domain exploratory — different arrays (8-element cube vs 24-element), 63.5% of positives score exactly 1.0 while WSU negatives have median 0.565, so the separation may partly reflect the domain gap; native IDMT negatives are required before any detection claim. Reported in Section 7.

**J. WSU unfolded comparison and epoch/dynamic-weighting analysis (list items 6.3.2 / 7.3 / 12.5).** Recomputing the WSU classical estimates on the full circle shows large folded-to-unfolded gaps (test1 SRP 37.46°→105.33°, GCC 44.51°→88.98°); we note these are post-hoc unfoldings of folded-label classical estimators and are not directly comparable to end-to-end 360° retraining. We corrected the epoch-sensitivity statement: the previous "5.62°→5.54° for equal weighting" mixed λ=1/20-epoch with λ=2/30-epoch. Using one consistent V4 paired protocol, dynamic improves 6.80°→4.20° and fixed λ=1 improves 4.97°→4.24°; we state that this paired protocol is not directly comparable in absolute value to the V3 numbers of Table I (only the trend is compared). The mechanism is now documented numerically: log-variance drifts from (−0.13,+0.10) at epoch 1 to (−3.79,+1.27) at epoch 30, the classification loss collapses to ≈10⁻⁵ while regression stays ≈5–6, and the effective regression weight falls to ≈0.28 — showing that the dynamic scheme mainly applies a time-varying gradient scale and is sensitive to the training budget. Reported in Section 6.1.

**K. Conclusion, limitations and future work.** The conclusion now restates that the gain comes from end-to-end representation rather than joint training or dynamic weighting, lists six limitations (ideal array, simplified source, IDMT negatives absent, in-sample fine-tuning, front-back ambiguity and its folded/full-circle trade-off, no measurements from our own MEMS hardware), and organizes future work as a list: multiple sources, elevation estimation, domain adaptation, MEMS non-ideality modelling, embedded deployment, and large-scale real datasets with background.

---

## English language and figures

All three reviewers flagged figures and tables. We redrew the architecture and geometry figures, adjusted Fig. 2, added the paired-benchmark and per-flight tables, and carefully checked all captions and numerical consistency. The manuscript was proofread for English; remaining language polish will be handled during production.

---

## Closing statement

Every numerical claim in the revised manuscript is generated by the released code and can be reproduced with a single command (`--mode simv3` for the simulation, plus the paired-benchmark and real-data scripts). We have chosen to report several unfavourable findings explicitly — the dynamic-weighting implementation defect, the lack of a measurable joint-training gain, the near-random classical baselines under the narrowband model, MUSIC outperforming the learned estimator, the folded real-data error, and the trade-off between folded and full-circle accuracy after 360° retraining — in preference to presenting a more favourable but less accurate account. We believe this makes the contribution clearer: the durable elements are the geometry-aware frontend (supported by a controlled ablation) with real-array adaptation, the paired and statistically tested evaluation, the NP threshold with explicit cross-domain recalibration caveat, and the honest characterization of what the learned model does and does not gain.
