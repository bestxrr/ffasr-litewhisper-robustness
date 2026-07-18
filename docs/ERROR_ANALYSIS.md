# Error Analysis

No per-utterance predictions have been generated yet.

Required future artifacts:

- `errors.csv` or parquet.
- `condition_summary.csv`.
- `confusion_pairs.csv`.
- `worst_100.md`.
- `slice_metrics.json`.
- `report.html`.

## 2026-07-17 04:55:25 UTC - Baseline full-proxy failure slices

Baseline full proxy average WER is 18.02. The hardest slices are low SNR `-3..0 dB` at 54.73 WER, RT60 `0.8..1.1 s` at 35.37 WER, soft clipping at 50.72 WER, and lowpass/bandpass filtering at 32.34/34.82 WER. Low-condition baseline errors are substitution-heavy with substantial insertions: S/D/I 2936/647/784.

Pilot A and B increased substitutions on dry/high/mid instead of reducing degraded errors, so the immediate issue is optimization/regularization rather than a demonstrated sequence-level MWER problem.


## 2026-07-17 05:37:30 UTC - Audits and clean-only sanity after label fix

Training manifest audit:
- Existing `librispeech_train_clean_l0`: 1,200 utterances, 4.24 h, 11 speakers; no corruption or train/proxy leakage, but smoke-sized.
- New `librispeech_train_clean_diag`: 2,400 utterances, 8.61 h, 22 speakers; 16 kHz, no corrupt files, no train/proxy audio/text/speaker overlap. Existing manifests were not overwritten.

Teacher-forcing audit:
- Diagnosed label-prefix inconsistency: tokenizer labels could omit `<|en|><|transcribe|>`.
- Fixed `make_labels` to set English/transcribe/no-timestamps prefix explicitly and mask padding to `-100`.
- Tests now cover label prefix and edit alignment edge cases.

Clean-only sanity:
- Command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/clean_only_attn_lr1e5_l0.yaml`
- Train updates: 50, wall 69.6s, peak VRAM 1.31 GB allocated.
- Fixed 64-utterance clean CE: base 0.8900, u5 0.8844, u10 0.8775, u25 0.8575, u50 0.8201.
- Dry mini WER: u5 1.82, u10 1.82, u25 1.76, u50 1.76; baseline dry is 1.82.
- Decision: clean-only sanity passes. The training path can reduce teacher-forced clean CE without dry WER regression.

Proxy calibration:
- Proxy v1 full average WER 18.02 is easier than official public baseline 26.04.
- Metadata: balanced 500 per condition; 1,500 degraded samples all use procedural exponential RIR; SNR low median 1.12 dB, mid 10.13 dB, high 19.49 dB; effect counts include additive noise/RIR for all degraded plus lowpass 182, bandpass 181, highpass 188, soft_clip 140.
- Decision: preserve proxy_v1 for current controlled ablations; do not create proxy_v2 mid-comparison until calibration is intentionally versioned.

GPU blocker:
- `nvidia-smi` shows an unrelated `VLLM::EngineCore` process using about 2.9 GB on GPU 0. Pilot D/E configs are prepared but not launched while this competing process is active.

## 2026-07-17 07:56:37 UTC - Pilot D paired error analysis

Pilot D (`lambda_clean=1.0`) vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 35 / 31 / 334.
- Mean paired WER delta: -0.42 points.
- Dry mean delta: +0.17; S/D/I delta 0/0/+2.
- High mean delta: +0.35; S/D/I delta +6/-5/+1.
- Mid mean delta: -0.17; S/D/I delta +11/-9/0.
- Low mean delta: -2.03; S/D/I delta -2/-15/-8.

Interpretation: the corrected-label clean-anchor path is no longer catastrophically damaging, and the low-condition gains are deletion/insertion reductions. The remaining blocker is insufficient mid/low movement at this short pilot scale; lambda=1.0 may over-anchor clean speech and under-adapt mid/high acoustic mismatch.

## 2026-07-17 08:12:00 UTC - Pilot E paired error analysis

Pilot E (`lambda_clean=0.25`) vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 36 / 37 / 327.
- Mean paired WER delta: -0.03 points.
- Dry mean delta: +0.17; S/D/I delta 0/0/+2.
- High mean delta: +0.35; S/D/I delta +6/-5/+1.
- Mid mean delta: +1.25; S/D/I delta +5/-10/+17.
- Low mean delta: -1.90; S/D/I delta -2/-14/-7.

Interpretation: reducing the clean-anchor weight did not unlock mid/low gains. It preserved the low-condition deletion/insertion reduction pattern but introduced a mid-condition insertion regression and higher catastrophic rate. Current evidence points away from clean-anchor weighting as the main blocker.

## 2026-07-17 08:20:00 UTC - Pilot F paired error analysis

Pilot F (degraded CE only) vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 33 / 36 / 331.
- Mean paired WER delta: +0.10 points.
- Dry mean delta: +0.10; S/D/I delta +1/0/+1.
- High mean delta: +0.43; S/D/I delta +10/-5/+1.
- Mid mean delta: +1.03; S/D/I delta +5/-10/+15.
- Low mean delta: -1.16; S/D/I delta +3/-14/-8.

Interpretation: deletion reductions on low/mid appear reproducible, but without clean anchoring the model picks up more substitutions and insertions. The useful adaptation signal is present but not strong enough with encoder-attention LoRA alone.

## 2026-07-17 08:34:00 UTC - Pilot G paired error analysis

Pilot G (encoder attention + `fc2` LoRA) vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 45 / 165 / 190.
- Mean paired WER delta: +6.21 points.
- Dry mean delta: +2.07; S/D/I delta +40/+13/+2.
- High mean delta: +4.17; S/D/I delta +79/+8/+5.
- Mid mean delta: +3.85; S/D/I delta +67/+33/-3.
- Low mean delta: +14.74; S/D/I delta +49/+21/+105.

Interpretation: increasing encoder adapter capacity with `fc2` causes over-adaptation rather than robust acoustic improvement. The model learns the training objective aggressively but damages decoding behavior, so further work should avoid broad FFN output LoRA unless paired with much stronger regularization and evidence from a small pilot.

## 2026-07-17 08:50:00 UTC - Pilot H paired error analysis

Pilot H u150 vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 44 / 78 / 278.
- Mean paired WER delta: +0.73 points.
- Dry mean delta: +1.49; S/D/I delta +20/+9/+3.
- High mean delta: +2.11; S/D/I delta +30/+8/+4.
- Mid mean delta: +0.33; S/D/I delta +2/-7/+11.
- Low mean delta: -1.01; S/D/I delta +20/-25/-15.

Pilot H u300 vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 56 / 182 / 162.
- Mean paired WER delta: +4.61 points.
- Dry mean delta: +3.27; S/D/I delta +50/+10/+5.
- High mean delta: +4.61; S/D/I delta +83/+4/+5.
- Mid mean delta: +6.89; S/D/I delta +89/+6/+28.
- Low mean delta: +3.49; S/D/I delta +74/+5/-23.

Interpretation: continuing the best attention-only recipe shifts low-condition errors away from deletions and insertions, but substitutions rise substantially and dry/high performance collapses. The training objective continues to optimize but becomes less aligned with proxy WER. The current bottleneck is not update count; it is the adaptation signal and/or simulator targeting.

## 2026-07-17 09:10:00 UTC - Pilot I paired error analysis

Pilot I u75 vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 18 / 8 / 374.
- Mean paired WER delta: -0.27 points.
- Dry mean delta: -0.07; S/D/I delta -2/0/+1.
- High mean delta: -0.34; S/D/I delta -3/-1/-2.
- Mid mean delta: -0.25; S/D/I delta +5/-5/-4.
- Low mean delta: -0.42; S/D/I delta +9/-4/-10.

Pilot I u100 vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 20 / 16 / 364.
- Mean paired WER delta: -0.06 points.
- Dry mean delta: -0.07; S/D/I delta -2/0/+1.
- High mean delta: +0.06; S/D/I delta +1/-1/0.
- Mid mean delta: -0.24; S/D/I delta +5/-5/-4.
- Low mean delta: +0.02; S/D/I delta +8/-5/-6.

Pilot I u150 vs baseline mini-proxy:

- Paired samples: 400.
- Improved/regressed/unchanged: 28 / 23 / 349.
- Mean paired WER delta: +0.07 points.
- Dry mean delta: -0.05; S/D/I delta -2/0/+1.
- High mean delta: +0.05; S/D/I delta +5/-5/0.
- Mid mean delta: -0.29; S/D/I delta +3/-6/-2.
- Low mean delta: +0.57; S/D/I delta +15/-18/+25.

Interpretation: lower LR produces small, sparse paired improvements and avoids the dry collapse, but it does not move enough utterances. The consistent pattern is deletion reduction offset by substitution increases and, at longer training, low-condition insertion growth. This supports changing simulator calibration or target selection rather than promoting the current SFT recipe.

## 2026-07-17 09:35:00 UTC - Tuning-dev baseline and simulator mismatch

Tuning-dev baseline WER: avg 14.87, dry 1.97, high 6.82, mid 14.06, low 36.64. The proxy is easier than official and is diagnostic only.

Simulator audit found that prior Pilot D-style training omitted highpass and bandpass effects that are present in proxy metadata. Tuning-dev baseline effect slices:

- soft_clip: 45.97 WER, S/D/I 279/39/64 over 47 samples.
- bandpass: 22.56 WER, S/D/I 255/43/32 over 58 samples.
- highpass: 21.33 WER, S/D/I 139/24/30 over 50 samples.
- lowpass: 19.08 WER.

Interpretation: hard spectral/clipping slices are substitution-heavy, so adding calibrated spectral and clipping effects to training was justified before changing architecture.

## 2026-07-17 10:06:00 UTC - Pilot J/K/L paired error pattern

Pilot J u100 vs tuning-dev baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 59 / 49 / 492.
- Mean paired WER delta: +0.51 points despite aggregate avg improving slightly.
- Dry S/D/I delta +4/+2/-1; high -1/+1/+2; mid -21/0/+1; low -50/+47/-4.

Interpretation: simulator-calibrated training reduces substitutions in mid/low but increases low-condition deletions enough to miss the gate. Adapter norm analysis shows updates concentrated in layers 30-31 under Pilot J.

Pilot K and Pilot L tuning-dev summaries:

- Pilot K layers 14-20: avg 15.60, low 39.38, catastrophic 1.00. Lower available v_proj-only adaptation is too weak.
- Pilot L layers 21-26: avg 15.97, low 40.94, catastrophic 1.33. Middle-only adaptation worsens low and catastrophic behavior.

Interpretation: useful adaptation is not coming from lower/middle locality under this objective. Complete late-only Pilot M before deciding whether to combine late targets differently or change the loss/simulator again.

## 2026-07-17 10:24:00 UTC - Pilot M paired error analysis

Pilot M late-only layers 27-31 vs tuning-dev baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 31 / 32 / 537.
- Mean paired WER delta: +0.71 points.
- Dry mean delta: +0.01; S/D/I delta +1/+1/-1.
- High mean delta: +0.08; S/D/I delta +5/0/+2.
- Mid mean delta: -0.00; S/D/I delta -6/+2/-3.
- Low mean delta: +2.77; S/D/I delta -8/+2/+77.

Interpretation: late-only LoRA can slightly reduce low substitutions, but it increases low insertions substantially. Combined with Pilot J's low deletion increase, the current simulator-calibrated SFT objective is moving errors between S/D/I buckets instead of reducing total WER. The next acoustic pilot should be deletion/insertion-safe: keep spectral/clipping coverage, remove frame dropout, soften the low-condition distribution, and keep the same full attention target family so the changed variable is simulator severity rather than target locality.

## 2026-07-17 10:43:00 UTC - Pilot N paired error analysis

Pilot N u50 vs tuning-dev baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 37 / 31 / 532.
- Mean paired WER delta: +0.62 points.
- Dry mean delta: -0.02; S/D/I delta 0/+1/-1.
- High mean delta: +0.19; S/D/I delta +6/-1/0.
- Mid mean delta: -0.32; S/D/I delta -9/0/-5.
- Low mean delta: +2.63; S/D/I delta +4/-11/+64.

Pilot N u100 vs tuning-dev baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 59 / 46 / 495.
- Mean paired WER delta: +0.60 points.
- Dry mean delta: +0.11; S/D/I delta +4/+2/-1.
- High mean delta: +0.50; S/D/I delta 0/+1/+2.
- Mid mean delta: -0.49; S/D/I delta -17/+2/-3.
- Low mean delta: +2.26; S/D/I delta -28/+8/+56.

Interpretation: removing frame dropout and softening low severity did not remove the low insertion problem. The useful part of adaptation is acoustic-substitution reduction in mid/low, but this is offset by low insertion/deletion drift. Clipping remains a plausible culprit because Pilot N still used it, so the next diagnostic should be spectral-only.

## 2026-07-17 11:02:00 UTC - Pilot O paired error analysis

Pilot O u50 vs tuning-dev baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 32 / 28 / 540.
- Mean paired WER delta: +1.10 points.
- Dry mean delta: -0.02; S/D/I delta 0/+1/-1.
- High mean delta: +0.19; S/D/I delta +6/-1/0.
- Mid mean delta: -0.31; S/D/I delta -8/-1/-5.
- Low mean delta: +4.53; S/D/I delta -1/-13/+157.

Pilot O u100 vs tuning-dev baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 56 / 47 / 497.
- Mean paired WER delta: +0.63 points.
- Dry mean delta: +0.11; S/D/I delta +4/+2/-1.
- High mean delta: +0.47; S/D/I delta 0/+1/+2.
- Mid mean delta: -0.60; S/D/I delta -19/+1/-4.
- Low mean delta: +2.52; S/D/I delta -21/+6/+61.

Interpretation: spectral-only training increases low insertions even more at u50 and remains worse at u100. The stable finding across J/N/O is that CE SFT can reduce mid substitutions and sometimes low substitutions, but the low-condition hypothesis length distribution shifts unfavorably. Continue with objective/decoding diagnostics before any longer run.

## 2026-07-17 11:06:00 UTC - Low-condition length diagnostic

Low-condition insertion regressions are concentrated in a few runaway/repetition samples, not uniformly spread across all low examples.

| run | low hyp/ref mean | low hyp/ref p90 | low ratio >1.5 | low S/D/I | insertion samples >=5 |
| --- | ---: | ---: | ---: | --- | ---: |
| baseline | 1.020 | 1.160 | 3 | 820/111/219 | 7 |
| Pilot J u100 | 1.030 | 1.160 | 2 | 770/158/215 | 5 |
| Pilot N u100 | 1.043 | 1.125 | 3 | 792/119/275 | 5 |
| Pilot O u100 | 1.045 | 1.158 | 3 | 799/117/280 | 5 |

Worst repeated-output samples:

- `low_0013`: baseline already repeats "to be a servant"; J/N/O extend it to 106 hypothesis words and 91 insertions.
- `low_0007`: baseline has a "state of the state" runaway; Pilot J fixes it to 53 hypothesis words, while N/O revert to 121 hypothesis words.
- `low_0070`: all runs remain substitution-heavy with moderate insertions.

Interpretation: robust CE updates can alter runaway behavior on individual low-condition utterances, sometimes positively and sometimes negatively. A decoding-control diagnostic is justified, but it must be paired against the baseline with identical decoder settings before any claim. Do not use no-repeat or repetition penalties as a final result without evaluating the baseline under the same setting.

## 2026-07-17 11:42:00 UTC - No-repeat and consistency paired analysis

Under the no-repeat-3 decoder, the baseline low WER improves substantially compared with default decoding, so all candidate claims use the no-repeat baseline as control.

Pilot F u100 no-repeat-3 vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 62 / 48 / 490.
- Mean paired WER delta: -0.12 points.
- Dry mean delta: +0.01; S/D/I delta +1/+1/-1.
- High mean delta: +0.54; S/D/I delta +3/+2/+2.
- Mid mean delta: -0.56; S/D/I delta -17/-8/-4.
- Low mean delta: -0.47; S/D/I delta -26/+43/-24.

Pilot Q u100 consistency vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 60 / 48 / 492.
- Mean paired WER delta: -0.03 points.
- Dry mean delta: +0.10; S/D/I delta +3/+2/-1.
- High mean delta: +0.56; S/D/I delta +2/+3/+2.
- Mid mean delta: -0.32; S/D/I delta -15/-1/-3.
- Low mean delta: -0.45; S/D/I delta -13/+6/-14.

Interpretation: consistency largely prevents the low deletion increase seen in E/F, but it also reduces the low substitution gain. The active blocker is now a weak acoustic improvement signal rather than catastrophic repetition alone. The next candidate should increase useful mid/low acoustic adaptation without broadening to unstable FFN LoRA or longer training.

## 2026-07-17 12:05:00 UTC - Pilot R paired error analysis

Pilot R u100 LayerNorm adaptation vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 64 / 52 / 484.
- Mean paired WER delta: -0.11 points.
- Dry mean delta: +0.30; S/D/I delta +7/+16/-1.
- High mean delta: +0.53; S/D/I delta 0/+4/+2.
- Mid mean delta: -0.32; S/D/I delta -14/-5/0.
- Low mean delta: -0.94; S/D/I delta -41/+45/-14.

Interpretation: LayerNorm adaptation strengthens the acoustic substitution reduction on low speech, but it pays for it with dry and low deletion increases. This reinforces the current blocker: the adaptation can make acoustically plausible corrections, but the token-level CE objective is not preserving deletion behavior.

## 2026-07-17 12:28:00 UTC - Pilot S rank-32 paired analysis

Pilot S u100 rank-32 attention LoRA vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 76 / 117 / 407.
- Mean paired WER delta: +0.41 points.
- Dry mean delta: +1.16; S/D/I delta +33/+24/0.
- High mean delta: +1.62; S/D/I delta +36/+29/+5.
- Mid mean delta: +0.10; S/D/I delta +8/-7/-9.
- Low mean delta: -1.22; S/D/I delta -50/+48/-15.

Interpretation: more attention capacity amplifies the existing tradeoff: low substitutions fall, but low deletions rise and dry/high regress substantially. This is not a simple capacity shortage; the SFT objective is not preserving recognition behavior while adapting to degraded acoustics.

## 2026-07-17 12:43:00 UTC - Interpolation paired analysis

Pilot T D/P interpolation vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 61 / 48 / 491.
- Mean paired WER delta: -0.09 points.
- Dry mean delta: +0.10; S/D/I delta +3/+2/-1.
- High mean delta: +0.56; S/D/I delta +2/+3/+2.
- Mid mean delta: -0.37; S/D/I delta -16/-3/-2.
- Low mean delta: -0.64; S/D/I delta -15/+2/-12.

Interpretation: D/P interpolation reduces the deletion drift compared with some stronger-adaptation runs, but it also gives up too much substitution reduction. This confirms Pilot D is near the best point found so far on the current attention-LoRA CE trajectory.

## 2026-07-17 13:01:00 UTC - Pilot V paired error analysis

Pilot V u100 condition-weighted CE vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 61 / 46 / 493.
- Mean paired WER delta: -0.07 points.
- Dry mean delta: +0.11; S/D/I delta +4/+2/-1.
- High mean delta: +0.58; S/D/I delta +3/+3/+2.
- Mid mean delta: -0.59; S/D/I delta -15/-5/-3.
- Low mean delta: -0.39; S/D/I delta -33/+48/-24.

Interpretation: increasing mid/low loss weight strengthens some mid movement but repeats the low deletion failure. The training objective is still turning low-condition substitution reductions into deletion-heavy hypotheses.

## 2026-07-17 13:13:00 UTC - Beam decoder paired analysis

Pilot D beam3/lp1.1/no-repeat vs beam baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 64 / 47 / 489.
- Mean paired WER delta: -0.42 points.
- Dry mean delta: -0.02; S/D/I delta +3/0/-1.
- High mean delta: -0.97; S/D/I delta -8/-5/-5.
- Mid mean delta: -0.07; S/D/I delta -5/-7/+3.
- Low mean delta: -0.60; S/D/I delta -8/-16/+8.

Interpretation: beam search changes the error mix and reduces deletions in paired counts, but catastrophic outputs increase and the aggregate mid/low gain remains weak. This supports trying decoder-conditioning adaptation before MWER, not more decoder-only knobs.

## 2026-07-17 13:42:00 UTC - Pilot W paired error analysis

Pilot W u100 decoder cross-attention LoRA vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 61 / 54 / 485.
- Mean paired WER delta: -0.12 points.
- Dry mean delta: +0.51; S/D/I delta +9/+15/-1.
- High mean delta: +0.83; S/D/I delta +13/+22/+4.
- Mid mean delta: -0.57; S/D/I delta -16/-2/-6.
- Low mean delta: -1.24; S/D/I delta -19/+5/-28.

Interpretation: decoder cross-attention adaptation is the first branch to materially reduce low insertions while improving low WER beyond Pilot D, but it overfits dry/high. The likely next useful test is the same target set at lower LR or earlier checkpoint selection.

## 2026-07-17 14:18:00 UTC - Pilots X/Y/Z paired error analysis

Pilot X u150 lower-LR decoder cross-attention LoRA vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 50 / 38 / 512.
- Mean paired WER delta: -0.19 points.
- Dry mean delta: +0.07; S/D/I delta +1/+2/-1.
- High mean delta: +0.65; S/D/I delta +6/-1/+3.
- Mid mean delta: -0.64; S/D/I delta -18/-3/-5.
- Low mean delta: -0.85; S/D/I delta -17/+7/-24.

Pilot Y u150 condition-weighted lower-LR decoder cross-attention LoRA vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 49 / 42 / 509.
- Mean paired WER delta: -0.09 points.
- Dry mean delta: +0.00; S/D/I delta +1/+1/-1.
- High mean delta: +0.36; S/D/I delta +6/-1/0.
- Mid mean delta: -0.28; S/D/I delta -16/-3/0.
- Low mean delta: -0.43; S/D/I delta +1/-4/-10.

Pilot Z u175 extended lower-LR decoder cross-attention LoRA vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 60 / 49 / 491.
- Mean paired WER delta: -0.15 points.
- Dry mean delta: +0.39; S/D/I delta +6/+15/-1.
- High mean delta: +0.65; S/D/I delta +7/-1/+3.
- Mid mean delta: -0.44; S/D/I delta -16/-5/-3.
- Low mean delta: -1.19; S/D/I delta -12/0/-16.

Interpretation: Pilot X has the best balance found so far: it reduces mid/low substitutions and low insertions with minimal dry damage, but the mid/low gain is still below the 5% promotion gate. Y confirms that simple mid/low CE weighting weakens the decoder-cross benefit. Z confirms a saturation/overfit trajectory: additional updates improve some low-condition terms but increase dry deletions and high errors, with catastrophic output rate rising to 1.00%.

## 2026-07-17 15:05:00 UTC - Pilot AC paired error analysis

Pilot AC u150 dropout-regularized decoder-cross LoRA vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 51 / 37 / 512.
- Mean paired WER delta: -0.23 points.
- Dry mean delta: +0.07; S/D/I delta +1/+2/-1.
- High mean delta: +0.36; S/D/I delta +6/-1/0.
- Mid mean delta: -0.76; S/D/I delta -19/-4/-7.
- Low mean delta: -0.59; S/D/I delta -8/+6/-18.

Interpretation: dropout regularization improves catastrophic rate and strengthens mid-condition paired gains, but it weakens low substitution reduction relative to Pilot X and still increases low deletions. The branch confirms that the remaining blocker is not just output instability; the model needs more low-condition acoustic gain without increasing deletion errors.

## 2026-07-18 02:30:00 UTC - Pilot AD (min_new_tokens=5) paired error analysis

Pilot AD min_new_tokens=5 (decode-only, Pilot X u150 adapter) vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 51 / 38 / 511.
- Mean paired WER delta: -0.28 points.
- Dry delta_s/d/i: +1/+2/-1 (mean delta +0.07).
- High delta_s/d/i: +6/-1/+3 (mean delta +0.65).
- Mid delta_s/d/i: -18/-3/-5 (mean delta -0.64).
- Low delta_s/d/i: -17/+6/-24 (mean delta -1.18).

Interpretation: the min_new_tokens floor is numerically identical to Pilot X's own paired deltas for substitutions/insertions and only trims low deletions by 1 unit relative to Pilot X (119->118 raw count). The +6 low deletion delta versus baseline is unchanged, confirming this is not a premature-EOS-fixable-by-floor problem — the model is genuinely failing to acoustically recover certain words under degradation, not just stopping early. This rules out decode-time fixes and points to a training-side length/coverage signal for Pilot AE.

## 2026-07-18 03:20:00 UTC - Pilot AE (u150, eos_suppress_weight=25.0) paired error analysis

Pilot AE u150 vs no-repeat baseline:

- Paired samples: 600.
- Improved/regressed/unchanged: 52 / 45 / 503.
- Mean paired WER delta: -0.19 points (weaker than Pilot AC's -0.23 and Pilot X's own gain).
- Dry delta_s/d/i: +1/+1/-1 (mean delta +0.01).
- High delta_s/d/i: +6/+1/0 (mean delta +0.37).
- Mid delta_s/d/i: -14/-9/0 (mean delta -0.58).
- Low delta_s/d/i: **+15/-18/-3** (mean delta -0.57).

Interpretation: this is the first pilot in the branch where the low-condition delta_d flips negative (-18, i.e. deletions actually fall versus the no-repeat baseline, unlike every prior pilot which showed deletions rising as substitutions/insertions fell). This confirms the EOS-suppression hypothesis was correctly targeted at the deletion mechanism. But the trade is now inverted: recovered positions are filled with the wrong word (delta_s +15) rather than the right one, so net low WER improvement is smaller than Pilot X's. The finding narrows the remaining problem further: the blocker is not just "when to stop" but "what acoustic evidence to trust under degradation" — fixing the stopping bias alone surfaces a separate substitution weakness. A follow-up should use a much smaller eos_suppress_weight (current 25.0 likely over-corrects) or gate the term to low/mid conditions only so it does not compete with the acoustic CE signal on dry/high samples.

## 2026-07-18 09:30:00 UTC - Pilot AJ/AK/AJK acoustic-bottleneck error analysis

Diagnosis that motivated the pilot (from slicing Pilot X u150 errors.csv):

- Low-condition WER by SNR: 48.2 (<0dB, n=43), 36.5 (0-3dB, n=62), 24.6 (3-6dB, n=45).
- Low-condition WER by reverb rt60: 28.3 (<0.5, n=59), 36.5 (0.5-0.8, n=48), 47.0 (>0.8, n=43).
- Monotonic in both physical difficulty axes; dry condition near-perfect (2.11). Signature of an acoustic-frontend (encoder) limit, not a decoder/LM limit.

Post-pilot error structure (low condition S/D/I, vs no-repeat baseline 817/112/140):

- Pilot X u150:  800/119/116.
- Pilot AJ u150: 798/118/125 (distillation w=1.0) -- essentially unchanged; -2 subs, -1 del, +9 ins.
- Pilot AK u150 (encoder FFN): error structure collapses (avg WER 17.92, dry 4.71) -- overfit, not a controlled S/D/I shift.

Interpretation: the per-frame clean-target distillation signal ran on every degraded step (mean term 0.22-0.24) yet moved the low-condition S/D/I by single digits -- the current LoRA target set (encoder self-attn + decoder cross-attn) simply cannot push the encoder's per-frame representation toward the clean one by enough to matter, and giving it the capacity to do so (encoder FFN) overfits the 2400-utterance pool instead. The error is correctly localized to the encoder, but it is not fixable by recipe changes on this data: it is an information/data-scale limit. The decisive next lever is more/real degraded training data, not another loss term or target-set edit.
