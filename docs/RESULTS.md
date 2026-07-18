# Results

No training results yet.

Current official public baseline:

- Avg WER: 26.04
- Dry: 4.33
- High: 13.33
- Mid: 29.48
- Low: 57.03

Current live targets:

- Top-7: Avg WER <= 17.34.
- Top-5: Avg WER <= 16.35.

Phase-0 trainability:

- Pass, with export smoke at
  `artifacts/runs/phase0_lite_whisper_trainability/merged_export`.
- Peak VRAM: 2.21 GB.
- Current project disk usage after model cache and export: 3.64 GB.

Proxy baseline, mini split:

- Run: `artifacts/reports/baseline_mini_proxy/summary.json`
- Samples: 400, 100 per condition.
- Avg WER: 20.45.
- Dry/high/mid/low WER: 1.82 / 7.76 / 22.02 / 50.19.
- S/D/I by condition:
  - dry: 25 / 7 / 2
  - high: 128 / 17 / 10
  - mid: 307 / 43 / 43
  - low: 546 / 99 / 159
- Catastrophic output rate: 3.5%.
- Promotion score: 22.20.
- Wall time: 71.65 s.
- Peak allocated/reserved VRAM: 1.17 / 1.27 GB.

Proxy baseline, full split:

- Run: `artifacts/reports/baseline_full_proxy/summary.json`
- Samples: 2,000, 500 per condition.
- Avg WER: 18.02.
- Dry/high/mid/low WER: 1.67 / 7.29 / 17.80 / 45.31.
- S/D/I by condition:
  - dry: 116 / 29 / 11
  - high: 572 / 70 / 80
  - mid: 1275 / 205 / 203
  - low: 2936 / 647 / 784
- Catastrophic output rate: 1.65%.
- Promotion score: 18.84.
- Wall time: 327.82 s.
- Peak allocated/reserved VRAM: 1.17 / 1.27 GB.

## 2026-07-17 04:49:53 UTC - R2 Pilot A Level-0 mini-proxy

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_a_l0.yaml`
- Eval command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_a_l0_mini.yaml`
- Training: 50 updates, wall 78.7 s, peak VRAM 1.33 GB allocated.
- Mini-proxy WER: avg 23.19, dry 4.17, high 12.17, mid 26.11, low 50.31.
- Baseline mini WER: avg 20.45, dry 1.82, high 7.76, mid 22.02, low 50.19.
- Promotion score: 28.65 vs baseline 22.20.
- Decision: reject Pilot A for promotion unless Pilot B is worse; it causes large dry/high/mid regressions and no useful low improvement.

## 2026-07-17 04:55:25 UTC - Stage 1 LoRA target pilot comparison

| run | avg | dry | high | mid | low | promotion score | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 22.20 | 3.50 |
| Pilot A attn | 23.19 | 4.17 | 12.17 | 26.11 | 50.31 | 28.65 | 3.50 |
| Pilot B attn+fc2 | 23.94 | 4.28 | 13.22 | 26.89 | 51.37 | 29.36 | 3.00 |

Decision: reject both Stage-1 target pilots for promotion. Pilot A is less bad than Pilot B, but neither shows mid/low improvement or acceptable dry regression. The next controlled action is an optimization pilot using Pilot A's narrower target set with a lower learning rate before trying clean-anchor, capacity, or MWER work.

## 2026-07-17 04:59:11 UTC - Optimization pilot: Pilot A at LR 1e-5

| run | avg | dry | high | mid | low | promotion score | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 22.20 | 3.50 |
| Pilot A LR 5e-5 | 23.19 | 4.17 | 12.17 | 26.11 | 50.31 | 28.65 | 3.50 |
| Pilot A LR 1e-5 | 20.74 | 1.82 | 7.81 | 22.13 | 51.19 | 22.74 | 4.00 |

Decision: LR 1e-5 prevents the large dry/high regression but does not beat baseline or improve mid/low. Do not promote to a Level-2 run. Next useful work is to add clean-anchor/paired training support and/or recalibrate simulator severity; MWER and capacity experiments remain unjustified.

## 2026-07-17 05:04:41 UTC - Clean-anchor Pilot C result

| run | avg | dry | high | mid | low | promotion score | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 22.20 | 3.50 |
| Pilot A LR 5e-5 | 23.19 | 4.17 | 12.17 | 26.11 | 50.31 | 28.65 | 3.50 |
| Pilot A LR 1e-5 | 20.74 | 1.82 | 7.81 | 22.13 | 51.19 | 22.74 | 4.00 |
| Pilot C clean anchor | 23.47 | 4.39 | 12.82 | 26.05 | 50.62 | 29.48 | 3.75 |

Decision: reject Pilot C. Clean CE anchor at LR 5e-5 did not preserve dry performance and did not improve degraded conditions. The best Level-0 adapter by promotion score is currently Pilot A LR 1e-5, but it still fails promotion because it does not beat baseline or improve mid/low. No Level-2/full run should be launched from these results.


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

## 2026-07-17 07:56:37 UTC - Pilot D result

Normalized clean-anchor `lambda_clean=1.0` with corrected labels and mild-to-hard curriculum:

| run | avg | dry | high | mid | low | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 3.50 |
| Pilot D lambda=1.0 | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 3.25 |

Resource use: 100 updates, 277.0 s training wall time, 73.7 s mini evaluation wall time, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 19.76 GB.

Decision: not promotable. Average improved only 1.52% and combined mid/low improved only 2.01%, below the gates of 2% and 5%. Dry regression stayed acceptable at +0.11 absolute.

## 2026-07-17 08:12:00 UTC - Pilot E result

Normalized clean-anchor `lambda_clean=0.25` with corrected labels and the same mild-to-hard curriculum:

| run | avg | dry | high | mid | low | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 3.50 |
| Pilot D lambda=1.0 | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 3.25 |
| Pilot E lambda=0.25 | 20.31 | 1.93 | 7.86 | 22.69 | 48.75 | 4.00 |

Resource use: 100 updates, 368.3 s training wall time, 75.3 s mini evaluation wall time, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 19.76 GB.

Decision: not promotable. Average improved only 0.68%, combined mid/low improved 1.06%, and catastrophic output rate increased to 4.00%.

## 2026-07-17 08:20:00 UTC - Pilot F result

Corrected-label degraded CE only, same mild-to-hard curriculum as Pilot D/E:

| run | avg | dry | high | mid | low | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 3.50 |
| Pilot D lambda=1.0 | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 3.25 |
| Pilot E lambda=0.25 | 20.31 | 1.93 | 7.86 | 22.69 | 48.75 | 4.00 |
| Pilot F degraded CE only | 20.39 | 1.93 | 8.06 | 22.58 | 49.00 | 4.00 |

Resource use: 100 updates, 148.1 s training wall time, 74.3 s mini evaluation wall time, peak train/eval VRAM 1.31/1.18 GB allocated, project disk 19.77 GB.

Decision: not promotable. Removing the clean anchor reduced low-condition benefit and increased catastrophic outputs; Pilot D remains the strongest corrected-label pilot so far.

## 2026-07-17 08:34:00 UTC - Pilot G result

Encoder attention + encoder `fc2` LoRA, corrected labels, normalized clean-anchor `lambda_clean=1.0`:

| run | avg | dry | high | mid | low | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 3.50 |
| Pilot D attention only | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 3.25 |
| Pilot G attention+fc2 | 26.42 | 4.76 | 12.37 | 27.45 | 61.11 | 3.00 |

Resource use: 100 updates, 388.8 s training wall time, 126.8 s mini evaluation wall time, peak train/eval VRAM 1.50/1.21 GB allocated, project disk 19.78 GB.

Decision: not promotable. FFN output adapters caused broad WER regression despite lower training CE, so capacity expansion by `fc2` is rejected at this LR/update budget.

## 2026-07-17 08:50:00 UTC - Pilot H result

Attention-only normalized clean-anchor `lambda_clean=1.0` continued to 300 updates with intermediate adapters:

| run | avg | dry | high | mid | low | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 3.50 |
| Pilot D attention only u100 | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 3.25 |
| Pilot H attention only u150 | 21.17 | 3.53 | 9.86 | 22.35 | 48.94 | 4.25 |
| Pilot H attention only u300 | 25.06 | 5.29 | 12.37 | 28.91 | 53.68 | 3.00 |

Resource use: 300 updates, 945.6 s training wall time, 81.8 s u150 mini evaluation, 94.3 s u300 mini evaluation, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 19.79 GB.

Relative to baseline, Pilot H u150 changes average WER by -3.55%, combined mid/low by +1.26%, and dry by +1.71 absolute. Pilot H u300 changes average WER by -22.58%, combined mid/low by -14.38%, and dry by +3.48 absolute. Neither checkpoint is promotable.

Current best corrected-label robust pilot remains Pilot D u100: average 20.14, dry 1.93, high 7.86, mid 22.13, low 48.63. It is not promotable because it misses the required >=2% average and >=5% mid/low pilot gates.

## 2026-07-17 09:10:00 UTC - Pilot I result

Attention-only normalized clean-anchor `lambda_clean=1.0`, lower LR `5e-6`, evaluated at 75/100/150 updates:

| run | avg | dry | high | mid | low | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 3.50 |
| Pilot D LR 1e-5 u100 | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 3.25 |
| Pilot I LR 5e-6 u75 | 20.22 | 1.76 | 7.46 | 21.79 | 49.88 | 3.75 |
| Pilot I LR 5e-6 u100 | 20.33 | 1.76 | 7.76 | 21.79 | 50.00 | 4.00 |
| Pilot I LR 5e-6 u150 | 20.71 | 1.76 | 7.76 | 21.74 | 51.56 | 4.00 |

Resource use: 150 updates, 421.2 s training wall time, 72.9/73.5/77.8 s mini evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 19.82 GB.

Pilot I u75 is the best lower-LR checkpoint by promotion score, but it is not promotable: average WER improves only 1.1%, combined mid/low improves only about 0.8%, and catastrophic output rate increases to 3.75%. The best robust pilot overall remains Pilot D u100 by promotion score, but it also remains below promotion gates.

## 2026-07-17 10:06:00 UTC - Tuning-dev proxy, Pilot J, and locality pilots K/L

Created frozen `tuning_dev_proxy` from LibriSpeech test-clean speakers disjoint from train, old mini, and old full proxy. Size: 600 samples, 150 per dry/high/mid/low. Baseline tuning-dev WER is easier than official and must not be used for top-7 claims.

| run | avg | dry | high | mid | low | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline tuning-dev | 14.87 | 1.97 | 6.82 | 14.06 | 36.64 | 1.00 |
| Pilot J u50 sim-cal | 15.26 | 1.97 | 6.97 | 13.76 | 38.36 | 1.17 |
| Pilot J u75 sim-cal | 15.99 | 2.01 | 7.03 | 13.67 | 41.26 | 1.00 |
| Pilot J u100 sim-cal | 14.72 | 2.14 | 6.88 | 13.46 | 36.41 | 1.17 |
| Pilot K u100 layers 14-20 | 15.60 | 2.17 | 6.97 | 13.88 | 39.38 | 1.00 |
| Pilot L u100 layers 21-26 | 15.97 | 2.11 | 7.15 | 13.67 | 40.94 | 1.33 |
| Pilot M u100 layers 27-31 | 15.45 | 2.01 | 7.03 | 13.85 | 38.90 | 1.00 |

Pilot J u100 is the best tuning-dev checkpoint so far by average WER, but it is not promotable: avg improves only 1.0%, combined mid/low improves only 1.7%, and catastrophic outputs increase from 1.00% to 1.17%. Pilot K/L locality restrictions are worse and are rejected. Continue to late-only Pilot M to complete the locality comparison before choosing the next branch.

## 2026-07-17 10:24:00 UTC - Pilot M and locality conclusion

Late-only Pilot M completed the controlled locality comparison.

| run | avg rel improvement | mid/low rel improvement | dry regression | gate |
| --- | ---: | ---: | ---: | --- |
| Pilot J u100 full attention | 1.00% | 1.62% | +0.17 | fail |
| Pilot K lower layers 14-20 | -4.89% | -5.05% | +0.20 | fail |
| Pilot L middle layers 21-26 | -7.35% | -7.72% | +0.13 | fail |
| Pilot M late layers 27-31 | -3.86% | -4.05% | +0.03 | fail |

Pilot M resources: 100 updates, 259.4 s training wall time, 115.7 s evaluation wall time, peak train/eval VRAM 1.32/1.18 GB allocated, project disk 20.39 GB.

Decision: target locality is rejected as the next promotion path. Full Pilot J attention remains the least bad tuning-dev SFT run, but no locality variant passes the average, mid/low, or catastrophic-output gates.

## 2026-07-17 10:43:00 UTC - Pilot N deletion-safe simulator result

Pilot N kept Pilot J's full encoder-attention LoRA target set and clean-anchor objective, but removed frame dropout, softened low SNR, and reduced clipping probability/drive.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline tuning-dev | 14.87 | 1.97 | 6.82 | 14.06 | 36.64 | 1.00 | 0.00% | 0.00% |
| Pilot J u100 sim-cal | 14.72 | 2.14 | 6.88 | 13.46 | 36.41 | 1.17 | 1.00% | 1.62% |
| Pilot N u50 deletion-safe | 15.26 | 1.97 | 6.97 | 13.64 | 38.45 | 1.33 | -2.60% | -2.76% |
| Pilot N u100 deletion-safe | 15.09 | 2.14 | 6.91 | 13.52 | 37.78 | 1.00 | -1.46% | -1.20% |

Resource use: 100 updates, 289.0 s training wall time, 140.7/164.0 s u50/u100 evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.41 GB.

Decision: reject Pilot N. Removing frame dropout and softening the low condition reduces some mid/low substitutions, but low insertions remain elevated and total low WER worsens. No full-proxy or promoted run is justified.

## 2026-07-17 11:02:00 UTC - Pilot O spectral-only simulator result

Pilot O kept Pilot N's setup but removed training-time clipping entirely, leaving spectral filtering plus RIR/noise as the only added hard-effect family.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline tuning-dev | 14.87 | 1.97 | 6.82 | 14.06 | 36.64 | 1.00 | 0.00% | 0.00% |
| Pilot N u100 deletion-safe | 15.09 | 2.14 | 6.91 | 13.52 | 37.78 | 1.00 | -1.46% | -1.20% |
| Pilot O u50 spectral-only | 15.94 | 1.97 | 6.97 | 13.64 | 41.19 | 1.33 | -7.21% | -7.93% |
| Pilot O u100 spectral-only | 15.14 | 2.14 | 6.91 | 13.40 | 38.10 | 1.00 | -1.79% | -1.66% |

Resource use: 100 updates, 370.7 s training wall time, 144.7/163.4 s u50/u100 evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.42 GB.

Decision: reject Pilot O. It improves mid WER more than Pilot N at u100, but low WER and low insertions remain worse than baseline. Spectral-only calibration is not promotable.

## 2026-07-17 11:42:00 UTC - No-repeat decoder control and Pilot Q consistency

All rows below use the same frozen `tuning_dev_proxy` and `no_repeat_ngram_size=3`; this is a diagnostic decoder control, not an official or top-7 claim.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | control |
| Pilot D u100 no-repeat-3 | 14.04 | 2.17 | 7.15 | 13.82 | 33.04 | 0.83 | 2.37% | 3.57% | fail |
| Pilot E u100 no-repeat-3 | 14.25 | 2.21 | 7.06 | 13.88 | 33.86 | 1.00 | 0.92% | 1.74% | fail |
| Pilot F u100 no-repeat-3 | 14.17 | 2.07 | 7.12 | 13.67 | 33.83 | 1.17 | 1.47% | 2.24% | fail |
| Pilot P u100 no-repeat-3 | 14.08 | 2.21 | 7.12 | 13.85 | 33.13 | 0.83 | 2.15% | 3.31% | fail |
| Pilot Q u50 consistency | 14.34 | 2.04 | 7.09 | 14.15 | 34.09 | 0.83 | 0.31% | 0.73% | fail |
| Pilot Q u100 consistency | 14.16 | 2.17 | 7.12 | 13.97 | 33.39 | 0.83 | 1.55% | 2.54% | fail |

Pilot Q resources: 100 updates, 273.8 s training wall time, 119.2/121.1 s u50/u100 evaluations, peak train/eval VRAM 1.45/1.18 GB allocated, project disk 20.46 GB.

Decision: reject Pilot Q and do not promote any current checkpoint to full proxy or full training. Consistency stabilized low deletions compared with Pilot E/F, but it weakened low substitution gains and still missed the required 5% combined mid/low improvement. Pilot D remains the closest controlled acoustic SFT result, but it is not promotable.

## 2026-07-17 12:05:00 UTC - Pilot R LayerNorm adaptation

Pilot R adds trainable encoder LayerNorm scale/bias parameters in layers 14-31 to the Pilot D attention-LoRA recipe. Adapter saving was extended to include explicitly trainable non-LoRA tensors.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | control |
| Pilot D u100 no-repeat-3 | 14.04 | 2.17 | 7.15 | 13.82 | 33.04 | 0.83 | 2.37% | 3.57% | fail |
| Pilot R u50 LN | 14.30 | 2.04 | 7.12 | 14.17 | 33.86 | 1.00 | 0.59% | 1.13% | fail |
| Pilot R u100 LN | 14.39 | 2.77 | 7.09 | 13.97 | 33.74 | 1.00 | -0.05% | 1.82% | fail |

Pilot R resources: 100 updates, 274.8 s training wall time, 119.1/120.8 s u50/u100 evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.48 GB.

Decision: reject Pilot R. LayerNorm adaptation increases low substitution gains but introduces dry deletions and low deletions; it is worse than Pilot D and not promotable.

## 2026-07-17 12:28:00 UTC - Pilot S rank-32 attention LoRA

Pilot S keeps the Pilot D recipe and attention-only target family, but increases LoRA rank from 16 to 32 with alpha 64.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | control |
| Pilot D rank-16 u100 | 14.04 | 2.17 | 7.15 | 13.82 | 33.04 | 0.83 | 2.37% | 3.57% | fail |
| Pilot S rank-32 u50 | 14.31 | 2.14 | 7.12 | 14.06 | 33.93 | 0.67 | 0.51% | 1.24% | fail |
| Pilot S rank-32 u100 | 15.19 | 3.94 | 9.00 | 14.29 | 33.51 | 1.00 | -5.58% | 1.61% | fail |

Pilot S resources: 100 updates, 269.8 s training wall time, 120.4/131.4 s u50/u100 evaluations, peak train/eval VRAM 1.44/1.18 GB allocated, project disk 20.51 GB.

Decision: reject Pilot S. Rank 32 lowers catastrophic rate at u50 but does not improve mid/low enough; by u100 it over-adapts dry/high badly. The closest current candidate remains Pilot D rank-16 u100 with no-repeat-3 decoding, but it is still not promotable.

## 2026-07-17 12:43:00 UTC - Adapter interpolation diagnostics

Created two evaluation-only interpolated adapters from compatible rank-16 attention-LoRA checkpoints:

- Pilot T: `0.5 * Pilot D + 0.5 * Pilot P`.
- Pilot U: `0.5 * Pilot D + 0.5 * Pilot F`.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | control |
| Pilot D no-repeat-3 | 14.04 | 2.17 | 7.15 | 13.82 | 33.04 | 0.83 | 2.37% | 3.57% | fail |
| Pilot T D/P interp | 14.11 | 2.17 | 7.12 | 13.91 | 33.26 | 0.83 | 1.88% | 2.93% | fail |
| Pilot U D/F interp | 14.15 | 2.21 | 7.21 | 13.88 | 33.32 | 1.00 | 1.61% | 2.86% | fail |

Decision: reject interpolation branch. Interpolation smooths some deletion behavior but weakens the useful substitution gains and does not beat Pilot D.

## 2026-07-17 13:01:00 UTC - Pilot V condition-weighted CE

Pilot V keeps Pilot D fixed except for loss weights on degraded examples: dry/high 1.0, mid 1.5, low 2.0. Clean anchor remains normalized.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | control |
| Pilot D no-repeat-3 | 14.04 | 2.17 | 7.15 | 13.82 | 33.04 | 0.83 | 2.37% | 3.57% | fail |
| Pilot V u50 | 14.32 | 2.04 | 7.12 | 14.12 | 33.99 | 0.83 | 0.47% | 0.99% | fail |
| Pilot V u100 | 14.24 | 2.21 | 7.15 | 13.85 | 33.77 | 0.83 | 0.99% | 2.00% | fail |

Pilot V resources: 100 updates, 267.7 s training wall time, 119.0/119.5 s u50/u100 evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.53 GB.

Decision: reject Pilot V. Condition weighting did not increase mid/low gains beyond Pilot D and still shows low deletion drift.

## 2026-07-17 13:13:00 UTC - Beam decoder diagnostic

Evaluated baseline and Pilot D with identical decoder settings: `num_beams=3`, `length_penalty=1.1`, `no_repeat_ngram_size=3`.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline beam3 lp1.1 ng3 | 13.94 | 2.07 | 7.09 | 13.22 | 33.39 | 1.17 | 0.00% | 0.00% | control |
| Pilot D beam3 lp1.1 ng3 | 13.63 | 2.14 | 6.56 | 12.95 | 32.88 | 1.50 | 2.24% | 1.67% | fail |

Decision: reject beam decoder branch. Pilot D average improves under beam search, but the mid/low gate still fails and catastrophic rate worsens.

## 2026-07-17 13:42:00 UTC - Pilot W decoder cross-attention LoRA

Pilot W adds decoder cross-attention q/v/out LoRA to the Pilot D encoder-attention recipe. This is the first decoder-conditioning pilot and does not use MWER/GRPO.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | control |
| Pilot D no-repeat-3 | 14.04 | 2.17 | 7.15 | 13.82 | 33.04 | 0.83 | 2.37% | 3.57% | fail |
| Pilot W u50 | 14.22 | 2.07 | 6.97 | 14.15 | 33.71 | 0.67 | 1.12% | 1.52% | fail |
| Pilot W u100 | 14.35 | 2.81 | 8.08 | 13.82 | 32.72 | 0.83 | 0.21% | 4.23% | fail |

Pilot W resources: 100 updates, 289.8 s training wall time, 138.6/143.9 s u50/u100 evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.55 GB.

Decision: reject Pilot W as configured. It gives the strongest low WER so far at u100, but dry/high over-adaptation prevents promotion. The result justifies a lower-LR decoder-cross-attention pilot rather than abandoning the decoder-conditioning branch.

## 2026-07-17 14:18:00 UTC - Pilots X/Y/Z decoder-cross lower-LR branch

Pilots X and Z keep the Pilot W encoder+decoder-cross LoRA target set but lower LR from `1e-5` to `5e-6`. Pilot Y adds mid/low condition CE weights on top of X. All were evaluated on frozen `tuning_dev_proxy` with the same no-repeat-3 decoder control.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | 0.00 | control |
| Pilot X u150 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.83% | +0.07 | fail |
| Pilot Y u150 weighted | 14.19 | 2.07 | 7.06 | 13.97 | 33.64 | 1.00 | 1.39% | 2.02% | +0.03 | fail |
| Pilot Z u175 | 14.22 | 2.71 | 7.18 | 13.82 | 33.16 | 1.00 | 1.17% | 3.31% | +0.67 | fail |
| Pilot Z u200 | 14.65 | 3.21 | 8.19 | 14.00 | 33.20 | 1.00 | -1.83% | 2.88% | +1.17 | fail |
| Pilot Z u225 | 14.85 | 3.44 | 9.06 | 14.00 | 32.91 | 1.00 | -3.24% | 3.47% | +1.40 | fail |

Pilot X resources: 150 updates, 428.6 s training wall time, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.58 GB.

Pilot Y resources: 150 updates, 429.4 s training wall time, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.60 GB.

Pilot Z resources: 225 updates, 638.8 s training wall time, 142.8/145.2/152.4 s u175/u200/u225 evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.63 GB.

Decision: reject Pilots X/Y/Z for promotion. Pilot X u150 is the current best diagnostic checkpoint and passes average, dry, and catastrophic gates, but combined mid/low improvement is 3.83% versus the required 5%. Y weakens the signal and raises catastrophic rate. Z confirms that extending the lower-LR run past 150 updates overfits dry/high before reaching the mid/low gate. No full training, MWER/GRPO, router, full-proxy promotion evaluation, or submission is justified yet.

## 2026-07-17 14:38:00 UTC - Pilots AA/AB adapter arithmetic diagnostics

Added `scripts/combine_adapters.py` for reproducible scalar combination of compatible safetensors adapters. Tested two evaluation-only probes on the X/W-compatible decoder-cross target set.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | 0.00 | control |
| Pilot X u150 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.83% | +0.07 | fail |
| Pilot AA 0.8X+0.2W | 14.16 | 2.64 | 7.18 | 13.76 | 33.07 | 0.83 | 1.55% | 3.63% | +0.60 | fail |
| Pilot AB 1.1X | 14.25 | 2.64 | 7.39 | 13.70 | 33.26 | 1.00 | 0.96% | 3.36% | +0.60 | fail |

AA/AB resources: evaluation-only, 140.1/142.5 s wall time, peak eval VRAM 1.18 GB allocated, project disk 20.64 GB. Tests passed after the helper addition: `python -m pytest tests/test_metrics.py tests/test_labels.py -q`.

Decision: reject adapter arithmetic branch. Blending toward W or scaling X does not cross the mid/low gate and either loses average improvement or worsens catastrophic rate. Pilot X u150 remains the best near-miss.

## 2026-07-17 15:05:00 UTC - Pilot AC dropout-regularized decoder-cross SFT

Pilot AC keeps Pilot X fixed except for LoRA dropout `0.05 -> 0.10`.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | 0.00 | control |
| Pilot X u150 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.83% | +0.07 | fail |
| Pilot AC u75 | 14.24 | 2.07 | 6.94 | 13.97 | 33.96 | 1.00 | 1.04% | 1.36% | +0.03 | fail |
| Pilot AC u100 | 14.17 | 2.11 | 6.94 | 14.23 | 33.39 | 0.67 | 1.51% | 1.99% | +0.07 | fail |
| Pilot AC u150 | 14.06 | 2.11 | 7.06 | 13.64 | 33.42 | 0.67 | 2.29% | 3.15% | +0.07 | fail |

Pilot AC resources: 150 updates, 427.4 s training wall time, 139.6/139.5/141.2 s u75/u100/u150 evaluations, peak train/eval VRAM 1.43/1.18 GB allocated, project disk 20.67 GB.

Decision: reject Pilot AC for promotion. Dropout regularization reduces catastrophic output rate and keeps dry/high stable, but weakens low-condition adaptation relative to X and still misses the 5% combined mid/low gate. AC is useful as a stable control, not a promoted candidate.

## 2026-07-18 02:30:00 UTC - Pilot AD decode-only min_new_tokens sweep

Added `min_new_tokens` support to `src/evaluation/proxy_eval.py` decoding kwargs. Tested a floor of 3/5/8 tokens on top of Pilot X u150, greedy `no_repeat_ngram_size=3`, no retraining.

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | 0.00 | control |
| Pilot X u150 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.86% | +0.07 | fail |
| Pilot AD min_new_tokens=3 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.86% | +0.07 | fail (no-op) |
| Pilot AD min_new_tokens=5 | 13.99 | 2.11 | 7.15 | 13.76 | 32.94 | 0.83 | 2.76% | 3.92% | +0.07 | fail |
| Pilot AD min_new_tokens=8 | 14.11 | 2.11 | 7.24 | 13.94 | 33.16 | 1.17 | 1.90% | 3.09% | +0.07 | fail (catastrophic gate too) |

Paired analysis (min_new_tokens=5 vs no-repeat baseline, 600 samples): low condition delta_s=-17, delta_d=+6, delta_i=-24 — deletions still rise by the same +6 seen in Pilot X alone, confirming the floor barely touches the deletion pattern.

Resources: evaluation-only, 141.5-143.9 s wall time per run, project disk 23.64 GB.

Decision: reject Pilot AD. A global decode-time length floor cannot resolve the low-condition deletion pattern and actively hurts short dry/high utterances at floor=8. The next diagnostic must act during training (length/coverage-aware loss), not at decode time.

## 2026-07-18 03:20:00 UTC - Pilot AE EOS-suppression training loss

Added `eos_suppress_weight` to `src/training/sft.py`: an auxiliary term equal to the mean predicted probability of the EOS token at every non-final, non-padding label position of the degraded pass, added to the loss with weight 25.0. Otherwise identical to Pilot X (LR 5e-6, dropout 0.05, same target set, same curriculum).

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 2.04 | 6.91 | 14.53 | 34.06 | 0.83 | 0.00% | 0.00% | 0.00 | control |
| Pilot X u150 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.86% | +0.07 | fail |
| Pilot AE u75 | 14.25 | 2.01 | 7.03 | 14.09 | 33.90 | 1.33 | 0.91% | 1.29% | -0.03 | fail |
| Pilot AE u100 | 14.23 | 2.04 | 7.03 | 14.03 | 33.83 | 1.17 | 1.06% | 1.54% | 0.00 | fail |
| Pilot AE u150 | 14.23 | 2.07 | 7.12 | 13.85 | 33.86 | 1.00 | 1.10% | 1.86% | +0.03 | fail |

Low-condition raw deletion counts (out of 150 low samples): baseline 112, Pilot X u150 119, Pilot AE u150 **94**. This is the first pilot to push low deletions below the no-repeat baseline, confirming the EOS-suppression hypothesis mechanistically. But low substitutions rose from baseline's 817 to 832 (Pilot X was 800), so the freed probability mass lands on wrong words, not the right ones, and the net WER is worse than Pilot X.

Resources: 150 updates, 427.7 s training wall time, peak train/eval VRAM 1.47/1.18 GB allocated, project disk 23.64 GB.

Decision: reject Pilot AE at weight=25.0 (fails avg, mid/low, and catastrophic-rate gates). The mechanism is validated and worth retrying at a lower weight or gated to low/mid conditions only in a future pilot, but not pursued further this session.

## 2026-07-18 04:10:00 UTC - Pilots AF/AG eos_suppress weight sweep + condition gating (parallel run)

Ran 4 training jobs concurrently on the same GPU (peak combined ~8.5GB VRAM / 24GB card, GPU util 100%, ~1694s wall time per job under contention vs ~430s run alone -- still a net wall-clock win over sequential since all 4 finished in ~28 minutes together instead of ~2 hours back-to-back). Added `eos_suppress_conditions` gating to `src/training/sft.py` for Pilot AG.

| run | avg | mid/low rel improvement | catastrophic % | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 0.00% | 0.83 | 0.00 | control |
| Pilot X u150 (weight=0) | 14.00 | 3.86% | 0.83 | +0.07 | fail |
| Pilot AF weight=3 u150 | 14.10 | 2.83% | 0.83 | +0.03 | fail |
| Pilot AF weight=8 u150 | 14.16 | 2.19% | 1.00 | 0.00 | fail |
| Pilot AF weight=12 u150 | 14.13 | 2.70% | 1.00 | 0.00 | fail |
| Pilot AG weight=15 (mid/low gated) u150 | 14.13 | 2.44% | 1.17 | 0.00 | fail |
| Pilot AE weight=25 u150 | 14.23 | 1.86% | 1.00 | +0.03 | fail |

Every nonzero weight underperforms Pilot X on mid/low improvement, and 3 of 4 also raise catastrophic rate above X's 0.83%. There is no dose-response sweet spot in this range; gating to mid/low conditions does not rescue the approach either.

Decision: close the eos_suppress_weight branch. Pilot X u150 remains the best near-miss (mid/low 3.86% vs the 5% gate). Next diagnostic should change the training data/curriculum mix directly rather than add further auxiliary EOS-probability loss terms.

## 2026-07-18 05:00:00 UTC - Pilot AH data-mix sweep (low share 25/30/40%, parallel run)

Ran 3 training jobs concurrently (peak combined ~4.3GB VRAM), changing only the late-stage curriculum's `condition_distribution` (mid fixed at 0.30, dry=high reduced to compensate for a larger low share). No loss changes, no decode changes -- isolates the effect of raw data exposure.

| run | avg | mid/low rel improvement | catastrophic % | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 0.00% | 0.83 | 0.00 | control |
| Pilot X u150 (low=0.20) | 14.00 | 3.86% | 0.83 | +0.07 | fail |
| Pilot AH low=0.25 u150 | 14.08 | 3.02% | 0.67 | +0.07 | fail |
| Pilot AH low=0.30 u150 | 14.00 | 3.73% | 1.00 | +0.07 | fail |
| Pilot AH low=0.40 u150 | 14.00 | 3.54% | 0.67 | +0.03 | fail |

All three levels land within a few tenths of a point of Pilot X (3.02-3.73% vs 3.86%) -- inside likely single-seed sampling noise on the 600-sample proxy set. Pilot AH low=0.40 has the best catastrophic rate (0.67%) and dry regression (+0.03) of the branch but does not beat X on mid/low.

Decision: reject Pilot AH. Direct data-mix reweighting does not unlock gain beyond Pilot X either. Combined with the eos_suppress and CE-weighting branches, the recipe now looks plateaued around 3.5-3.9% mid/low improvement across every training-side lever tried. See docs/DECISIONS.md for the full closing rationale and recommended next step (a structural recipe change, to be scoped with the user).

## 2026-07-18 06:00:00 UTC - Pilot AI: audit and fix train/eval augmentation mismatch (parallel run)

Audited `artifacts/manifests/librispeech_train_clean_diag/manifest.jsonl` (Pilot X's training data: 2400 LibriSpeech train.100 utterances, 22 speakers, 53 chapters) against `configs/eval/proxy.yaml` (the frozen eval set's degradation config) and found two quantified gaps:

- **SNR floor**: Pilot X's low-condition training SNR never goes below 0dB (late-stage `[0,6]`, early-stage `[3,8]`). The eval manifest's low-condition SNR is uniform `[-3,6]`: **28.7%** of the 150 low eval samples are below 0dB, **40.7%** below 1dB -- entirely unseen during training.
- **Missing effects config**: Pilot X (and its whole V/W/X/Y/Z/AA-AH lineage) sets no `augmentation.effects` block, so training falls back to weak defaults: 0% spectral filtering for `high` (eval: 15%), 0% clipping for `high`/`mid` (eval: 2%/8%), and frame dropout never triggers (eval applies it to ~10% of low samples, hardcoded in `src/evaluation/proxy_eval.py`).

Ran 3 parallel training jobs fixing these on top of Pilot X's exact recipe:

| run | avg | mid/low rel improvement | catastrophic % | dry regression | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline no-repeat-3 | 14.39 | 0.00% | 0.83 | 0.00 | control |
| Pilot X u150 (unfixed) | 14.00 | 3.86% | 0.83 | +0.07 | fail |
| Pilot AI snrfix u150 | 14.10 | 2.83% | 0.67 | +0.07 | fail |
| Pilot AI effectsfix u150 | 14.10 | 2.96% | 1.00 | +0.07 | fail |
| Pilot AI combined u150 | 14.12 | 2.70% | 0.83 | +0.07 | fail |

Resources: 150 updates, 1079.2 s training wall time per job (parallel), peak train VRAM 1.43 GB per job, project disk 23.94 GB.

Decision: reject Pilot AI. The mismatches are real and worth fixing for production training, but closing them does not cross Pilot X on the current 150-update budget -- all three land 0.9-1.2 points below X on mid/low improvement. Augmentation-severity mismatch is ruled out as the primary blocker for this recipe. Pilot X u150 remains the best result overall.

## 2026-07-18 09:30:00 UTC - Pilot AJ/AK/AJK: attack the acoustic bottleneck (parallel run)

Error-driven diagnosis first. Pilot X's residual low-condition WER scales monotonically with the physical difficulty of each sample:

| slice | mean low-condition WER |
| --- | ---: |
| SNR < 0dB (n=43) | 48.2 |
| SNR 0-3dB (n=62) | 36.5 |
| SNR 3-6dB (n=45) | 24.6 |
| reverb rt60 > 0.8 (n=43) | 47.0 |
| reverb rt60 < 0.5 (n=59) | 28.3 |

That signature (plus the zero-sum substitution<->deletion trade seen in AC/AE) says the encoder representation of hard samples lacks acoustic information. Two levers were tried in parallel, both new code in `src/training/sft.py`:

- **AJ** — per-frame clean->degraded encoder feature distillation (`feature_distill_weight=1.0`, gated to high/mid/low): a proper version of Pilot Q's coarse pooled-mean consistency.
- **AK** — add encoder FFN (`fc1`/`fc2`) to the LoRA target set (1.72M -> 4.08M trainable): capacity to reshape features, not just reweight them.
- **AJK** — both.

| run | avg | dry | mid/low rel improvement | avg rel improvement | dry regression | catastrophic % | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot X u150 (control) | 14.00 | 2.11 | 3.86% | 2.70% | +0.07 | 0.83 | fail |
| Pilot AJ u100 | 14.17 | 2.11 | 2.12% | 1.49% | +0.07 | 0.83 | fail |
| Pilot AJ u150 | 13.99 | 2.07 | 3.86% | 2.74% | +0.03 | 1.17 | fail (ties X) |
| Pilot AK u75 | 14.49 | 2.84 | 2.44% | -0.70% | +0.80 | 0.83 | fail |
| Pilot AK u150 | 17.92 | 4.71 | -14.6% | -24.6% | +2.67 | 1.00 | fail (overfit) |
| Pilot AJK u75 | 14.44 | 2.84 | 2.25% | -0.40% | +0.80 | 0.83 | fail |
| Pilot AJK u150 | 18.56 | 4.98 | -19.0% | -29.0% | +2.94 | 1.17 | fail (overfit) |

Encoder-FFN capacity (AK/AJK) overfits catastrophically -- training CE drops well below X (0.68 vs 0.84) but proxy WER blows up to 17.9/18.6 with dry regressing +2.7/+2.9, refuting the "capacity-limited" hypothesis. Feature distillation (AJ) is neutral: it ties X almost exactly (low S/D/I 798/118/125 vs X's 800/119/116) and slightly worsens catastrophic rate.

Decision: reject all three. This is the strongest evidence yet that the ceiling is **training-data scale**, not the recipe -- more capacity memorizes the tiny 2400-utterance / 22-speaker / 8.6h pool, and a better clean-target signal changes nothing. See docs/DECISIONS.md for the closing rationale and the recommended data-acquisition next step.
