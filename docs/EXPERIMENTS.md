# Experiment Registry

| Run | Hypothesis | Changed variable | Command | Result | Decision | Next action |
|---|---|---|---|---|---|---|
| R0 | Public baseline row is retrievable from official Space | none | `bash scripts/fetch_ffasr_leaderboard.sh` | saved `artifacts/leaderboard/leaderboard_latest.csv`; top-5 16.35, top-7 17.34 | accept | full official reevaluation remains external submission action |
| R1 | LiteWhisper HF checkpoint supports trainable LoRA path | Phase-0 LoRA smoke | `bash scripts/phase0_trainability.sh configs/experiments/phase0.yaml` | pass; loss 8.5781, 788,480 trainable params, reload max abs 0.0, merge max abs 0.03516, peak VRAM 2.21 GB | accept core trainability | build bounded proxy/data path |
| D0 | Manifest downloader can materialize bounded audio without large downloads | tiny Parquet LibriSpeech fixture | `bash scripts/download_bounded_data.sh configs/data/librispeech_smoke.yaml` | pass; 4 wavs written, disk total 3.64 GB | accept smoke | use `configs/data/librispeech_bounded.yaml` only after deciding corpus budget |
| R0-mini | Establish mini-proxy baseline control | fixed 400-sample proxy | `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/baseline_mini.yaml` | Avg 20.45; dry/high/mid/low 1.82/7.76/22.02/50.19; catastrophic 3.5%; peak VRAM 1.17 GB | accept control | run full proxy baseline |
| R0-full | Establish full-proxy baseline control | fixed 2,000-sample proxy | `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/baseline_full.yaml` | Avg 18.02; dry/high/mid/low 1.67/7.29/17.80/45.31; catastrophic 1.65%; peak VRAM 1.17 GB | accept control | produce error analysis and run LoRA Level-0 |

## 2026-07-17 04:43:41 UTC - R2 smoke: Pilot A 2-update SFT

- Command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_a_smoke2.yaml`
- Result: pass, 2 updates / 16 micro-batches, wall time 12.3 s.
- Peak VRAM: 1.33 GB allocated, 1.42 GB reserved.
- Adapter: `artifacts/runs/pilot_a_train_smoke2/adapter.safetensors`.
- Decision: proceed to Level-0 Pilot A.

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

## 2026-07-17 07:56:37 UTC - Pilot D normalized clean-anchor lambda=1.0

- GPU blocker cleared by user before launch; `nvidia-smi` showed GPU 0 idle.
- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_d_clean_anchor_lam1_l0.yaml`
- Eval command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_d_clean_anchor_lam1_l0_mini.yaml`
- Training: 100 updates / 800 micro-batches, wall 277.0 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved.
- Mini-proxy WER: avg 20.14, dry 1.93, high 7.86, mid 22.13, low 48.63.
- Baseline mini WER: avg 20.45, dry 1.82, high 7.76, mid 22.02, low 50.19.
- Average relative improvement: 1.52%; mid/low relative improvement: 2.01%; dry regression: +0.11 absolute.
- Catastrophic output rate: 3.25% vs baseline 3.50%.
- Paired regressions: 35 improved, 31 regressed, 334 unchanged. Low improved with S/D/I deltas -2/-15/-8, while dry/high had small regressions.
- Decision: reject Pilot D for promotion. It is directionally better than pre-fix pilots and improves low WER, but fails the required >=2% average and >=5% mid/low promotion gates. Continue to controlled Pilot E (`lambda_clean=0.25`) before changing other variables.

## 2026-07-17 08:12:00 UTC - Pilot E normalized clean-anchor lambda=0.25

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_e_clean_anchor_lam025_l0.yaml`
- Eval command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_e_clean_anchor_lam025_l0_mini.yaml`
- Training: 100 updates / 800 micro-batches, wall 368.3 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved.
- Mini-proxy WER: avg 20.31, dry 1.93, high 7.86, mid 22.69, low 48.75.
- Average relative improvement: 0.68%; mid/low relative improvement: 1.06%; dry regression: +0.11 absolute.
- Catastrophic output rate: 4.00% vs baseline 3.50%.
- Paired regressions: 36 improved, 37 regressed, 327 unchanged. Low improved with S/D/I deltas -2/-14/-7, but mid regressed with +17 insertions.
- Decision: reject Pilot E. Lowering the clean-anchor weight did not increase useful acoustic adaptation and worsened catastrophic rate. Next diagnostic is a corrected-label degraded-CE-only mild-curriculum pilot to isolate whether the clean-anchor branch is under-adapting the model or whether the simulator/target modules are the bottleneck.

## 2026-07-17 08:20:00 UTC - Pilot F degraded-CE-only diagnostic

- User clarified that the VLLM server belongs to another workload and should not be stopped; future FFASR runs may ignore that contention.
- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_f_degraded_ce_mild_l0.yaml`
- Eval command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_f_degraded_ce_mild_l0_mini.yaml`
- Training: 100 updates / 800 micro-batches, wall 148.1 s, peak VRAM 1.31 GB allocated / 1.48 GB reserved.
- Mini-proxy WER: avg 20.39, dry 1.93, high 8.06, mid 22.58, low 49.00.
- Average relative improvement: 0.27%; mid/low relative improvement: 0.87%; dry regression: +0.11 absolute.
- Catastrophic output rate: 4.00% vs baseline 3.50%.
- Paired regressions: 33 improved, 36 regressed, 331 unchanged. Low improved with S/D/I deltas +3/-14/-8, but mid regressed with +15 insertions and high gained +10 substitutions.
- Decision: reject Pilot F. The clean-anchor branch is not the main blocker; removing it reduces useful low gains and increases catastrophic outputs. The best corrected-label short pilot remains Pilot D. Next diagnostic should test LoRA target capacity by adding encoder FFN output LoRA to Pilot D's recipe at the same LR/curriculum.

## 2026-07-17 08:34:00 UTC - Pilot G encoder attention + FFN output LoRA

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_g_attn_ffn_clean_lam1_l0.yaml`
- Eval command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_g_attn_ffn_clean_lam1_l0_mini.yaml`
- Training: 100 updates / 800 micro-batches, wall 388.8 s, peak VRAM 1.50 GB allocated / 1.56 GB reserved, 2.66M trainable parameters.
- Mini-proxy WER: avg 26.42, dry 4.76, high 12.37, mid 27.45, low 61.11.
- Catastrophic output rate: 3.00% vs baseline 3.50%, but WER regressions dominate.
- Paired regressions: 45 improved, 165 regressed, 190 unchanged. S/D/I deltas vs baseline: dry +40/+13/+2, high +79/+8/+5, mid +67/+33/-3, low +49/+21/+105.
- Decision: reject Pilot G. Adding encoder `fc2` LoRA sharply lowers training CE but over-adapts the model and damages WER, especially substitutions and low-condition insertions. Keep encoder-attention-only as the active target family. Next diagnostic: run Pilot D's attention-only recipe for 300 updates with saved intermediate adapters to determine whether the 100-update result was under-trained.

## 2026-07-17 08:50:00 UTC - Pilot H attention-only 300-update undertraining check

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_h_attn_clean_lam1_300.yaml`
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_h_attn_clean_lam1_u150_mini.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_h_attn_clean_lam1_u300_mini.yaml`
- Hypothesis: Pilot D may have been under-trained at 100 updates; continuing the same encoder-attention-only, normalized clean-anchor recipe should improve mid/low WER without excessive dry regression.
- Training: 300 updates / 2,400 micro-batches, wall 945.6 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved, project disk 19.79 GB.
- Checkpoints retained: `artifacts/runs/pilot_h_attn_clean_lam1_300/adapter_update_150.safetensors`, `artifacts/runs/pilot_h_attn_clean_lam1_300/adapter_update_300.safetensors`, and final `adapter.safetensors`.

Mini-proxy results:

| run | avg | dry | high | mid | low | score | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 22.20 | 3.50 |
| Pilot D u100 | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 21.76 | 3.25 |
| Pilot H u150 | 21.17 | 3.53 | 9.86 | 22.35 | 48.94 | 25.72 | 4.25 |
| Pilot H u300 | 25.06 | 5.29 | 12.37 | 28.91 | 53.68 | 32.52 | 3.00 |

Decision: reject Pilot H. The undertraining hypothesis is false for this recipe: low WER stays slightly better than baseline at 150 updates, but dry/high regress sharply, and by 300 updates all conditions except catastrophic rate are worse. Do not promote to a full run or full-proxy evaluation. The best corrected-label robust pilot remains Pilot D at 100 updates, but it still fails promotion gates.

## 2026-07-17 09:10:00 UTC - Pilot I lower-LR attention-only optimization check

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_i_attn_clean_lam1_lr5e6_150.yaml`
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_i_attn_clean_lam1_lr5e6_u75_mini.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_i_attn_clean_lam1_lr5e6_u100_mini.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_i_attn_clean_lam1_lr5e6_u150_mini.yaml`
- Hypothesis: lowering LR to `5e-6` may slow the attention-only overfit observed in Pilot H while preserving Pilot D's low-condition gain.
- Training: 150 updates / 1,200 micro-batches, wall 421.2 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved, project disk 19.81 GB.
- Checkpoints retained: `artifacts/runs/pilot_i_attn_clean_lam1_lr5e6_150/adapter_update_75.safetensors`, `adapter_update_100.safetensors`, `adapter_update_150.safetensors`, and final `adapter.safetensors`.

Mini-proxy results:

| run | avg | dry | high | mid | low | score | catastrophic % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline mini | 20.45 | 1.82 | 7.76 | 22.02 | 50.19 | 22.20 | 3.50 |
| Pilot D u100 LR 1e-5 | 20.14 | 1.93 | 7.86 | 22.13 | 48.63 | 21.76 | 3.25 |
| Pilot I u75 LR 5e-6 | 20.22 | 1.76 | 7.46 | 21.79 | 49.88 | 22.10 | 3.75 |
| Pilot I u100 LR 5e-6 | 20.33 | 1.76 | 7.76 | 21.79 | 50.00 | 22.33 | 4.00 |
| Pilot I u150 LR 5e-6 | 20.71 | 1.76 | 7.76 | 21.74 | 51.56 | 22.71 | 4.00 |

Decision: reject Pilot I for promotion. Lower LR controls dry/high regression better than Pilot H, but the best checkpoint (u75) improves average WER only 1.1% and combined mid/low only 0.8% relative, while catastrophic output rate rises from 3.50% to 3.75%. No full-proxy evaluation or full training is justified.

## 2026-07-17 09:35:00 UTC - Tuning-dev proxy and simulator audit

- Built `artifacts/manifests/proxy/tuning_dev_proxy.jsonl`: 600 samples, 150 dry/high/mid/low, disjoint from train/mini/full proxy by utterance and speaker.
- Baseline command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/baseline_tuning_dev.yaml`, then `python -m src.evaluation.summarize_errors configs/eval/baseline_tuning_dev.yaml` after fixing nullable `baseline_dry_wer`.
- Baseline tuning-dev WER: avg 14.87, dry 1.97, high 6.82, mid 14.06, low 36.64, catastrophic 1.00%.
- Simulator audit found previous training missed proxy highpass/bandpass coverage. Baseline tuning-dev error slices showed soft_clip hardest (45.97 WER), followed by bandpass (22.56), highpass (21.33), and lowpass (19.08).
- Implemented configurable train-time spectral, clipping, and frame-dropout effects in `src/training/sft.py`, plus `src/analysis/simulator_audit.py`.

## 2026-07-17 09:53:00 UTC - Pilot J simulator-calibrated SFT

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_j_simcal_attn_clean_lam1_l0.yaml`
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_j_simcal_u50_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_j_simcal_u75_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_j_simcal_u100_tuning_dev.yaml`
- Training: 100 updates / 800 micro-batches, wall 328.0 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved, 1.23M trainable parameters, disk 20.37 GB.
- Results: u50 avg 15.26, u75 avg 15.99, u100 avg 14.72 against baseline 14.87.
- Decision: reject for promotion. U100 improves avg by 1.0% and mid/low by 1.7% only; catastrophic rate worsens to 1.17%. Paired errors show mid/low substitutions drop, but low deletions rise by 47 words.

## 2026-07-17 10:06:00 UTC - Target-locality pilots K/L

- Pilot K command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_k_lower_available_attn_simcal.yaml` then `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_k_lower_available_u100_tuning_dev.yaml`.
- Pilot K: layers 14-20 attention targets only; 100 updates, wall 283.0 s, peak VRAM 1.39 GB, 245,760 trainable parameters. Tuning-dev avg 15.60, dry 2.17, high 6.97, mid 13.88, low 39.38, catastrophic 1.00%. Rejected.
- Pilot L command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_l_middle_attn_simcal.yaml` then `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_l_middle_u100_tuning_dev.yaml`.
- Pilot L: layers 21-26 attention targets only; 100 updates, wall 213.2 s, peak VRAM 1.35 GB, 491,520 trainable parameters. Tuning-dev avg 15.97, dry 2.11, high 7.15, mid 13.67, low 40.94, catastrophic 1.33%. Rejected.
- Next: run late-only Pilot M (layers 27-31) to complete target-locality comparison.

## 2026-07-17 10:24:00 UTC - Pilot M late target-locality result

- Pilot M command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_m_late_attn_simcal.yaml` then `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_m_late_u100_tuning_dev.yaml`.
- Hypothesis: restricting LoRA to late encoder attention layers may keep the useful top-layer adaptation seen in Pilot J while reducing deletion regressions.
- Changed variable: target regex limited to encoder layers 27-31 attention projections.
- Control: baseline tuning-dev and Pilot J full available encoder-attention targets.
- Training: 100 updates / 800 micro-batches, wall 259.4 s, peak VRAM 1.32 GB allocated / 1.48 GB reserved, 491,520 trainable parameters.
- Tuning-dev WER: avg 15.45, dry 2.01, high 7.03, mid 13.85, low 38.90, catastrophic 1.00%.
- Promotion deltas vs tuning-dev baseline: avg -3.86%, mid/low -4.05%, dry +0.03 absolute.
- Paired analysis: 31 improved, 32 regressed, 537 unchanged. Low S/D/I delta -8/+2/+77.
- Decision: reject. Late-only adaptation avoids dry collapse but creates a low-condition insertion regression and is worse than baseline and Pilot J. The locality branch is closed for now; next pilot should change the simulator/objective to address low deletions/insertions rather than selecting another encoder layer subset.

## 2026-07-17 10:43:00 UTC - Pilot N deletion-safe simulator

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_n_deletion_safe_simcal.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_n_deletion_safe_u50_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_n_deletion_safe_u100_tuning_dev.yaml`
- Hypothesis: removing frame dropout and softening low-condition severity will preserve Pilot J's spectral gains while avoiding low deletion/insertion regressions.
- Changed variable: simulator severity only; no frame dropout, low SNR softened, clipping probability and drive reduced.
- Control: baseline tuning-dev and Pilot J u100.
- Training: 100 updates / 800 micro-batches, wall 289.0 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved, 1.23M trainable parameters.
- Results:
  - u50: avg 15.26, dry 1.97, high 6.97, mid 13.64, low 38.45, catastrophic 1.33%.
  - u100: avg 15.09, dry 2.14, high 6.91, mid 13.52, low 37.78, catastrophic 1.00%.
- Promotion deltas for u100 vs tuning-dev baseline: avg -1.46%, mid/low -1.20%, dry +0.17.
- Decision: reject. The change improves mid WER but worsens low WER through insertions and some deletions. The next controlled acoustic pilot should isolate clipping by keeping spectral coverage but removing both dropout and clipping.

## 2026-07-17 11:02:00 UTC - Pilot O spectral-only simulator

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_o_spectral_only_simcal.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_o_spectral_only_u50_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_o_spectral_only_u100_tuning_dev.yaml`
- Hypothesis: removing clipping as well as dropout will preserve mid/low substitution gains without low insertion drift.
- Changed variable: spectral filtering retained, clipping and frame dropout disabled.
- Control: baseline tuning-dev, Pilot J, and Pilot N.
- Training: 100 updates / 800 micro-batches, wall 370.7 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved, 1.23M trainable parameters.
- Results:
  - u50: avg 15.94, dry 1.97, high 6.97, mid 13.64, low 41.19, catastrophic 1.33%.
  - u100: avg 15.14, dry 2.14, high 6.91, mid 13.40, low 38.10, catastrophic 1.00%.
- Promotion deltas for u100 vs tuning-dev baseline: avg -1.79%, mid/low -1.66%, dry +0.17.
- Decision: reject. Removing clipping does not solve low insertion drift. The next work should inspect whether the augmented-training condition mix is causing hypothesis-length bias on low-condition audio rather than continuing severity tweaks.

## 2026-07-17 11:20:00 UTC - No-repeat decoder control diagnostics

- Implemented optional `repetition_penalty`, `no_repeat_ngram_size`, `length_penalty`, and `early_stopping` fields in `src/evaluation/proxy_eval.py`; defaults preserve previous decoding.
- Controlled baseline: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/baseline_tuning_dev_decode_ng3.yaml`.
- Baseline no-repeat-3 result: avg 14.39, dry 2.04, high 6.91, mid 14.53, low 34.06, catastrophic 0.83%.
- Pilot D no-repeat-3 is the closest current result: avg 14.04, dry 2.17, high 7.15, mid 13.82, low 33.04, catastrophic 0.83%. It passes the average gate but misses the mid/low gate at 3.57% relative.
- Pilot E no-repeat-3: avg 14.25, mid/low improvement 1.74%, catastrophic 1.00%; rejected.
- Pilot F no-repeat-3: avg 14.17, mid/low improvement 2.24%, catastrophic 1.17%; rejected.
- Pilot P no-repeat-3: avg 14.08, mid/low improvement 3.31%, catastrophic 0.83%; rejected.

## 2026-07-17 11:42:00 UTC - Pilot Q paired consistency diagnostic

- Code change: `src/training/sft.py` now supports `loss.consistency_weight`; it mean-pools valid encoder frames and applies `1 - cosine(pool(degraded), stop_gradient(pool(clean)))`.
- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_q_consistency_lam1_l0.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_q_u50_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_q_u100_decode_ng3_tuning_dev.yaml`
- Training: 100 updates / 800 micro-batches, wall 273.8 s, peak VRAM 1.45 GB allocated / 1.65 GB reserved, 1.23M trainable parameters, disk 20.46 GB.
- Results: u50 avg 14.34, dry/high/mid/low 2.04/7.09/14.15/34.09, catastrophic 0.83%; u100 avg 14.16, dry/high/mid/low 2.17/7.12/13.97/33.39, catastrophic 0.83%.
- Decision: reject. Pilot Q improves average only 1.55% and combined mid/low only 2.54% at u100. It does not beat Pilot D and does not pass promotion gates.

## 2026-07-17 12:05:00 UTC - Pilot R LayerNorm adaptation

- Code change: `src/training/sft.py` supports `trainable_parameter_regex`; `src/models/lora.py` saves explicitly trainable non-LoRA tensors with adapters.
- Initial bug: unfrozen LayerNorm parameters were FP16 after `model.to(dtype=float16)`, causing `GradScaler` to raise `ValueError: Attempting to unscale FP16 gradients`.
- Fix: explicitly cast regex-unfrozen parameters to FP32 before training.
- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_r_attn_lora_ln_l0.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_r_u50_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_r_u100_decode_ng3_tuning_dev.yaml`
- Training: 100 updates / 800 micro-batches, wall 274.8 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved, 1.32M trainable parameters, disk 20.48 GB.
- Results: u50 avg 14.30, dry/high/mid/low 2.04/7.12/14.17/33.86, catastrophic 1.00%; u100 avg 14.39, dry/high/mid/low 2.77/7.09/13.97/33.74, catastrophic 1.00%.
- Decision: reject. LayerNorm adaptation worsened dry WER and catastrophic rate, and did not improve mid/low enough to pass.

## 2026-07-17 12:28:00 UTC - Pilot S rank-32 attention LoRA

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_s_attn_rank32_l0.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_s_u50_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_s_u100_decode_ng3_tuning_dev.yaml`
- Training: 100 updates / 800 micro-batches, wall 269.8 s, peak VRAM 1.44 GB allocated / 1.61 GB reserved, 2.46M trainable parameters, disk 20.51 GB.
- Results: u50 avg 14.31, dry/high/mid/low 2.14/7.12/14.06/33.93, catastrophic 0.67%; u100 avg 15.19, dry/high/mid/low 3.94/9.00/14.29/33.51, catastrophic 1.00%.
- Decision: reject. Increasing rank within attention-only LoRA gives a small low gain and lower catastrophic rate at u50, but does not pass average or mid/low gates. At u100 it over-adapts clean/high speech.

## 2026-07-17 12:43:00 UTC - Adapter interpolation diagnostics

- Created:
  - `artifacts/runs/pilot_t_interp_d_p_50/adapter.safetensors`
  - `artifacts/runs/pilot_u_interp_d_f_50/adapter.safetensors`
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_t_interp_d_p_50_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_u_interp_d_f_50_decode_ng3_tuning_dev.yaml`
- Pilot T D/P interpolation: avg 14.11, dry/high/mid/low 2.17/7.12/13.91/33.26, catastrophic 0.83%.
- Pilot U D/F interpolation: avg 14.15, dry/high/mid/low 2.21/7.21/13.88/33.32, catastrophic 1.00%.
- Decision: reject. Neither interpolation beats Pilot D, and neither passes the mid/low gate.

## 2026-07-17 13:01:00 UTC - Pilot V condition-weighted CE

- Code change: `src/training/sft.py` supports `loss.condition_ce_weights`; default behavior remains weight 1.0 for all conditions.
- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_v_condition_weighted_l0.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_v_u50_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_v_u100_decode_ng3_tuning_dev.yaml`
- Training: 100 updates / 800 micro-batches, wall 267.7 s, peak VRAM 1.43 GB allocated / 1.59 GB reserved, 1.23M trainable parameters, disk 20.53 GB.
- Results: u50 avg 14.32, dry/high/mid/low 2.04/7.12/14.12/33.99, catastrophic 0.83%; u100 avg 14.24, dry/high/mid/low 2.21/7.15/13.85/33.77, catastrophic 0.83%.
- Decision: reject. The objective change is stable but weaker than Pilot D on average and mid/low.

## 2026-07-17 13:13:00 UTC - Beam decoder diagnostic

- Baseline command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/baseline_tuning_dev_beam3_lp11_ng3.yaml`.
- Pilot D command: `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_d_u100_beam3_lp11_ng3_tuning_dev.yaml`.
- Baseline beam3/lp1.1/ng3: avg 13.94, dry/high/mid/low 2.07/7.09/13.22/33.39, catastrophic 1.17%.
- Pilot D beam3/lp1.1/ng3: avg 13.63, dry/high/mid/low 2.14/6.56/12.95/32.88, catastrophic 1.50%.
- Decision: reject. Beam search helps average WER but increases catastrophic-output rate and does not improve combined mid/low enough.

## 2026-07-17 13:42:00 UTC - Pilot W decoder cross-attention LoRA

- Target audit: decoder has four layers with exposed `encoder_attn.{q_proj,v_proj,out_proj}` modules; Pilot W adds these 12 LoRA targets to Pilot D's 30 encoder attention targets.
- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_w_decoder_cross_attn_l0.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_w_u50_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_w_u100_decode_ng3_tuning_dev.yaml`
- Training: 100 updates / 800 micro-batches, wall 289.8 s, peak VRAM 1.43 GB allocated / 1.60 GB reserved, 1.72M trainable parameters, disk 20.55 GB.
- Results: u50 avg 14.22, dry/high/mid/low 2.07/6.97/14.15/33.71, catastrophic 0.67%; u100 avg 14.35, dry/high/mid/low 2.81/8.08/13.82/32.72, catastrophic 0.83%.
- Decision: reject u100 for promotion due to dry/high regression, but keep decoder cross-attention as an active branch because low WER and low insertions improved.

## 2026-07-17 14:18:00 UTC - Pilots X/Y/Z lower-LR decoder-cross branch

- Pilot X command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_x_decoder_cross_attn_lr5e6_150.yaml`.
- Pilot X eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_x_u75_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_x_u100_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_x_u150_decode_ng3_tuning_dev.yaml`
- Pilot Y command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_y_decoder_cross_attn_lr5e6_weighted_150.yaml`.
- Pilot Y eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_y_u100_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_y_u150_decode_ng3_tuning_dev.yaml`
- Pilot Z command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_z_decoder_cross_attn_lr5e6_225.yaml`.
- Pilot Z eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_z_u175_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_z_u200_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_z_u225_decode_ng3_tuning_dev.yaml`

Hypothesis: lower LR should preserve Pilot W's low-condition gain while avoiding dry/high over-adaptation. Y additionally tests whether explicit mid/low CE weights can push combined mid/low over the promotion gate. Z tests whether the lower-LR trajectory crosses the mid/low gate after 150 updates.

Results against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot X u150 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.83% | reject, best near-miss |
| Pilot Y u150 | 14.19 | 2.07 | 7.06 | 13.97 | 33.64 | 1.00 | 1.39% | 2.02% | reject |
| Pilot Z u175 | 14.22 | 2.71 | 7.18 | 13.82 | 33.16 | 1.00 | 1.17% | 3.31% | reject |
| Pilot Z u200 | 14.65 | 3.21 | 8.19 | 14.00 | 33.20 | 1.00 | -1.83% | 2.88% | reject |
| Pilot Z u225 | 14.85 | 3.44 | 9.06 | 14.00 | 32.91 | 1.00 | -3.24% | 3.47% | reject |

Resource use stayed inside budget: training peak allocated VRAM 1.43 GB for X/Y/Z, evaluation peak allocated VRAM 1.18 GB, project disk 20.63 GB after Z. Regression tests passed after paired analysis: `python -m pytest tests/test_metrics.py tests/test_labels.py -q`.

Conclusion: decoder cross-attention LoRA is the strongest branch so far, but it still fails the explicit mid/low gate. Do not promote to full run. The next useful acoustic pilot should target the remaining high/dry regression in decoder-cross adaptation or improve low-condition gains without longer training.

## 2026-07-17 14:38:00 UTC - Pilots AA/AB adapter arithmetic diagnostics

- Code change: added `scripts/combine_adapters.py` to combine compatible safetensors adapters with scalar weights.
- Pilot AA adapter command:
  - `python scripts/combine_adapters.py --input 0.8:artifacts/runs/pilot_x_decoder_cross_attn_lr5e6_150/adapter.safetensors --input 0.2:artifacts/runs/pilot_w_decoder_cross_attn_l0/adapter.safetensors --output artifacts/runs/pilot_aa_interp_x80_w20/adapter.safetensors`
- Pilot AB adapter command:
  - `python scripts/combine_adapters.py --input 1.1:artifacts/runs/pilot_x_decoder_cross_attn_lr5e6_150/adapter.safetensors --output artifacts/runs/pilot_ab_scale_x110/adapter.safetensors`
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_aa_interp_x80_w20_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ab_scale_x110_decode_ng3_tuning_dev.yaml`

Results against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot AA 0.8X+0.2W | 14.16 | 2.64 | 7.18 | 13.76 | 33.07 | 0.83 | 1.55% | 3.63% | reject |
| Pilot AB 1.1X | 14.25 | 2.64 | 7.39 | 13.70 | 33.26 | 1.00 | 0.96% | 3.36% | reject |

Conclusion: adapter arithmetic did not reveal a hidden promotable point. AA retains catastrophic rate but loses X's average gain; AB nudges mid WER but worsens low/high and catastrophic output rate. Keep X as the near-miss control.

## 2026-07-17 15:05:00 UTC - Pilot AC dropout-regularized decoder-cross SFT

- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_ac_decoder_cross_dropout10_150.yaml`.
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ac_u75_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ac_u100_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ac_u150_decode_ng3_tuning_dev.yaml`

Hypothesis: increasing LoRA dropout from 0.05 to 0.10 may regularize the decoder-cross target set enough to preserve dry/high while retaining X's mid/low gains.

Results against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot AC u75 | 14.24 | 2.07 | 6.94 | 13.97 | 33.96 | 1.00 | 1.04% | 1.36% | reject |
| Pilot AC u100 | 14.17 | 2.11 | 6.94 | 14.23 | 33.39 | 0.67 | 1.51% | 1.99% | reject |
| Pilot AC u150 | 14.06 | 2.11 | 7.06 | 13.64 | 33.42 | 0.67 | 2.29% | 3.15% | reject |

Resource use stayed inside budget: 427.4 s training wall time, peak allocated train/eval VRAM 1.43/1.18 GB, project disk 20.67 GB. Regression tests passed after paired analysis: `python -m pytest tests/test_metrics.py tests/test_labels.py -q`.

Conclusion: dropout regularizes catastrophic behavior but under-adapts low speech compared with X. It is not promotable.

## 2026-07-18 02:30:00 UTC - Pilot AD decode-time min_new_tokens floor

- Code change: added `min_new_tokens` to the optional decode kwargs in `src/evaluation/proxy_eval.py:transcribe()`.
- Eval commands (Pilot X u150 adapter, no retraining):
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ad_minnt3_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ad_minnt5_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ad_minnt8_decode_ng3_tuning_dev.yaml`

Hypothesis: low-condition deletions (seen rising across every training pilot so far, e.g. Pilot AC low delta_d +6) come from premature EOS; forcing a minimum generation length should recover them cheaply without retraining.

Results against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot AD min_new_tokens=3 | 14.00 | 2.11 | 7.15 | 13.76 | 32.97 | 0.83 | 2.70% | 3.86% | reject (no-op) |
| Pilot AD min_new_tokens=5 | 13.99 | 2.11 | 7.15 | 13.76 | 32.94 | 0.83 | 2.76% | 3.92% | reject |
| Pilot AD min_new_tokens=8 | 14.11 | 2.11 | 7.24 | 13.94 | 33.16 | 1.17 | 1.90% | 3.09% | reject |

Conclusion: a global min_new_tokens floor is the wrong lever — it cannot tell a naturally short dry reference from a truncated low-condition one, so raising the floor trades a few recovered low deletions for more forced insertions/substitutions elsewhere (catastrophic rate 0.83%->1.17% at floor=8). Confirms the deletion pattern is training-time, not decode-time.

## 2026-07-18 03:20:00 UTC - Pilot AE EOS-suppression training loss

- Code change: added `eos_suppression_penalty()` and `loss.eos_suppress_weight` support to `src/training/sft.py`. The penalty is the mean softmax probability assigned to the EOS token at every label position that is not padding and not the true final token, computed on the degraded (not clean-anchor) forward pass and added to the training loss with a scalar weight.
- Train command: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_ae_eos_suppress_150.yaml` (Pilot X recipe + `eos_suppress_weight: 25.0`, single changed variable).
- Eval commands:
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ae_u75_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ae_u100_decode_ng3_tuning_dev.yaml`
  - `CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_ae_u150_decode_ng3_tuning_dev.yaml`

Hypothesis: Pilot AC's paired analysis showed low-condition deletions rising even as substitutions/insertions fell (delta_d +6). If that is early-stop bias, explicitly penalizing EOS probability at non-final positions during training should recover those deletions.

Results against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | dry | high | mid | low | catastrophic % | avg rel improvement | mid/low rel improvement | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Pilot AE u75 | 14.25 | 2.01 | 7.03 | 14.09 | 33.90 | 1.33 | 0.91% | 1.29% | reject |
| Pilot AE u100 | 14.23 | 2.04 | 7.03 | 14.03 | 33.83 | 1.17 | 1.06% | 1.54% | reject |
| Pilot AE u150 | 14.23 | 2.07 | 7.12 | 13.85 | 33.86 | 1.00 | 1.10% | 1.86% | reject |

Conclusion: the EOS-suppression term works exactly as hypothesized on deletions (low deletions 119->94, below even the no-repeat baseline's 112) but weight=25.0 over-corrects: the recovered probability mass goes to substitutions (+15 vs baseline in low) and insertions instead of the right word, and catastrophic rate climbs to 1.0-1.33%. Net WER is worse than Pilot X on every gate. This is the first pilot to isolate and directly move the deletion count, so a lower-weight or condition-gated follow-up is a reasonable next step, but not run this session.

## 2026-07-18 04:10:00 UTC - Pilots AF/AG eos_suppress weight sweep and condition gating

- Code change: added `loss.eos_suppress_conditions` (optional list) to `src/training/sft.py` so the EOS-suppression penalty can be gated to specific augmentation conditions instead of applying to every micro-batch.
- 4 training+eval jobs launched in parallel on the same GPU (`&` background processes, all `CUDA_VISIBLE_DEVICES=0`):
  - `configs/train/pilot_af_w3_150.yaml` (eos_suppress_weight=3.0, all conditions)
  - `configs/train/pilot_af_w8_150.yaml` (eos_suppress_weight=8.0, all conditions)
  - `configs/train/pilot_af_w12_150.yaml` (eos_suppress_weight=12.0, all conditions)
  - `configs/train/pilot_ag_w15_midlow_150.yaml` (eos_suppress_weight=15.0, eos_suppress_conditions=[mid, low])
  - Each followed by `scripts/evaluate_proxy.sh` on its u75/u100/u150 checkpoints.

Hypothesis: Pilot AE's weight=25.0 over-corrected; a smaller dose, or restricting the penalty to the conditions that actually have a deletion problem (mid/low), should recover the deletion fix without AE's substitution/catastrophic-rate cost.

Best (u150) result per run against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | mid/low rel improvement | catastrophic % | decision |
| --- | ---: | ---: | ---: | --- |
| Pilot X (weight=0) | 14.00 | 3.86% | 0.83 | fail (near-miss, best so far) |
| Pilot AF weight=3 | 14.10 | 2.83% | 0.83 | reject |
| Pilot AF weight=8 | 14.16 | 2.19% | 1.00 | reject |
| Pilot AF weight=12 | 14.13 | 2.70% | 1.00 | reject |
| Pilot AG weight=15 (gated) | 14.13 | 2.44% | 1.17 | reject |

Conclusion: there is no weight in [3, 25] where eos_suppress_weight beats Pilot X, and gating to mid/low conditions does not change that conclusion. This rules out the auxiliary-EOS-probability-loss family as a promotion path for this branch. Parallelizing the 4 runs on one GPU (peak combined ~8.5GB VRAM) completed the whole sweep in ~28 minutes instead of ~2 hours sequential.

## 2026-07-18 05:00:00 UTC - Pilot AH data-mix share sweep (parallel)

- No code change; only `configs/train/pilot_ah_low{25,30,40}_150.yaml` differ from Pilot X in the late-stage `augmentation.curriculum[1].condition_distribution`: `low` in {0.25, 0.30, 0.40}, `mid` fixed at 0.30, `dry`/`high` reduced equally to compensate (e.g. low=0.30 -> dry=high=0.20). Early-stage curriculum (updates 0-50) untouched.
- 3 training+eval jobs launched in parallel on the same GPU (peak combined ~4.3GB VRAM):
  - `configs/train/pilot_ah_low25_150.yaml`
  - `configs/train/pilot_ah_low30_150.yaml`
  - `configs/train/pilot_ah_low40_150.yaml`
  - Each followed by `scripts/evaluate_proxy.sh` on u75/u100/u150.

Hypothesis: since CE-weighting (Pilot V/Y) and an auxiliary EOS-probability loss (AE/AF/AG) both failed to push mid/low past X, maybe the model simply needs more raw exposure to low-condition audio rather than a reweighted signal on the same exposure.

Best (u150) result per run against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | mid/low rel improvement | catastrophic % | decision |
| --- | ---: | ---: | ---: | --- |
| Pilot X (low=0.20) | 14.00 | 3.86% | 0.83 | fail (near-miss, best so far) |
| Pilot AH low=0.25 | 14.08 | 3.02% | 0.67 | reject |
| Pilot AH low=0.30 | 14.00 | 3.73% | 1.00 | reject |
| Pilot AH low=0.40 | 14.00 | 3.54% | 0.67 | reject |

Conclusion: raising the low-condition sampling share (up to 2x) does not cross Pilot X, and the spread across 25/30/40% is small enough to be sampling noise rather than a trend. Combined with every other lever tried (LR, dropout, training length, CE weighting, adapter arithmetic, decode floors, EOS-suppression loss, data mix), this recipe -- encoder self-attn + decoder cross-attn LoRA, rank 16, on the current augmentation pipeline -- looks plateaued around 3.5-3.9% mid/low improvement. Recommend a structural change (larger/different LoRA target set, higher rank, or different data source) as the next diagnostic, scoped with the user rather than run automatically.

## 2026-07-18 06:00:00 UTC - Pilot AI: training-data-source audit and augmentation-mismatch fix (parallel)

- No src code change; audited `artifacts/manifests/librispeech_train_clean_diag/manifest.jsonl` (2400 samples, 22 speakers, 53 chapters, LibriSpeech train.100) and Pilot X's `augmentation` block against `configs/eval/proxy.yaml` and `src/evaluation/proxy_eval.py`'s degradation logic.
- Found: Pilot X's low-condition SNR floor (0dB late-stage) misses 28.7% of eval low samples (SNR < 0dB) and 40.7% (SNR < 1dB); Pilot X sets no `augmentation.effects`, so eval-realistic clipping (high/mid) and frame dropout (~10% of low samples) are never trained on. Noted Pilot J tried effects coverage earlier but on an older recipe (pre decoder-cross-attn, higher LR) and was rejected as insufficient -- never retested on X's current best recipe.
- 3 training+eval jobs launched in parallel on the same GPU:
  - `configs/train/pilot_ai_snrfix_150.yaml` -- low SNR range widened to match eval exactly (late-stage `[-3,6]`), no effects change.
  - `configs/train/pilot_ai_effectsfix_150.yaml` -- `augmentation.effects` added to match `configs/eval/proxy.yaml`'s spectral/clipping probabilities plus `dropout_probability.low=0.08`, no SNR change.
  - `configs/train/pilot_ai_combined_150.yaml` -- both fixes together.
  - Each followed by `scripts/evaluate_proxy.sh` on u75/u100/u150.

Best (u150) result per run against `baseline_tuning_dev_decode_ng3_proxy`:

| run | avg | mid/low rel improvement | catastrophic % | decision |
| --- | ---: | ---: | ---: | --- |
| Pilot X (unfixed) | 14.00 | 3.86% | 0.83 | fail (best so far) |
| Pilot AI snrfix | 14.10 | 2.83% | 0.67 | reject |
| Pilot AI effectsfix | 14.10 | 2.96% | 1.00 | reject |
| Pilot AI combined | 14.12 | 2.70% | 0.83 | reject |

Conclusion: closing the identified train/eval augmentation gaps does not cross Pilot X -- if anything it costs 0.9-1.2 points of mid/low improvement at the same 150-update budget (plausibly because harder augmentation needs more updates to converge, untested here, or because LoRA capacity is the real ceiling). This rules out augmentation-severity mismatch as the primary blocker on this recipe. The gaps are still worth fixing for any eventual production run, but they are not this branch's answer. Closes the data-source review requested by the user.
