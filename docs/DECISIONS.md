# Decisions

- Use the FFASR Space Gradio API for public leaderboard retrieval because the
  underlying storage bucket rejects anonymous direct dataset access.
- Do not submit or publish externally without explicit authorization.
- Keep project-local caches under `.cache/` to make disk accounting explicit.
- Start with custom Linear LoRA support in addition to PEFT because LiteASR uses
  remote custom modules and compression-specific layers.
- Treat FP16 merged-export logit drift up to 0.05 max absolute as acceptable for
  Phase-0 export smoke; observed drift was 0.03515625. WER equivalence still
  needs a real audio proxy before final export.

## 2026-07-17 04:49:53 UTC - Pilot A promotion decision

Pilot A is not promotable on mini-proxy evidence: average WER worsened from 20.45 to 23.19, dry WER regressed by 2.35 absolute, and low WER did not improve. Continue to the controlled Pilot B comparison before choosing the target family.

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


## 2026-07-17 05:27:13 UTC - Label prefix blocker diagnosed

Symptom: prior SFT pilots degraded dry/high WER quickly or failed to improve degraded speech at lower LR.

Root cause found during teacher-forcing audit: `processor.tokenizer(text=...)` can omit `<|en|><|transcribe|>` depending on tokenizer prefix state, producing labels with only `<|startoftranscript|><|notimestamps|>...<|endoftext|>`. Evaluation decoding attempts English/transcribe prompts, so training and decoding were not reliably aligned.

Fix: `src.training.sft.make_labels` now explicitly calls `set_prefix_tokens(language='en', task='transcribe', predict_timestamps=False)` before tokenization and masks padding to `-100`. Regression test `tests/test_labels.py` verifies the first four label IDs are `[50258, 50259, 50360, 50364]`.

Affected experiments: all previous SFT pilots are retained as rejected pre-fix runs and must not be used for promotion decisions except as evidence of the label-path bug.


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

## 2026-07-17 07:56:37 UTC - Pilot D promotion decision

Reject Pilot D (`lambda_clean=1.0`) for promotion. It improved low WER from 50.19 to 48.63 and reduced catastrophic output rate from 3.50% to 3.25%, but the average improvement was only 1.52% and the combined mid/low improvement was only 2.01%. Both miss the explicit pilot gates. The next controlled run remains Pilot E with identical settings except `lambda_clean=0.25`.

## 2026-07-17 08:12:00 UTC - Pilot E promotion decision

Reject Pilot E (`lambda_clean=0.25`) for promotion. It improved average WER only 0.68%, combined mid/low WER only 1.06%, and increased catastrophic output rate from 3.50% to 4.00%. Compared with Pilot D, the lower clean-anchor weight did not produce more useful acoustic adaptation and worsened mid insertions. Run one short degraded-CE-only corrected-label diagnostic before modifying simulator calibration or LoRA target scope.

## 2026-07-17 08:20:00 UTC - Pilot F promotion decision

Reject Pilot F (degraded CE only) for promotion. It improved average WER only 0.27%, combined mid/low WER only 0.87%, and increased catastrophic output rate to 4.00%. Removing the clean anchor did not improve adaptation; the current blocker is more likely LoRA target capacity or simulator/condition targeting than clean-anchor weighting. Continue with a short encoder attention + FFN output LoRA diagnostic using Pilot D's normalized clean-anchor recipe.

## 2026-07-17 08:34:00 UTC - Pilot G promotion decision

Reject Pilot G (encoder attention + `fc2` LoRA). It worsened mini-proxy average WER from 20.45 to 26.42 and regressed all conditions. The lower training CE did not indicate better ASR behavior; paired errors show broad substitution increases and severe low-condition insertion growth. Do not promote encoder FFN output LoRA in this configuration. Keep the active candidate family as encoder-attention-only with normalized clean anchor.

## 2026-07-17 08:50:00 UTC - Pilot H promotion decision

Reject Pilot H (attention-only normalized clean-anchor continued to 150/300 updates). Pilot H u150 retains a small low-condition gain but dry regression is +1.71 absolute and catastrophic output rate increases to 4.25%. Pilot H u300 broadly overfits and worsens average WER to 25.06. The 100-update Pilot D result was not under-trained; longer training at LR 1e-5 damages clean/high behavior before producing promotable mid/low gains.

Do not run full training, MWER, router, or submission. The diagnosed blocker is now objective/condition mismatch rather than basic trainability: clean-only CE sanity passes, but robust SFT with the current simulator and target modules produces only small low-condition deletion/insertion reductions while increasing substitutions or dry/high errors. The next safe branch should be another short controlled acoustic pilot, not a promoted run.

## 2026-07-17 09:10:00 UTC - Pilot I promotion decision

Reject Pilot I (attention-only normalized clean-anchor, LR `5e-6`). Lowering LR prevents the severe dry/high collapse seen in Pilot H, but it does not create a promotable acoustic adaptation signal. The best checkpoint, u75, has avg 20.22 vs baseline 20.45 and mid/low 35.83 vs 36.10, below both promotion gates, and catastrophic output rate increases from 3.50% to 3.75%.

Current branch conclusion: with corrected labels, encoder-attention LoRA is trainable and can make small, stable changes, but the current simulator/objective/target setup is too weak or mismatched to beat the baseline meaningfully. Do not run full training, MWER, router, submission, or full-proxy evaluation from Pilots D/H/I. The next experiment should change the acoustic signal itself, not simply run longer or add broad FFN capacity.

## 2026-07-17 09:35:00 UTC - Preserve proxy v1 and add tuning-dev proxy

Preserve old mini/full proxy results for historical comparison, but use the new frozen `tuning_dev_proxy` for diagnosis because it is speaker/utterance-disjoint from train and old proxies. Do not claim FFASR top-7/top-5 progress from tuning-dev because baseline avg 14.87 is much easier than official public baseline 26.04.

## 2026-07-17 09:53:00 UTC - Reject Pilot J for promotion

Pilot J fixed a simulator coverage mismatch by adding highpass/bandpass/dropout-like effects to training, but it failed promotion gates on tuning-dev. U100 improves avg by only 1.0% and combined mid/low by 1.7%, while catastrophic output rate increases to 1.17%. Do not evaluate Pilot J on old mini/full as a candidate. Use it as diagnostic evidence that hard-effect coverage alone is insufficient.

## 2026-07-17 10:06:00 UTC - Reject lower and middle locality pilots

Pilot K (layers 14-20) is too weak: only six v_proj targets attach and low WER worsens to 39.38. Pilot L (layers 21-26) is worse still, with low WER 40.94 and catastrophic rate 1.33%. Continue to late-only Pilot M to complete the controlled target-locality comparison; do not run full training, MWER, router, or submission.

## 2026-07-17 10:24:00 UTC - Reject late locality and close locality branch

Pilot M (layers 27-31) is rejected. It keeps dry nearly stable, but tuning-dev average WER worsens from 14.87 to 15.45 and low WER worsens from 36.64 to 38.90. Paired analysis shows only sparse movement and a low-condition insertion increase of 77 words.

Target locality alone does not produce a promotable acoustic SFT configuration. Pilot J's full attention target set remains the strongest SFT diagnostic, but it still fails gates. The next controlled pilot should modify simulator severity/objective behavior to reduce low-condition deletion/insertion regressions; do not start full training, MWER, router, full-proxy promotion evaluation, or submission.

## 2026-07-17 10:43:00 UTC - Reject Pilot N deletion-safe simulator

Pilot N is rejected. It removed frame dropout and softened low severity, but u100 still worsened average WER from 14.87 to 15.09 and low WER from 36.64 to 37.78. Catastrophic rate recovered to baseline, but the promotion gate requires real average and mid/low gains.

Paired analysis shows the simulator change helped mid substitutions and low substitutions, but low insertions increased by 56 and low deletions by 8 at u100. This indicates that dropout was not the only source of insertion/deletion drift. The next safe pilot should isolate training-time clipping: use spectral-only calibration with no dropout and no clipping before considering any objective change, MWER, or full training.

## 2026-07-17 11:02:00 UTC - Reject Pilot O spectral-only simulator

Pilot O is rejected. Removing clipping did not remove low insertion drift. U100 improves mid WER to 13.40 but worsens low WER to 38.10 and average WER to 15.14. U50 is worse, with low WER 41.19.

The simulator-only branch has now tested hard-effect coverage, dropout removal, clipping reduction, clipping removal, and target locality. None produced a promotable acoustic SFT configuration. The diagnosed blocker is not basic trainability or one isolated hard effect; it is a mismatch between token-level CE adaptation and low-condition decoding behavior. The next safe branch should be a short non-MWER acoustic/objective diagnostic, not full training or submission.

## 2026-07-17 11:42:00 UTC - Reject no-repeat-controlled E/F/P/Q; no promotion

Reject Pilot E, Pilot F, Pilot P, and Pilot Q under the controlled no-repeat-3 decoder. The common control is `baseline_tuning_dev_decode_ng3_proxy` with avg 14.39 and catastrophic 0.83%.

Pilot D remains the closest controlled acoustic SFT result: avg improves 2.37% and catastrophic rate stays flat, but combined mid/low improves only 3.57%, below the 5% gate. Pilot Q consistency reduced the low deletion increase seen in E/F, but total mid/low improvement was only 2.54% and average improvement only 1.55%.

Decision: do not run full training, full-proxy promotion evaluation, MWER/GRPO, router, submission, or publication from these checkpoints. The first promotable acoustic SFT configuration has not been obtained yet.

## 2026-07-17 12:05:00 UTC - Reject LayerNorm adaptation

Reject Pilot R. Selected LayerNorm adaptation in encoder layers 14-31 is technically supported now, but it is not a promotion path: u100 dry WER regresses from 2.04 to 2.77 under no-repeat-3, catastrophic rate rises from 0.83% to 1.00%, and combined mid/low improves only 1.82%.

Decision: do not broaden trainable non-LoRA parameters without a stronger regularizer or a narrower hypothesis. The first promotable acoustic SFT configuration remains unresolved.

## 2026-07-17 12:28:00 UTC - Reject rank-32 attention-only LoRA

Reject Pilot S. Rank 32 does not solve the mid/low gate. The u50 checkpoint has lower catastrophic rate but only 0.51% average and 1.24% mid/low relative improvement. The u100 checkpoint over-adapts, with dry WER 3.94 and high WER 9.00 under the no-repeat-3 control.

Decision: keep Pilot D rank-16 u100 as the best current diagnostic candidate, but do not promote it. No full training, MWER/GRPO, router, submission, or official-result claim is justified.

## 2026-07-17 12:43:00 UTC - Reject adapter interpolation branch

Reject Pilot T and Pilot U. Interpolating D with P or F does not uncover a better point on the adapter trajectory. D/P preserves catastrophic rate but weakens mid/low improvement to 2.93%; D/F raises catastrophic rate to 1.00%.

Decision: do not use adapter interpolation as the first promotable acoustic SFT configuration. Continue only with genuinely new evidence-based objective/simulator changes.

## 2026-07-17 13:01:00 UTC - Reject condition-weighted CE

Reject Pilot V. Increasing degraded CE weight for mid and low conditions does not overcome the mid/low gate. U100 improves combined mid/low only 2.00% and average only 0.99%, both worse than Pilot D.

Decision: condition weighting alone is not a promotion path. The unresolved blocker is low deletion drift when acoustic substitutions improve.

## 2026-07-17 13:13:00 UTC - Reject beam decoder branch

Reject the `beam3 + length_penalty=1.1 + no-repeat-3` decoder branch. It improves Pilot D average WER by 2.24% relative to the same decoder baseline, but combined mid/low improves only 1.67% and catastrophic rate worsens from 1.17% to 1.50%.

Decision: do not use beam search as the first promotable configuration. The failure pattern still points to model-side decoder conditioning or deletion-safe training rather than pure decoding.

## 2026-07-17 13:42:00 UTC - Reject Pilot W at LR 1e-5 but continue decoder-conditioning branch

Reject Pilot W u100 for promotion. It improves low WER to 32.72 and combined mid/low by 4.23%, the closest mid/low result so far, but average improvement is only 0.21% and dry regression is +0.77, just over the +0.7 gate. High WER also regresses to 8.08.

Decision: a lower-LR decoder-cross-attention pilot is justified. Do not promote W, do not run full training, and do not start MWER/GRPO.

## 2026-07-17 14:18:00 UTC - Reject lower-LR decoder-cross branch for promotion

Reject Pilots X, Y, and Z for promotion. Pilot X u150 is the current best diagnostic checkpoint: avg improves 2.70%, dry regression is only +0.07, and catastrophic rate stays flat at 0.83%, but combined mid/low improves only 3.83% against the required 5%. Pilot Y condition weights weaken the result and raise catastrophic rate to 1.00%. Pilot Z shows that extending the same lower-LR recipe overfits dry/high before reaching the mid/low gate.

Decision: do not run a 1.5-3 hour full run, full-proxy promotion evaluation, MWER/GRPO, router, submission, or official-format claim from this branch. Keep Pilot X u150 as the best near-miss control for the next acoustic diagnostic. The diagnosed blocker is now narrower: decoder-cross adaptation can reduce low insertions and substitutions, but the useful mid/low gain saturates below the gate while dry/high errors rise with additional updates.

## 2026-07-17 14:38:00 UTC - Reject adapter arithmetic branch

Reject Pilots AA and AB. A 0.8X+0.2W blend and a 1.1X scale-up both fail the same promotion gate as X. AA loses average improvement and nearly hits the dry-regression limit; AB increases catastrophic rate to 1.00%.

Decision: do not spend more time on adapter arithmetic for this target family. The first promotable acoustic SFT configuration remains unresolved. The next justified work is a new training diagnostic that changes regularization or data mix around Pilot X, not full training or sequence-level MWER.

## 2026-07-17 15:05:00 UTC - Reject Pilot AC dropout regularization

Reject Pilot AC. Increasing LoRA dropout to 0.10 reduces catastrophic rate to 0.67% at u100/u150 and preserves dry/high, but mid/low relative improvement remains 3.15% at best, below the 5% gate and weaker than Pilot X.

Decision: Pilot X u150 remains the best near-miss by promotion criteria. Pilot AC is a useful stable control showing the tradeoff: stronger regularization protects output stability but loses low-condition adaptation. Do not promote, do not run full training, and do not start MWER/GRPO/router.

## 2026-07-18 02:30:00 UTC - Reject Pilot AD decode-time min_new_tokens floor

Reject Pilot AD. A global `min_new_tokens` floor (3/5/8) applied to Pilot X u150 decoding does not fix the low-condition deletion pattern: floor=5 only removes 1 of 119 low deletions (pooled mid/low improvement 3.92% vs X's 3.86%, still short of the 5% gate); floor=8 forces extra tokens onto short dry/high utterances, raising catastrophic rate from 0.83% to 1.17% and pooled mid/low improvement to only 3.09%.

Decision: do not pursue decode-time length floors further; a global floor cannot distinguish naturally short dry references from truncated low-condition outputs. Paired analysis confirms the deletion excess in low-condition audio is a training-time effect, so the next diagnostic (Pilot AE) adds a length/coverage-aware loss term during SFT instead of touching decoding.

## 2026-07-18 03:20:00 UTC - Reject Pilot AE EOS-suppression loss, but mechanism confirmed

Reject Pilot AE (eos_suppress_weight=25.0 added to Pilot X's training loss). It does what it was designed to do: low-condition deletions fall from Pilot X's 119 to 94 (below even the no-repeat baseline's 112). But the recovered probability mass goes to substitutions and insertions instead of correct words, so pooled mid/low improvement drops to 1.86% (u150) versus X's 3.86%, average improvement falls to 1.10% (misses the 2% gate), and catastrophic rate rises to 1.0-1.33% (worse than X's 0.83%).

Decision: do not promote Pilot AE at weight=25.0. This is the first pilot in the branch to directly cut low-condition deletion count below baseline, so the mechanism is worth a follow-up at a much lower weight or gated to mid/low conditions only, but do not spend further budget on it this session. Pilot X u150 remains the best near-miss control. No full training, MWER/GRPO, router, or promotion evaluation has been run.

## 2026-07-18 04:10:00 UTC - Close the eos_suppress_weight branch (Pilots AF/AG)

Ran 4 training jobs in parallel on the single RTX 3090 (peak combined ~8.5GB VRAM, well under the 24GB card): eos_suppress_weight in {3, 8, 12} applied to all conditions, and weight=15 gated to `eos_suppress_conditions: [mid, low]` only. Best pooled mid/low relative improvement across all four was 2.83% (Pilot AF weight=3, u150) -- still below Pilot X's 3.86% and far short of the 5% gate. Gating to mid/low (Pilot AG) did not help either and raised catastrophic rate to 1.17%.

Decision: close the eos_suppress_weight branch entirely. Every weight tested (3, 8, 12, 15-gated, 25) underperforms Pilot X with weight=0; this is not a dose-response optimum to keep tuning, the auxiliary EOS-probability loss appears to conflict with the existing clean-anchor CE term rather than complement it. Pilot X u150 remains the best near-miss control for this branch. The next diagnostic should change the data/curriculum mix directly (e.g. raise the low-condition sampling proportion) rather than add another auxiliary loss term, and should not touch decoding. No full training, MWER/GRPO, router, or promotion evaluation has been run.

## 2026-07-18 05:00:00 UTC - Reject Pilot AH data-mix sweep; declare the recipe plateaued

Ran 3 parallel training jobs raising the late-stage low-condition curriculum share directly (0.20 -> 0.25/0.30/0.40, mid fixed, dry/high reduced to compensate), otherwise identical to Pilot X. Best result (low=0.30, u150) reached 3.73% pooled mid/low relative improvement -- still below Pilot X's 3.86%, and all three levels (3.02-3.73%) land within likely single-seed noise of X.

Decision: reject Pilot AH. Do not tune `condition_distribution` further on this recipe. Across 8+ pilots (V, W, X, Y, Z, AA-AH) covering LR, dropout, training length, CE weighting, adapter arithmetic, decode-time floors, auxiliary EOS-suppression loss, and now direct data-mix share, none has beaten Pilot X u150's 3.86% mid/low improvement against the 5% gate -- the recipe (encoder self-attn + decoder cross-attn LoRA, rank 16, on this data/augmentation pipeline) appears to have plateaued. Crossing the gate likely needs a structural change (different/larger LoRA target set, higher rank, or a different data source) rather than another hyperparameter tweak, and that should be scoped with the user before further GPU spend. Pilot X u150 remains the best near-miss control. No full training, MWER/GRPO, router, or promotion evaluation has been run.

## 2026-07-18 06:00:00 UTC - Reject Pilot AI train/eval augmentation-mismatch fixes; close the data-source review

Audited the training data source per the user's request. Found two real, quantified mismatches between Pilot X's training augmentation and the frozen `tuning_dev_proxy` eval set: (1) the low-condition SNR floor stops at 0dB in training vs -3dB in eval (28.7% of eval low samples are below 0dB, 40.7% below 1dB -- a regime never trained on); (2) Pilot X sets no `augmentation.effects` block at all, so high/mid clipping (eval: 2%/8%) and frame dropout (eval: ~10% of low samples) are never seen in training. Note Pilot J (an early pilot, on an older higher-LR encoder-attention-only recipe, before the Pilot D->W->X decoder-cross-attn + low-LR lineage) tried adding effects coverage and was rejected as "insufficient alone" -- but it was never combined with X's current recipe, and even Pilot J's SNR floor (1dB) didn't reach eval's -3dB.

Ran 3 parallel training jobs fixing these gaps on top of Pilot X's exact recipe: SNR floor only, effects only, and both combined. None crossed X: all three land at 2.70-2.96% pooled mid/low improvement (u150), below X's 3.86%. u100 checkpoints across all three reach a better catastrophic rate (0.67% vs X's 0.83%) but weaker mid/low gain (1.8-2.1%).

Decision: close the data-source review. The identified augmentation gaps are real and worth fixing for any eventual production training run (they represent genuine eval-fidelity defects), but they are not the lever that crosses the 5% mid/low gate at the current 150-update training budget -- augmentation-severity mismatch is ruled out as the primary blocker. Do not keep tuning augmentation severity. Pilot X u150 remains the best near-miss control (mid/low 3.86%). Any further pilot (longer training under corrected augmentation, larger LoRA rank/target set, or a different training objective) is a bigger structural step than the bounded single-variable pilots run so far and should be scoped with the user before more GPU spend. No full training, MWER/GRPO, router, or promotion evaluation has been run.
