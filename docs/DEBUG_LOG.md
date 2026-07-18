# Debug Log

## 2026-07-17 setup

Symptom: system `python` is not on PATH in non-login shell before activating
`/venv/main`.

Root cause: Vast image provides Python through the default venv.

Fix: all scripts source `/venv/main/bin/activate`.

Regression test: `bash scripts/check_environment.sh`.

## 2026-07-17 Phase-0 adapter device mismatch

Symptom: labeled forward failed with `mat2 is on cpu, different from other
tensors on cuda:0` after LoRA insertion.

Root cause: custom LoRA adapter parameters were initialized on CPU after the base
model had been moved to CUDA.

Fix: initialize `lora_A` and `lora_B` on `base.weight.device` and
`base.weight.dtype`.

Regression test: `bash scripts/phase0_trainability.sh configs/experiments/phase0.yaml`.

## 2026-07-17 LibriSpeech loader compatibility

Symptom: `datasets==5.0.0` rejected legacy LibriSpeech dataset scripts, and
streaming `torchcodec.AudioDecoder` examples caused aborts on shutdown.

Root cause: upstream `datasets` removed script loading in v4+, while the
OpenSLR LibriSpeech loader and older dummy dataset still rely on script paths.

Fix: pin `datasets>=3.6,<4`; use the Parquet-converted
`hf-internal-testing/librispeech_asr_dummy` for smoke tests; prefer
`Audio(decode=False)` and embedded bytes in the downloader.

Regression test: `bash scripts/download_bounded_data.sh configs/data/librispeech_smoke.yaml`.

## 2026-07-17 OpenSLR split preparation disk blow-up

Symptom: `load_dataset("openslr/librispeech_asr", split="validation")` attempted
to prepare all clean LibriSpeech splits, including `train.360`, and hit
`No space left on device`.

Root cause: the dataset builder prepares more than the requested validation split.

Fix: deleted only the explicit failed cache path
`.cache/huggingface/hub/datasets--openslr--librispeech_asr` and replaced proxy
materialization with direct download of `clean/validation/0000.parquet`.

Regression test: `bash scripts/materialize_parquet_audio.sh configs/data/librispeech_proxy_clean_parquet.yaml`.

## 2026-07-17 Phase-0 NaN equivalence guard

Symptom: FP16 AdamW smoke update produced NaN logits, but the first equivalence
guard did not fail because `nan > threshold` is false.

Root cause: comparison code did not explicitly check `torch.isfinite`.

Fix: save/reload and merge equivalence checks now reject non-finite logits; the
export equivalence perturbation is deterministic and finite.

Regression test: `bash scripts/phase0_trainability.sh configs/experiments/phase0.yaml`.

## phase0_lite_whisper_trainability failure

Symptom: `RuntimeError('Expected all tensors to be on the same device, but got mat2 is on cpu, different from other tensors on cuda:0 (when checking argument in method wrapper_CUDA_mm)')`

Root cause: pending investigation.

Fix: not applied yet.

Regression test: Phase-0 trainability script.

## phase0_lite_whisper_trainability failure

Symptom: `RuntimeError('Merged export logits mismatch: max_abs=0.04296875')`

Root cause: pending investigation.

Fix: not applied yet.

Regression test: Phase-0 trainability script.

## phase0_lite_whisper_trainability failure

Symptom: `RuntimeError('Merged export logits mismatch: max_abs=0.05859375')`

Root cause: pending investigation.

Fix: not applied yet.

Regression test: Phase-0 trainability script.

## 2026-07-17 04:43:41 UTC - SFT mixed-precision GradScaler failure

Symptom: the first SFT smoke failed at `GradScaler.unscale_` with `ValueError: Attempting to unscale FP16 gradients`.

Root cause: custom LoRA adapter parameters inherited the FP16 dtype of the frozen base model, so trainable gradients were FP16.

Fix: keep `LoRALinear` adapter weights in FP32, compute the adapter branch in FP32, and cast the delta back to the base output dtype.

Regression test: `python -m pytest tests -q` passed; `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_a_smoke2.yaml` completed 2 updates with finite gradients and saved an adapter.

Affected experiments: only failed pre-pilot smoke; no retained pilot metrics were produced before the fix.

## 2026-07-17 04:49:53 UTC - AMP loss-scale overflow during Pilot A

Symptom: Pilot A failed reproducibly at update 6 with all adapter gradients NaN when GradScaler used the default initial scale.

Root cause: AMP loss-scale overflow in the adapter backward pass; a 10-update diagnostic showed the same failure point, and reducing initial AMP scale to 128 eliminated it.

Fix: trainer now uses configurable `amp.init_scale` with default 128 and logs `amp_scale` in `train_log.csv`.

Regression test: `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_a_diag10.yaml` and then `configs/train/pilot_a_l0.yaml` completed with finite gradients.

## 2026-07-17 05:26:17 UTC - Label and teacher-forcing audit

- Tokenizer: `openai/whisper-large-v3`.
- BOS/EOS/PAD IDs: 50257/50257/50257.
- Tokenizer default prefix tokens: [50258, 50259, 50360, 50364].
- English transcribe prompt IDs: [(1, 50259), (2, 50360), (3, 50364)].
- Example text: `HELLO WORLD`.
- Encoded IDs: `[50258, 50364, 39, 8763, 46, 30029, 23704, 50257]`.
- Decoded with specials: `<|startoftranscript|><|notimestamps|>HELLO WORLD<|endoftext|>`.
- Decoded without specials: `HELLO WORLD`.
- Trainer fix: label creation now masks tokenizer padding positions to `-100` before teacher forcing. With micro-batch 1 there is normally no padding, but the path is now correct for future batching.
- Model audit: LiteWhisper config uses `decoder_start_token_id=50258`; generation config has no forced decoder IDs, so explicit generation language/task kwargs remain the eval control.


## 2026-07-17 05:27:13 UTC - Label prefix blocker diagnosed

Symptom: prior SFT pilots degraded dry/high WER quickly or failed to improve degraded speech at lower LR.

Root cause found during teacher-forcing audit: `processor.tokenizer(text=...)` can omit `<|en|><|transcribe|>` depending on tokenizer prefix state, producing labels with only `<|startoftranscript|><|notimestamps|>...<|endoftext|>`. Evaluation decoding attempts English/transcribe prompts, so training and decoding were not reliably aligned.

Fix: `src.training.sft.make_labels` now explicitly calls `set_prefix_tokens(language='en', task='transcribe', predict_timestamps=False)` before tokenization and masks padding to `-100`. Regression test `tests/test_labels.py` verifies the first four label IDs are `[50258, 50259, 50360, 50364]`.

Affected experiments: all previous SFT pilots are retained as rejected pre-fix runs and must not be used for promotion decisions except as evidence of the label-path bug.

## 2026-07-17 08:50:00 UTC - Robust SFT WER divergence after trainability fix

Symptom: after the label-prefix fix, clean-only sanity reduces clean CE and preserves dry WER, but robust SFT pilots still fail promotion. Pilot D gives only a small average/low gain at 100 updates; Pilot H continuation to 150/300 updates worsens dry/high and then all conditions.

Root cause: not a basic trainability failure. The likely blocker is objective and simulator/condition mismatch: the current degraded CE signal encourages some low-condition deletion/insertion reductions, but also increases substitutions and clean/high errors. Adding encoder `fc2` LoRA worsens this, so simply adding adapter capacity is not sufficient.

Fix: no final fix yet. Rejected Pilot H for promotion and blocked full training/MWER/router/submission. Next experiments should remain short controlled acoustic SFT pilots that change one variable at a time, such as lower LR/earlier checkpoint selection, simulator severity balance, or a narrower target/module schedule.

Regression test: keep using mini-proxy paired candidate-vs-baseline analysis plus clean CE/dry sanity before any full-proxy or promoted run.

## 2026-07-17 09:10:00 UTC - Lower-LR robust SFT still below promotion gate

Symptom: Pilot I (`lr=5e-6`) avoids Pilot H's severe dry/high regression but still fails to beat baseline meaningfully. Best checkpoint u75 gives avg 20.22 vs baseline 20.45 and mid/low 35.83 vs 36.10, with catastrophic output rate 3.75% vs 3.50%.

Root cause: lower LR reduces overfit but does not solve acoustic mismatch. Paired analysis shows only 18 improved and 8 regressed utterances at u75, with most samples unchanged. Error movement is small and dominated by deletion/insertion reductions offset by extra substitutions.

Fix: no final fix yet. Rejected lower-LR continuation and retained artifacts for comparison. Next short pilot should change simulator severity/condition targeting or adapter target locality rather than LR/update count alone.

Regression test: any new SFT pilot must be compared against baseline and Pilot D/Pilot I using the fixed mini-proxy and paired regression artifacts.

## 2026-07-17 09:35:00 UTC - Baseline tuning-dev summary crash

Symptom: baseline tuning-dev evaluation completed decoding but crashed while writing the summary because `baseline_dry_wer: null` was cast to `float`.

Root cause: `src.evaluation.proxy_eval.summarize` assumed a numeric baseline dry WER even for baseline runs.

Fix: allow `baseline_dry_wer` to be `None` and use the run's dry WER as the dry-regression reference in that case. Added `src.evaluation.summarize_errors` to regenerate summaries from an existing `errors.csv` without rerunning decoding.

Regression test: `python -m src.evaluation.summarize_errors configs/eval/baseline_tuning_dev.yaml` rebuilt `artifacts/reports/baseline_tuning_dev_proxy/summary.json`.

## 2026-07-17 10:24:00 UTC - Simulator metadata limitation

Symptom: simulator audit can show low-condition `frame_dropout` in detailed `effect_logs`, but proxy manifest `effects` summaries do not list it as a top-level effect.

Root cause: the proxy manifest records top-level sampled effect labels incompletely for frame-level dropout; detailed parameters remain in `effect_logs`.

Fix: no manifest rewrite applied to preserve frozen proxy versions. Analysis tools should inspect both `effects` and `effect_logs` when auditing simulator coverage.

Regression test: `python -m src.analysis.simulator_audit ...` reports both top-level effect counts and effect-log evidence.

## 2026-07-17 11:06:00 UTC - Low-condition runaway insertion blocker

Symptom: Pilots J/N/O reduce some mid and low substitutions but fail promotion because low-condition WER worsens or improves too little. Aggregate low insertion counts are sensitive to a few repeated-output samples.

Root cause: not isolated to frame dropout or clipping. Pilot N removed dropout and softened severity, and Pilot O removed clipping too, but low insertions remained elevated. The worst samples show repeated phrases such as "to be a servant" and "state of the state", indicating a decoding/objective length-bias problem under low-condition acoustics.

Fix: no final fix yet. Next diagnostic should add configurable decoder repetition controls or an objective-side repetition/length regularizer and evaluate both baseline and candidate with identical settings. MWER remains out of scope until the ordinary SFT/decoder diagnostic shows a stable sequence-level problem and a same-checkpoint continued-SFT control is defined.

Regression test: paired proxy analysis by `proxy_id` plus low-condition hypothesis/reference length diagnostics must accompany any future decoder or sequence-level change.

## 2026-07-17 11:50:00 UTC - FP16 gradients when unfreezing LayerNorm

Symptom: first Pilot R launch failed before update 1 with `ValueError: Attempting to unscale FP16 gradients.`

Root cause: regex-unfrozen LayerNorm parameters were converted to FP16 with the base model, unlike LoRA adapter weights which are initialized in FP32. `torch.amp.GradScaler` refuses to unscale FP16 gradients.

Fix: `src.training.sft` now casts parameters matched by `trainable_parameter_regex` to FP32 before setting `requires_grad=True`. `src.models.lora.save_lora` now saves all explicitly trainable parameters, not only `.lora_A` and `.lora_B`.

Regression test: `python -m pytest tests/test_metrics.py tests/test_labels.py -q` passed, and `CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_r_attn_lora_ln_l0.yaml` completed 100 updates with finite gradients.
