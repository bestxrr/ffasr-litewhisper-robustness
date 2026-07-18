# Training

Phase 0 is active. Expensive SFT is gated until:

- actual model forward with labels returns finite loss,
- gradients reach selected LoRA modules,
- adapter save/reload equivalence passes,
- export path preserves adapters,
- baseline/export WER equivalence is understood.

Initial LoRA SFT defaults are in `configs/train/sft_lora.yaml`.

## Phase 0 Result

Run: `artifacts/runs/phase0_lite_whisper_trainability/phase0_report.json`

- Status: pass.
- Finite labeled loss: 8.578125 on a synthetic 1 s smoke example.
- LoRA targets found/attached: 56.
- Trainable parameters: 788,480.
- Frozen parameters: 594,411,520.
- Nonzero gradients: observed on adapter `lora_B` parameters. `lora_A`
  gradients are zero at initialization because `lora_B` starts at zero, which is
  expected for this LoRA initialization.
- Save/reload logit max absolute difference: 0.0.
- Merge/export logit max absolute difference: 0.03515625 with FP16 merge.
- Deterministic generation: pass.
- Peak VRAM: 2.21 GB.
- Export path: `artifacts/runs/phase0_lite_whisper_trainability/merged_export`.

Known issue found and fixed:

- Initial custom LoRA wrapper allocated adapter parameters on CPU after the model
  was moved to CUDA. The wrapper now initializes on the wrapped Linear layer's
  device and dtype.
- An FP16 AdamW smoke update caused NaN logits and exposed a missing NaN guard in
  the equivalence checker. The checker now rejects non-finite logits and uses a
  deterministic tiny adapter perturbation for export equivalence.

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
