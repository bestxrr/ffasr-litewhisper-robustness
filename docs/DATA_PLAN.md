# Next step: scaling the training data (for the H100 run)

**Status as of 2026-07-18.** Every recipe-side lever has plateaued at ~3.86% mid/low
relative improvement (Pilot X u150), short of the 5% promotion gate. The decisive
evidence that the ceiling is **training-data scale, not the recipe**, came from
Pilot AJ/AK/AJK (see `docs/DECISIONS.md`, `docs/RESULTS.md`): adding encoder-FFN
capacity (AK) **overfit catastrophically** on the 2400-utterance pool (avg WER
17.9-18.6, dry +2.7), and a proper clean-target distillation signal (AJ) was
**neutral**. On a 2400-utterance / 22-speaker / 8.6h pool, more capacity just
memorizes and a better signal has nothing new to teach.

This document is the actionable data plan to run on the H100 server. It is grounded
in (a) our own error analysis and (b) how the current leaderboard **rank-1 model,
Mega-ASR, actually built its data** (repo vendored at `external/Mega-ASR`).

## What rank-1 (Mega-ASR) did with data

- **2.4M training samples** ("Voices-in-the-Wild-2M" on the Hub) — ~1000x our 2400.
- **7 atomic acoustic effects**: reverberation, echo, additive noise, far-field,
  frequency dropout, bandwidth limitation, clipping distortion — combined into
  **54 compound scenarios**.
- Still **simulated** (spectral-manipulation pipeline) but with an "agentic check
  for physical plausibility".
- **Difficulty filtering**: they **drop samples above 70% WER** (by a reference
  model), keeping training "hard but learnable".
- Note it is a **full foundation model** (Qwen3-ASR base + A2S-SFT + DG-WGPO RL),
  a different scale of effort from our LoRA-on-LiteWhisper. Realistic target for us
  is clearing the mid/low gate and landing near top-7 (avg WER <= 17.34), not
  rank-1's 13.38.

## Gaps in our current data/simulator vs the benchmark

The FFASR benchmark scores: Near Field, Lab Measured, Lab Simulated, High/Mid/Low
SNR, and **Moving Sources (Low/Mid/High SNR)**. Our current pipeline
(`configs/data/librispeech_train_clean_diag.yaml` + `src/training/sft.py`
`augment_audio` + `src/evaluation/proxy_eval.py`):

| axis | rank-1 | ours now | benchmark implication |
| --- | --- | --- | --- |
| speech source | 2.4M, very diverse | 2400 utts / **22 speakers** | AK proved this overfits |
| reverb | real RIRs | **procedural** exponential-decay RIR | benchmark has "Lab Measured" (real rooms) |
| noise | real | synthetic **colored** noise | benchmark uses real noise |
| moving source | yes | **none** | benchmark has a whole "Moving Sources" category we never simulate |
| echo (distinct from reverb) | yes | no | missing 1 of the 7 atoms |
| difficulty filtering | drop >70% WER | **none** | our low-<0dB samples hit 48% WER, partly unlearnable |

## The plan — 4 tiers, priority order

### Tier 1 (highest leverage, near-free, do first) — scale the clean speech source
The binding constraint is 22 speakers. Move from 2 parquet shards to more of
LibriSpeech (same CC BY 4.0 license, already the source):

- **train-clean-100**: ~100h, **251 speakers**, ~6GB — 11x more speakers.
- **train-clean-360**: ~360h, **921 speakers**, ~23GB.
- **train-other-500**: ~500h, harder/noisier read speech, ~30GB — good for robustness.
- Full corpus: ~960h / ~2338 speakers.

Change: extend `files:` and raise `max_items` in
`configs/data/librispeech_train_clean_diag.yaml` (or make a new
`configs/data/librispeech_train_clean_big.yaml`), rebuild via
`scripts/download_bounded_data.sh`, then retrain Pilot X's exact recipe on the
larger pool. **On H100 there is no 48GB disk / 24GB VRAM cap** (our box had both),
so go straight to train-clean-360 or the full 960h.

### Tier 2 (realism — what the benchmark actually measures) — real RIRs + real noise
Replace synthetic reverb/noise with real recordings:

- **Real RIRs**: OpenSLR SLR28 (RIRS_NOISES, ~1GB) — matches "Lab Measured/Simulated".
- **Real noise**: MUSAN (OpenSLR SLR17, ~11GB) — mix real ambient/babble/music.
- Optional real far-field with clean transcripts: **VOiCES** (LibriSpeech re-recorded
  in real rooms) — directly labeled far-field data.

Code: teach `src/augmentation` / `augment_audio` to convolve with a sampled real RIR
file and mix a sampled real noise clip, instead of `procedural_rir` /
`colored_noise`. Keep the SNR/condition taxonomy.

### Tier 3 (code, no disk) — add the missing atoms + compound taxonomy
- **Echo**: discrete delayed+attenuated copies (distinct from diffuse reverb).
- **Moving source**: time-varying RIR (interpolate between RIRs along a trajectory)
  — this is the benchmark's "Moving Sources" category, currently unsimulated.
- Restructure augmentation from a fixed pipeline into **sampled combinations of
  atoms** (the 7→54 idea), so training covers compound scenarios systematically.

### Tier 4 (cheap quality fix, rank-1 calls it critical) — difficulty filtering
Run the base model over each simulated sample and **drop those above ~70% WER**.
Our error analysis showed low-<0dB samples (WER 48%, zero-sum substitution<->deletion)
are partly unlearnable and inject hallucination/noise into training. Keeps the
training distribution "hard but learnable".

## Recommended order on H100

1. **Tier 1** first (retrain Pilot X on train-clean-360 / full LibriSpeech). If just
   scaling speakers moves mid/low, the data-limited diagnosis is confirmed before
   investing in Tier 2-4 code.
2. Then **Tier 4** (difficulty filtering) — cheap, rank-1 emphasizes it.
3. Then **Tier 2** (real RIR/noise) and **Tier 3** (echo, moving source, compound
   taxonomy) — more disk + code.

## Re-open what the small-data regime forced us to reject

Several reject decisions were conditioned on the tiny 2400-sample pool and the
45-min / 150-update budget. **With big data on H100 these should be re-tested**, as
their failure mode was overfitting a tiny set, which more data directly fixes:

- **Encoder-FFN LoRA (Pilot AK)** and **higher LoRA rank (Pilot S)** — rejected only
  because they overfit 2400 samples; with 100x data the extra capacity may now help.
- **Longer training / more updates (Pilot Z, u150 cap)** — the u150 ceiling was an
  overfitting ceiling on small data, not a fundamental one.
- **Feature distillation weight sweep (Pilot AJ)** — neutral at w=1.0 on small data;
  worth a weight sweep once the encoder actually has diverse data to align.

See `configs/train/pilot_x_decoder_cross_attn_lr5e6_150.yaml` for the recipe to scale
up, and `docs/REPRODUCE.md` for environment setup on a fresh server.
