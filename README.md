# FFASR LiteWhisper Robustness Project

This repository contains a bounded, configuration-driven workflow for improving
`efficient-speech/lite-whisper-large-v3-turbo-acc` on FFASR without using hidden
evaluation transcripts.

## Status

Not promotable yet. Best result is **Pilot X u150** (LoRA rank 16 on encoder
self-attention + decoder cross-attention, LR 5e-6, 150 updates): +2.70% average
WER, +3.86% combined mid/low WER, catastrophic output rate flat at 0.83%, against
a required >=5% mid/low improvement to clear the promotion gate. Adapter weights
are published on Hugging Face (see `docs/REPRODUCE.md` section 7).

11+ single-variable pilots on top of Pilot X — learning rate, dropout, training
length, loss-weighting by condition, adapter arithmetic, decode-time floors, an
auxiliary EOS-suppression loss, direct data-mix reweighting, and a full
train/eval augmentation-mismatch audit+fix — have all failed to beat it. See
`docs/DECISIONS.md` (chronological, most recent at the bottom) for the full
rationale and what's left to try.

| doc | contents |
| --- | --- |
| `docs/RESULTS.md` | per-pilot metrics tables against the frozen `tuning_dev_proxy` eval set |
| `docs/EXPERIMENTS.md` | what was run and how, with commands |
| `docs/DECISIONS.md` | promotion/reject decisions and reasoning, chronological |
| `docs/ERROR_ANALYSIS.md` | paired substitution/deletion/insertion analysis per pilot |
| `docs/REPRODUCE.md` | how to reproduce any of the above on a fresh server |
| `artifacts/runs/experiment_registry.jsonl` | machine-readable log of every pilot (one JSON object per line) |

## Quick start

```bash
bash scripts/check_environment.sh
bash scripts/fetch_ffasr_leaderboard.sh
bash scripts/phase0_trainability.sh configs/experiments/phase0.yaml
```

For a full environment setup, data rebuild, and step-by-step reproduction of the
best result (or any other pilot) on a different machine, see
**`docs/REPRODUCE.md`**.

Project caches are local to `.cache/` and run artifacts are under `artifacts/`
(both gitignored — see `docs/REPRODUCE.md` to regenerate). Large downloads and
training scripts call `scripts/check_disk.sh` before running.
