# Goal

Starting model: `efficient-speech/lite-whisper-large-v3-turbo-acc` at revision
`ef2c0dd768cc9832a8a5a3397ab7218c838fea66`.

Objective: improve FFASR robustness toward top-7 performance without training on
hidden FFASR evaluation transcripts or replacing the base model with a stronger
architecture.

Current hard constraints:

- GPU: one RTX 3090, 24 GB VRAM.
- Peak allocated VRAM target: below 23 GB.
- Project disk budget: below 48 GB.
- Per-run wall clock: 1.5 to 3 hours.
- Main adaptation method: LoRA or similarly parameter-efficient training.

Decision gates are implemented as scripts and reports under `artifacts/runs/`.
Phase 0 must pass before any expensive training.
