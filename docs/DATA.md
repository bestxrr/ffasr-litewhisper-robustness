# Data

No large training corpora have been downloaded yet.

Smoke manifest:

- Config: `configs/data/librispeech_smoke.yaml`
- Dataset: `hf-internal-testing/librispeech_asr_dummy`, Parquet-converted tiny
  LibriSpeech fixture.
- Output: `artifacts/manifests/librispeech_smoke/manifest.jsonl`
- Items: 4
- Purpose: validate manifest/audio materialization without a large corpus pull.

Fixed proxy manifests:

- Clean source: `artifacts/manifests/librispeech_proxy_clean/manifest.jsonl`
  with 2,200 LibriSpeech validation-clean utterances materialized from direct
  Parquet shard `openslr/librispeech_asr:clean/validation/0000.parquet`.
- Full proxy: `artifacts/manifests/proxy/full_proxy.jsonl`, 2,000 rows.
- Mini proxy: `artifacts/manifests/proxy/mini_proxy.jsonl`, 400 rows.
- Condition balance: dry/high/mid/low = 500/500/500/500 full and
  100/100/100/100 mini.
- Proxy config hash: recorded in `artifacts/manifests/proxy/proxy_summary.json`.

Planned bounded sources:

- Clean speech: bounded LibriSpeech train-clean-100 subset, speaker-disjoint.
- RIR: Treble10-RIR if accessible, plus bounded measured/procedural RIRs.
- Noise: bounded MUSAN/DNS/OpenSLR-style subsets.

Rules:

- Store a single clean master copy.
- Keep generated degradations on the fly.
- Fixed proxy manifests must be versioned and never regenerated silently.
- Proxy splits must be disjoint by speaker, utterance, noise file, and RIR/room.

Implementation notes:

- `datasets>=5` removed support for legacy dataset scripts. The project pins
  `datasets>=3.6,<4` so established LibriSpeech-style loaders can still be used.
- Some current HF audio datasets expose `torchcodec.AudioDecoder`; the downloader
  avoids this path when possible by using `Audio(decode=False)` and reading
  embedded bytes or file paths with `soundfile`.
