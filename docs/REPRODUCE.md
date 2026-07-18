# Reproducing this project on another server

This project is a bounded, configuration-driven LoRA fine-tuning workflow for
`efficient-speech/lite-whisper-large-v3-turbo-acc` on the FFASR robustness task.
Everything is config-driven under `configs/`, `scripts/` are thin wrappers, and
`src/` holds the implementation. All results, decisions, and the full pilot-by-pilot
history are in `docs/RESULTS.md`, `docs/EXPERIMENTS.md`, `docs/DECISIONS.md`,
`docs/ERROR_ANALYSIS.md`, and `artifacts/runs/experiment_registry.jsonl`.

The current best result is **Pilot X u150** (see `docs/RESULTS.md`). It does not
yet clear the promotion gate (needs >=5% relative mid/low WER improvement; Pilot X
reaches 3.86%). Its adapter weights and eval report are published on Hugging Face —
see the "Get the best result without retraining" section below.

## 1. Environment

- Python 3.12, a CUDA GPU (everything here was run on a single RTX 3090, 24GB VRAM,
  peak usage per training/eval job is under 2GB, so any recent NVIDIA GPU with
  >=8GB works).
- `torch`/`transformers`/`peft` stack; see `requirements.txt`.

```bash
git clone https://github.com/bestxrr/ffasr-litewhisper-robustness.git
cd ffasr-litewhisper-robustness
python3 -m venv .venv && source .venv/bin/activate   # or use an existing venv/conda env
bash scripts/setup.sh          # pip installs requirements.txt into the active env
bash scripts/check_environment.sh   # sanity-check python/torch/CUDA/package versions
```

`scripts/setup.sh` calls `uv pip install -r requirements.txt` — install `uv`
first (`pip install uv`) or edit the script to use plain `pip` if you don't have it.

Set credentials as environment variables (or in a local `.env`, which is
gitignored) before running anything that touches the Hub:

```bash
export HF_TOKEN=...        # read access is enough to download the base model/data
export WANDB_API_KEY=...   # optional, only needed if you enable wandb logging
```

## 2. External tool dependencies

`external/` is gitignored (each is its own repo, vendored locally, not part of
this repo's history). Fetch them at the exact commits used during this project if
you need them (the core pilots in `configs/train/` and `configs/eval/` do not
require them — only clone these if a script in `scripts/` references them):

```bash
mkdir -p external && cd external
git clone https://github.com/efeslab/LiteASR && git -C LiteASR checkout da5be85382a6fc22245803c18c820b5c1b049ef8
git clone https://github.com/xzf-thu/Mega-ASR && git -C Mega-ASR checkout bd75877b29b695d852234169fd3738bb5080046b
git clone https://huggingface.co/spaces/treble-technologies/ffasr && git -C ffasr checkout 99160fddc3ccefa946fd247b9f116dfcccf85b9b
cd ..
```

## 3. Rebuild the data manifests

All audio is derived from public LibriSpeech Parquet shards on the Hugging Face
Hub (`openslr/librispeech_asr`), downloaded on demand — no raw audio is committed
to this repo (`artifacts/manifests/` is gitignored, ~2.1GB when materialized).

```bash
# training pool: 2400 utterances from train-clean-100 shards 0000-0001 (22 speakers, 53 chapters)
bash scripts/download_bounded_data.sh configs/data/librispeech_train_clean_diag.yaml

# tuning-dev source pool: 900 utterances from test-clean shard 0000 (speaker/utterance-disjoint from train)
bash scripts/materialize_parquet_audio.sh configs/data/librispeech_tuning_dev_clean_parquet.yaml

# build the frozen 600-sample (150/condition) proxy eval set used by every pilot below
bash scripts/build_proxy_manifests.sh configs/eval/tuning_dev_proxy.yaml
```

This regenerates `artifacts/manifests/librispeech_train_clean_diag/manifest.jsonl`,
`artifacts/manifests/librispeech_tuning_dev_clean/manifest.jsonl`, and
`artifacts/manifests/proxy/tuning_dev_proxy.jsonl` (the exact eval set referenced
throughout `docs/RESULTS.md`). The proxy build is seeded (`seed: 20260718` in
`configs/eval/tuning_dev_proxy.yaml`), so condition assignment (dry/high/mid/low),
SNR, RIR, and effect parameters per sample are deterministic and reproducible.

## 4. Reproduce the no-repeat-3 control baseline

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/baseline_tuning_dev_decode_ng3.yaml
```

Expect `avg_wer ~= 14.39`, `catastrophic_output_rate ~= 0.83%` (see
`docs/RESULTS.md` for the full per-condition breakdown). Every pilot's
`avg_relative_improvement_pct` / `midlow_relative_improvement_pct` is computed
against this run.

## 5. Reproduce the best result (Pilot X)

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/pilot_x_decoder_cross_attn_lr5e6_150.yaml
# writes artifacts/runs/pilot_x_decoder_cross_attn_lr5e6_150/{adapter.safetensors, adapter_update_{75,100}.safetensors, train_log.csv, train_report.json}

CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/pilot_x_u150_decode_ng3_tuning_dev.yaml
# writes artifacts/reports/pilot_x_u150_decode_ng3_tuning_dev_proxy/{summary.json, errors.csv, condition_summary.csv}
```

Training is ~430s / 150 updates on a single RTX 3090 (peak ~1.5GB VRAM allocated),
seeded (`seed: 1337`), so re-running should reproduce the numbers in
`docs/RESULTS.md` (Pilot X u150: avg 14.00, dry 2.11, high 7.15, mid 13.76, low
32.97, catastrophic 0.83%, avg improvement 2.70%, mid/low improvement 3.86%) up to
minor nondeterminism from cuDNN/AMP kernels.

## 6. Reproduce any other pilot

Every pilot referenced in `docs/RESULTS.md` / `docs/EXPERIMENTS.md` /
`docs/DECISIONS.md` / `artifacts/runs/experiment_registry.jsonl` has a matching
config pair:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sft.sh configs/train/<pilot>.yaml
CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate_proxy.sh configs/eval/<pilot>_<checkpoint>_decode_ng3_tuning_dev.yaml
```

Some pilots are evaluation-only (adapter arithmetic, decode-time sweeps like
Pilot AD) and skip the train step; check `changed_variable` / `command` in the
matching `artifacts/runs/experiment_registry.jsonl` entry for the exact command
that was run. Training runs are cheap enough (under 2GB VRAM, 7-30 minutes each
depending on GPU contention) to run several in parallel on one GPU — see the
Pilot AF/AG/AH/AI entries in `docs/EXPERIMENTS.md` for the pattern (launch with
`nohup ... &`, one process per config, same `CUDA_VISIBLE_DEVICES`).

## 7. Get the best result without retraining

The Pilot X u150 adapter (best result so far) and the full docs/registry snapshot
are published on Hugging Face at
`banhchungtuongot/ffasr-litewhisper-robustness` (private — request access from the
repo owner). To load it directly:

```python
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

adapter_path = hf_hub_download(
    repo_id="banhchungtuongot/ffasr-litewhisper-robustness",
    filename="adapters/pilot_x_u150/adapter.safetensors",
    token="<your HF token with access to this repo>",
)
state_dict = load_file(adapter_path)
```

Then attach it the same way `src/evaluation/proxy_eval.py:maybe_load_adapter`
does: LoRA rank 16 / alpha 32 / dropout 0.0 at inference, targeting
`model.encoder.layers.*.self_attn.{q,v,out}_proj` and
`model.decoder.layers.*.encoder_attn.{q,v,out}_proj` on
`efficient-speech/lite-whisper-large-v3-turbo-acc` (revision
`ef2c0dd768cc9832a8a5a3397ab7218c838fea66`), with `openai/whisper-large-v3` as the
processor. The exact recipe and LoRA config are in
`configs/train/pilot_x_decoder_cross_attn_lr5e6_150.yaml` and
`configs/eval/pilot_x_u150_decode_ng3_tuning_dev.yaml`.

## 8. Run the regression tests

```bash
python -m pytest tests/ -q
```

## 9. Next step: scaling the data (H100 run)

The single documented next step is scaling the training data — see
**`docs/DATA_PLAN.md`** for the full plan (grounded in the leaderboard rank-1
model's data recipe). Short version: the recipe has plateaued because the 2400-
utterance / 22-speaker pool is too small (adding capacity overfits, per Pilot AK),
so the highest-leverage move is to retrain Pilot X's exact recipe on much more
LibriSpeech (train-clean-360 / full 960h) on a box without this instance's
48GB-disk / 24GB-VRAM caps, then add difficulty filtering and real RIR/noise.

## 10. Current status / where to pick up

Read `docs/DECISIONS.md` bottom-to-top for the most recent closing rationale.
As of the last entry: **11+ single-variable pilots** (LR, dropout, training
length, CE weighting, adapter arithmetic, decode-time floors, auxiliary
EOS-suppression loss, direct data-mix reweighting, and a full train/eval
augmentation-mismatch audit+fix) have all failed to beat Pilot X u150's 3.86%
mid/low improvement against the 5% promotion gate. The recipe (encoder self-attn +
decoder cross-attn LoRA, rank 16, 150 updates) appears plateaued. The two
un-tried, larger-scope directions noted for a future session are (a) longer
training under the corrected augmentation from Pilot AI, and (b) a larger/different
LoRA target set or rank. Neither should be started without confirming scope, since
both are bigger steps than the bounded single-variable pilots this history is built
from.
