# Baseline And Official Protocol

Retrieval date: 2026-07-17 UTC.

Sources:

- FFASR Space: `https://huggingface.co/spaces/treble-technologies/ffasr`
- Space commit inspected locally: `99160fddc3ccefa946fd247b9f116dfcccf85b9b`
- Leaderboard retrieved through the public Gradio `_on_startup` API.

Official default ranking protocol observed in the Space:

- Metric: word error rate.
- Default average: mean WER percentage over Near Field Speech, High SNR, Mid SNR,
  and Low SNR.
- Text normalization: Whisper English normalization for references and
  predictions.
- Hidden evaluation data: private packed tensors in the FFASR storage bucket.
- Additional reported columns include Lab Measured, Lab Simulated, moving-source
  splits, RTFx, and parameter count, but they are not part of the default average.

Current thresholds from live leaderboard:

- Top-5 threshold: Avg WER <= 16.35, rank 5
  `ibm-granite/granite-speech-4.1-2b`.
- Top-7 threshold: Avg WER <= 17.34, rank 7
  `nvidia/parakeet-tdt-1.1b`.

Starting model public row:

| Model | Rank | Avg | Dry | High | Mid | Low | RTFx | Params B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| efficient-speech/lite-whisper-large-v3-turbo-acc | 19 | 26.04 | 4.33 | 13.33 | 29.48 | 57.03 | 92.5779 | 0.594 |

Reproduction status:

- Public leaderboard row was reproduced by querying the official Space UI API.
- Full official reevaluation requires submitting/running a model through the FFASR
  Space or maintainer Hub Job path. That is an external action and will not be
  launched without authorization.
