from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class EffectLog:
    name: str
    params: dict

    def to_dict(self) -> dict:
        return asdict(self)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, EffectLog]:
    if len(noise) < len(clean):
        reps = int(math.ceil(len(clean) / max(len(noise), 1)))
        noise = np.tile(noise, reps)
    noise = noise[: len(clean)]
    scale = rms(clean) / (rms(noise) * (10 ** (snr_db / 20.0)))
    out = clean + noise * scale
    return out.astype(np.float32), EffectLog("additive_noise", {"snr_db": float(snr_db), "noise_scale": float(scale)})


def convolve_rir(audio: np.ndarray, rir: np.ndarray) -> tuple[np.ndarray, EffectLog]:
    rir = rir.astype(np.float32)
    peak = float(np.max(np.abs(rir)) + 1e-12)
    rir = rir / peak
    out = signal.fftconvolve(audio, rir, mode="full")[: len(audio)]
    return out.astype(np.float32), EffectLog("rir", {"rir_peak": peak, "mode": "truncated_full"})


def butter_filter(
    audio: np.ndarray, sr: int, kind: str, cutoff: float | tuple[float, float], order: int = 5
) -> tuple[np.ndarray, EffectLog]:
    sos = signal.butter(order, cutoff, btype=kind, fs=sr, output="sos")
    out = signal.sosfiltfilt(sos, audio).astype(np.float32)
    return out, EffectLog("spectral_filter", {"kind": kind, "cutoff": cutoff, "order": order})


def soft_clip(audio: np.ndarray, drive_db: float) -> tuple[np.ndarray, EffectLog]:
    gain = 10 ** (drive_db / 20.0)
    out = np.tanh(audio * gain) / np.tanh(gain)
    return out.astype(np.float32), EffectLog("soft_clip", {"drive_db": float(drive_db)})


def frame_dropout(audio: np.ndarray, sr: int, start_s: float, duration_ms: float) -> tuple[np.ndarray, EffectLog]:
    out = audio.copy()
    start = max(0, min(len(out), int(start_s * sr)))
    n = max(1, int(duration_ms * sr / 1000.0))
    out[start : min(len(out), start + n)] = 0.0
    return out.astype(np.float32), EffectLog("frame_dropout", {"start_s": float(start_s), "duration_ms": float(duration_ms)})
