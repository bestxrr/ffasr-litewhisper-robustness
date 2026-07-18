import numpy as np

from src.augmentation.effects import frame_dropout, mix_at_snr, rms, soft_clip


def test_mix_at_snr_shape_and_log():
    clean = np.ones(1600, dtype=np.float32) * 0.1
    noise = np.random.default_rng(0).normal(size=800).astype(np.float32)
    mixed, log = mix_at_snr(clean, noise, 10.0)
    assert mixed.shape == clean.shape
    assert log.name == "additive_noise"
    assert rms(mixed) > 0


def test_soft_clip_and_dropout_are_finite():
    audio = np.linspace(-1, 1, 1600, dtype=np.float32)
    clipped, _ = soft_clip(audio, 4.0)
    dropped, _ = frame_dropout(clipped, 16000, 0.01, 20)
    assert np.isfinite(dropped).all()
    assert (dropped == 0).any()
