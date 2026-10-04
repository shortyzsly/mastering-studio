from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Returns (data, sample_rate). data is shape (n_samples,) for mono or
    (n_samples, n_channels) for multi-channel, float32 in [-1, 1]."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return np.mean(data, axis=1).astype(np.float32)


def save_audio(path: str, data: np.ndarray, sr: int, subtype: str | None = None) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if subtype is None:
        subtype = "PCM_24" if path.lower().endswith((".wav", ".aiff", ".aif")) else None
    sf.write(path, data, sr, subtype=subtype)


def save_mp3(path: str, data: np.ndarray, sr: int, kbps: int = 192, out_sr: int = 44100) -> None:
    """CBR MP3 (ACX: 192 kbps, 44.1 kHz). Resamples if needed; TPDF-dithered to 16 bit.
    Streamed in ~30 s chunks so a long chapter never needs full-length float64 copies."""
    import lameenc
    from scipy.signal import resample_poly

    x = data if data.ndim == 2 else data[:, None]
    g = np.gcd(sr, out_sr)
    up, down = out_sr // g, sr // g
    chunk, margin = down * 8192, down * 16   # whole multiples of `down`: chunk edges map to exact output samples
    rng = np.random.default_rng(0)
    enc = lameenc.Encoder()
    enc.set_bit_rate(kbps)
    enc.set_in_sample_rate(out_sr)
    enc.set_channels(x.shape[1])
    enc.set_quality(2)   # 2 = high quality (slow) encoder search
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "wb") as f:
        for s0 in range(0, len(x), chunk):
            s1 = min(len(x), s0 + chunk)
            if up == down:
                y = x[s0:s1].astype(np.float32)
            else:   # resample with a margin on each side so the filter is settled at the chunk edges
                a, b = max(0, s0 - margin), min(len(x), s1 + margin)
                y = resample_poly(x[a:b], up, down, axis=0).astype(np.float32)
                lead = (s0 - a) * up // down
                y = y[lead : lead + -(-(s1 - s0) * up // down)]
            d = (rng.random(y.shape, dtype=np.float32) - rng.random(y.shape, dtype=np.float32)) / 32768.0
            pcm = np.clip(np.round((y + d) * 32767.0), -32768, 32767).astype("<i2")
            f.write(enc.encode(pcm.tobytes()))
        f.write(enc.flush())
