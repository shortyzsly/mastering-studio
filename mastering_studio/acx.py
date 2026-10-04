"""ACX (Audible) submission requirements: measure a finished file against them,
and the one fix that is safe to make after everything else -- a final gain
trim (through the true-peak limiter) when RMS lands outside the window.

ACX audio submission requirements (per uploaded chapter file):
  * RMS -23 to -18 dB, peaks no higher than -3 dB, noise floor no higher than -60 dB RMS
  * 0.5-1 s of room tone at the head, 1-5 s at the tail
  * 192 kbps+ CBR MP3, 44.1 kHz, all files mono (or all stereo)
"""
from __future__ import annotations

import numpy as np

from .dsp import activity as act_mod, limiter

RMS_RANGE = (-23.0, -18.0)
PEAK_MAX_DB = -3.0
NOISE_MAX_DB = -60.0
HEAD_RANGE_S = (0.5, 1.0)
TAIL_RANGE_S = (1.0, 5.0)
MP3_SR = 44100
MP3_KBPS = 192

RMS_SAFE = (RMS_RANGE[0] + 0.5, RMS_RANGE[1] - 0.5)   # land inside with margin
RMS_TRIM_TARGET = -20.5
EDGE_MARGIN_S = 0.05   # soft word onsets/decays sit just outside the speech detector's frames


def speech_bounds(mono: np.ndarray, sr: int) -> tuple[int, int] | None:
    """(first, last) sample of performance audio, with a small safety margin."""
    act = act_mod.analyze_activity(mono, sr)
    idx = np.flatnonzero(act.speech_mask)
    if len(idx) == 0:
        return None
    m = int(EDGE_MARGIN_S * sr)
    return max(0, idx[0] * act.hop - m), min(len(mono), idx[-1] * act.hop + act.frame + m)


def _quietest_window_db(mono: np.ndarray, sr: int, seconds: float = 0.5) -> float:
    w = int(sr * seconds)
    n = len(mono) // w
    if n == 0:
        return float(10 * np.log10(np.mean(mono.astype(np.float64) ** 2) + 1e-12))
    b = mono[: n * w].reshape(n, w)
    return float(10 * np.log10(np.einsum("ij,ij->i", b, b, dtype=np.float64).min() / w + 1e-12))


def measure(y: np.ndarray, sr: int) -> dict:
    mono = y if y.ndim == 1 else np.mean(y, axis=1)
    rms = float(10 * np.log10(np.einsum("i,i->", mono, mono, dtype=np.float64) / max(len(mono), 1) + 1e-12))
    peak = float(20 * np.log10(max(float(np.max(np.abs(y))), 1e-9)))
    b = speech_bounds(mono, sr)
    head, tail = (b[0] / sr, (len(mono) - b[1]) / sr) if b else (0.0, 0.0)
    return {"rms_db": round(rms, 2), "peak_db": round(peak, 2), "noise_floor_db": round(_quietest_window_db(mono, sr), 1),
            "head_s": round(head, 2), "tail_s": round(tail, 2), "sample_rate": sr,
            "channels": 1 if y.ndim == 1 else y.shape[1]}


def check(m: dict, mp3: bool = False) -> list[tuple[str, bool, str]]:
    """[(requirement, passed, detail)]. WAV masters are checked on level/edges only; sample rate
    and channel count are checked against the MP3 that gets uploaded."""
    items = [
        ("RMS -23..-18 dB", RMS_RANGE[0] <= m["rms_db"] <= RMS_RANGE[1], f'{m["rms_db"]:.1f} dB'),
        ("Peak <= -3 dB", m["peak_db"] <= PEAK_MAX_DB, f'{m["peak_db"]:.1f} dB'),
        ("Noise floor <= -60 dB", m["noise_floor_db"] <= NOISE_MAX_DB, f'{m["noise_floor_db"]:.1f} dB'),
        ("Head room tone 0.5-1 s", HEAD_RANGE_S[0] <= m["head_s"] <= HEAD_RANGE_S[1] + 0.05, f'{m["head_s"]:.2f} s'),
        ("Tail room tone 1-5 s", TAIL_RANGE_S[0] <= m["tail_s"] <= TAIL_RANGE_S[1], f'{m["tail_s"]:.2f} s'),
    ]
    if mp3:
        items.append(("44.1 kHz", m["sample_rate"] == MP3_SR, f'{m["sample_rate"]} Hz'))
        items.append(("Mono", m["channels"] == 1, f'{m["channels"]} ch'))
    return [(name, bool(ok), detail) for name, ok, detail in items]


def passed(items: list[tuple[str, bool, str]]) -> bool:
    return all(ok for _, ok, _ in items)


def summary(items: list[tuple[str, bool, str]]) -> str:
    fails = [f"{name} ({detail})" for name, ok, detail in items if not ok]
    return "ACX PASS" if not fails else "ACX FAIL: " + "; ".join(fails)


def enforce_rms(y: np.ndarray, sr: int, ceiling_dbfs: float) -> tuple[np.ndarray, dict]:
    """Final gain trim into the RMS window (through the true-peak limiter), never raising the
    noise floor past -61 dB. Usually a no-op: the loudness stage already lands inside."""
    stats: dict = {}
    for _ in range(2):
        m = measure(y, sr)
        if RMS_SAFE[0] <= m["rms_db"] <= RMS_SAFE[1]:
            break
        gain = RMS_TRIM_TARGET - m["rms_db"]
        gain = min(gain, (NOISE_MAX_DB - 1.0) - m["noise_floor_db"])
        if abs(gain) < 0.1:
            break
        y = limiter.limit(y * np.float32(10 ** (gain / 20.0)), sr, ceiling_dbfs)
        stats["acx_rms_trim_db"] = round(stats.get("acx_rms_trim_db", 0.0) + gain, 2)
    return y, stats
