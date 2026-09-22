"""Analyze a voice/audiobook recording and report the metrics the pipeline
uses to auto-select processing settings: noise floor, loudness, spectral
balance, clipping, DC offset, and click density.

All measurements are descriptive only -- this module never applies gain or
filtering. Keeping analysis and processing separate makes it possible to
inspect *why* the pipeline chose a setting before trusting the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .dsp import activity as activity_mod, blocks, declick as declick_mod

# 1/3-octave-ish center frequencies spanning speech-relevant range.
OCTAVE_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


@dataclass
class Analysis:
    sample_rate: int
    duration_s: float
    channels: int

    peak_dbfs: float
    integrated_lufs: float
    noise_floor_dbfs: float
    dynamic_range_db: float
    dc_offset: float
    clipped_sample_pct: float
    click_count: int
    click_rate_per_min: float

    # relative energy per octave band in dB, normalized so the mean is 0 dB
    band_freqs: list[int] = field(default_factory=lambda: list(OCTAVE_BANDS))
    band_levels_db: list[float] = field(default_factory=list)

    notes: list[str] = field(default_factory=list)

    rms_dbfs: float = -120.0             # whole-file RMS (what ACX measures)
    speech_rms_dbfs: float = -120.0      # active-speech RMS
    quietest_500ms_dbfs: float = -120.0  # quietest 0.5 s window (ACX noise-floor check)


def _frame_rms_db(mono: np.ndarray, sr: int, frame_ms: float = 50.0) -> np.ndarray:
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(mono) // frame_len
    if n_frames == 0:
        return np.array([-120.0])
    trimmed = mono[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(np.maximum(rms, 1e-9))


def estimate_noise_floor_db(mono: np.ndarray, sr: int) -> float:
    """Room-tone level (dBFS RMS) from the activity detector's 5th percentile
    of 20 ms frames -- the same figure the denoiser and gain staging use."""
    return float(activity_mod.analyze_activity(mono, sr).floor_db)


def quietest_window_db(mono: np.ndarray, sr: int, seconds: float = 0.5) -> float:
    w = int(sr * seconds)
    n = len(mono) // w
    if n == 0:
        return float(20 * np.log10(np.sqrt(np.mean(mono.astype(np.float64) ** 2)) + 1e-9))
    blocks = mono[: n * w].reshape(n, w)
    p = np.einsum("ij,ij->i", blocks, blocks, dtype=np.float64) / w
    return float(10 * np.log10(p.min() + 1e-12))


def estimate_click_density(mono: np.ndarray, sr: int) -> tuple[int, float]:
    """Count isolated mouth-click events (same detector the declicker uses,
    so what is reported is what gets fixed)."""
    if len(mono) < sr // 10:
        return 0, 0.0
    n_clicks = len(declick_mod.find_clicks(mono, sr))
    duration_min = len(mono) / sr / 60.0
    return n_clicks, (n_clicks / duration_min if duration_min > 0 else 0.0)


def octave_band_levels(mono: np.ndarray, sr: int) -> list[float]:
    """Relative energy (dB) in each band of OCTAVE_BANDS, mean-centered.

    Uses Welch PSD then integrates octave-weighted bins per band (weighting
    by the linear frequency span each FFT bin actually covers) rather than a
    plain arithmetic mean over linearly spaced bins -- a plain mean
    over-weights the octave-sparse high end and previously caused a ~12 dB
    underestimate of low-frequency content in this codebase's earlier EQ
    module. See project memory: audiofix_processing_pitfalls.
    """
    nperseg = min(65536, len(mono))
    if nperseg < 256:
        return [0.0] * len(OCTAVE_BANDS)
    # scipy's welch materialises every segment of the input at once (~1 GB per 20 min);
    # average the PSD over ~90 s pieces instead. Equal-length pieces => plain mean.
    piece = max(nperseg * 8, 1 << 22)
    psds, weights = [], []
    for s0 in range(0, len(mono), piece):
        seg = mono[s0 : s0 + piece]
        if len(seg) < nperseg:
            continue
        freqs, p = signal.welch(seg, fs=sr, nperseg=nperseg)
        psds.append(p)
        weights.append(len(seg))
    if not psds:
        return [0.0] * len(OCTAVE_BANDS)
    psd = np.average(np.stack(psds), axis=0, weights=weights)
    levels = []
    for center in OCTAVE_BANDS:
        lo, hi = center / np.sqrt(2), center * np.sqrt(2)
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            levels.append(-120.0)
            continue
        band_energy = np.trapezoid(psd[mask], freqs[mask])
        levels.append(10 * np.log10(max(band_energy, 1e-15)))
    levels = np.array(levels)
    finite = levels[levels > -100]
    center_ref = np.mean(finite) if len(finite) else 0.0
    return (levels - center_ref).tolist()


def analyze(mono: np.ndarray, sr: int, channels: int = 1) -> Analysis:
    duration_s = len(mono) / sr

    peak = max(float(mono.max()), float(-mono.min())) if len(mono) else 0.0
    peak_dbfs = 20 * np.log10(max(peak, 1e-9))

    try:
        integrated_lufs = blocks.lufs_integrated(mono, sr)
    except Exception:
        integrated_lufs = -70.0
    if not np.isfinite(integrated_lufs):
        integrated_lufs = -70.0

    act = activity_mod.analyze_activity(mono, sr)
    noise_floor = float(act.floor_db)
    loud_ref = float(np.percentile(act.level_db, 95))
    dynamic_range = loud_ref - noise_floor
    rms_dbfs = float(10 * np.log10(np.einsum("i,i->", mono, mono, dtype=np.float64) / max(len(mono), 1) + 1e-12))

    dc_offset = float(np.mean(mono, dtype=np.float64)) if len(mono) else 0.0

    clipped = int(np.count_nonzero(mono >= 0.999) + np.count_nonzero(mono <= -0.999))
    clipped_pct = 100.0 * clipped / max(1, len(mono))

    click_count, click_rate = estimate_click_density(mono, sr)
    band_levels = octave_band_levels(mono, sr)

    notes = []
    if noise_floor > -60:
        notes.append("Noise floor above ACX's -60 dB -- denoising needed.")
    if clipped_pct > 0.001:
        notes.append("Clipped samples detected -- source already distorted.")
    if abs(dc_offset) > 0.01:
        notes.append("DC offset present -- will be removed.")
    if click_rate > 5:
        notes.append("Frequent clicks/pops detected -- likely mouth noise/plosives.")

    return Analysis(
        sample_rate=sr,
        duration_s=duration_s,
        channels=channels,
        peak_dbfs=peak_dbfs,
        integrated_lufs=integrated_lufs,
        noise_floor_dbfs=noise_floor,
        dynamic_range_db=dynamic_range,
        rms_dbfs=rms_dbfs,
        speech_rms_dbfs=float(act.speech_db),
        quietest_500ms_dbfs=quietest_window_db(mono, sr),
        dc_offset=dc_offset,
        clipped_sample_pct=clipped_pct,
        click_count=click_count,
        click_rate_per_min=click_rate,
        band_levels_db=band_levels,
        notes=notes,
    )
