"""Orchestrates analysis -> automatic parameter selection -> DSP chain.

Chain order follows standard restoration/mastering practice (iZotope RX
dialogue workflow, Auphonic): rumble/DC removal, declick, de-hum, HF de-hiss, spectral denoise,
subtractive EQ + voice polish + low-end (mud/bass), de-breath, compression, de-esser, then a single loudness gain into a true-peak
limiter. Denoise comes BEFORE any dynamics/gain so nothing ever amplifies
the noise floor; loudness and limiting come last and only once.

Multi-channel files: denoise/declick/EQ run per channel (each learns its own
room tone); compression, loudness and limiting use a single linked gain so
the stereo image never shifts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import io_utils
from .analysis import Analysis, analyze
from .dsp import (
    activity as activity_mod,
    compressor,
    declick as declick_mod,
    dehiss as dehiss_mod,
    dehum as dehum_mod,
    bass as bass_mod,
    debreath as debreath_mod,
    deess as deess_mod,
    denoise as denoise_mod,
    eq as eq_mod,
    highpass,
    limiter,
    loudness,
    neural as neural_mod,
    roomtone as roomtone_mod,
    soften as soften_mod,
)


@dataclass
class PipelineSettings:
    enable_highpass: bool = True
    enable_declick: bool = True
    enable_dehum: bool = True
    enable_dehiss: bool = True
    enable_neural: bool = field(default_factory=lambda: neural_mod.installed())   # DeepFilterNet3 cleanup pass; on when the neural env exists
    neural_atten_limit_db: Optional[float] = None   # None = unlimited (gaps ~-95 dBFS): the setting the user preferred by ear. 12 keeps ~-90
    neural_position: str = "after"     # "after" the Wiener denoiser (a cleanup pass); "before" measured WORSE (breaks the expander/Wiener synergy)
    neural_wiener_db: float = 8.0     # Wiener depth when it follows the neural model (it cleans the low-frequency room noise the model leaves)
    dehiss_strength: str = "medium"   # HF hiss under soft words: "light" | "medium" | "strong"
    enable_denoise: bool = True
    enable_eq: bool = True
    enable_compress: bool = True
    enable_loudness: bool = True
    enable_limiter: bool = True

    # None => auto-selected from analysis
    highpass_hz: float = 80.0
    enable_deess: bool = True
    deess_strength: str = "medium"      # "light" | "medium" | "strong"
    enable_debreath: bool = True
    debreath_strength: str = "medium"   # "light" (-6 dB) | "medium" (-10) | "strong" (-14)
    warmth_db: float = 0.0        # optional bell at 150 Hz (off: the low-end stage tightens instead)
    enable_lowend: bool = True
    lowend_strength: str = "medium"   # "light" | "medium" | "strong": mud dip + bass shelf + dynamic bass control
    hf_tame: bool = True          # ease the top end (curve chosen by soften_strength)
    enable_soften: bool = True    # harshness control: static top-end curve + dynamic 2.2-4.4 kHz presence control
    soften_strength: str = "strong"   # "light" | "medium" | "strong"; strong is the default: evening the bass leaves the top relatively brighter, and by ear "a little less harsh" was wanted
    denoise_strength: Optional[str] = None       # "gentle" | "moderate" | "aggressive"
    denoise_reduction_db: Optional[float] = None  # explicit dB, overrides strength
    eq_amount: float = 1.0
    target_lufs: float = loudness.ACX_LUFS
    peak_ceiling_dbfs: float = loudness.DEFAULT_CEILING_DBFS
    compressor_override: Optional[tuple[float, float]] = None  # (threshold_db, ratio)

    enable_roomtone: bool = True
    roomtone_head_s: float = roomtone_mod.DEFAULT_PAD_S   # ACX asks for 0.5-1 s of room tone at each end
    roomtone_tail_s: float = roomtone_mod.DEFAULT_PAD_S
    enable_roomtone_floor: bool = True   # smooth over isolated gaps far quieter than this file's own typical gap
    roomtone_floor_below_typical_db: float = roomtone_mod.DEFAULT_FLOOR_BELOW_TYPICAL_DB


@dataclass
class ProcessResult:
    pre_analysis: Analysis
    post_analysis: Analysis
    settings_used: dict
    clicks_fixed: int
    output_path: str


StageCallback = Callable[[str, float], None]


def process_array(
    data: np.ndarray,
    sr: int,
    settings: PipelineSettings,
    pre: Analysis,
    on_stage: Optional[StageCallback] = None,
) -> tuple[np.ndarray, dict]:
    """data: (n,) or (n, ch) float32. Returns (processed, settings_used)."""
    used: dict = {}
    x = data[:, None] if data.ndim == 1 else data
    chans = [np.ascontiguousarray(x[:, c]) for c in range(x.shape[1])]

    def stage(msg: str, frac: float) -> None:
        if on_stage:
            on_stage(msg, frac)

    if settings.enable_highpass:
        chans = [highpass.remove_dc_and_rumble(c, sr, settings.highpass_hz) for c in chans]
        stage("Removing rumble/DC offset", 0.10)

    clicks_fixed = 0
    if settings.enable_declick:
        fixed = []
        for i, c in enumerate(chans):
            c, n = declick_mod.declick(c, sr)
            fixed.append(c)
            clicks_fixed += n
        chans = fixed
        stage("De-clicking (mouth noise)", 0.20)
    used["clicks_fixed"] = clicks_fixed

    # Denoise learns its noise profile from the pre-de-hum audio, so hum stays part of the
    # "noise" it treats under speech. Learn it now and drop the reference (saves a full copy).
    profiles = None
    if settings.enable_denoise and settings.enable_dehum:
        profiles = [
            denoise_mod.learn_noise_psd(c, sr, activity_mod.analyze_activity(c, sr)) for c in chans
        ]
    if settings.enable_dehum:
        out = []
        for c in chans:
            c, hum_stats = dehum_mod.dehum(c, sr)
            out.append(c)
            if hum_stats:
                used.update(hum_stats)
        chans = out
        stage("Removing mains hum" if "hum_f0_hz" in used else "Checking for hum (none found)", 0.30)

    if settings.enable_dehiss:
        out = []
        for c in chans:
            c, hs = dehiss_mod.dehiss(c, sr, settings.dehiss_strength)
            out.append(c)
            for k, v in hs.items():
                used[k] = v
        chans = out
        stage("De-hissing top octaves", 0.40)

    neural_used = False
    if settings.enable_neural and settings.neural_position == "before":
        if neural_mod.available():
            chans = [neural_mod.enhance(c, sr, settings.neural_atten_limit_db) for c in chans]
            profiles = None   # the Wiener denoiser now learns its profile from the neural output's room tone
            neural_used = True
            used["neural"] = "DeepFilterNet3 (before Wiener)"
            stage("Neural denoise (DeepFilterNet3)", 0.45)
        else:
            used["neural"] = "unavailable - skipped"

    if settings.enable_denoise:
        if settings.denoise_reduction_db is not None:
            reduction = settings.denoise_reduction_db
        elif settings.denoise_strength:
            reduction = denoise_mod.STRENGTH_PRESETS_DB.get(settings.denoise_strength, 10.0)
        elif neural_used:
            reduction = settings.neural_wiener_db
        else:
            gain_needed = (settings.target_lufs - pre.integrated_lufs) if settings.enable_loudness else 0.0
            reduction = denoise_mod.auto_reduction_db(pre.noise_floor_dbfs, gain_needed)
        used["denoise_reduction_db"] = reduction
        out = []
        for i, c in enumerate(chans):
            act = activity_mod.analyze_activity(c, sr)
            psd = profiles[i] if profiles else None
            out.append(denoise_mod.denoise(c, sr, reduction_db=reduction, activity=act, noise_psd=psd))
        chans = out
        stage(f"Spectral denoise ({reduction:.0f} dB)", 0.55)

    if settings.enable_neural and settings.neural_position == "after":
        if neural_mod.available():
            chans = [neural_mod.enhance(c, sr, settings.neural_atten_limit_db) for c in chans]
            used["neural"] = "DeepFilterNet3 (after Wiener)"
            stage("Neural denoise (DeepFilterNet3)", 0.58)
        else:
            used["neural"] = "unavailable - skipped"

    if (settings.enable_eq and settings.eq_amount > 0) or settings.enable_lowend:
        out, cuts_used, low_info = [], {}, {}
        for c in chans:
            act = activity_mod.analyze_activity(c, sr)
            cuts, shelf, bells = {}, 0.0, []
            if settings.enable_eq and settings.eq_amount > 0:
                cuts = eq_mod.compute_cuts(eq_mod.measure_speech_bands(c, sr, act))
                if settings.enable_lowend:  # the low-end stage owns <600 Hz; don't stack cuts
                    cuts = {f: g for f, g in cuts.items() if f >= eq_mod.LOWEND_OWNS_BELOW_HZ}
                cuts_used = cuts or cuts_used
            if settings.enable_lowend:
                fine = eq_mod.measure_speech_bands(c, sr, act, eq_mod.FINE_CENTERS, octave_frac=6.0)
                shelf, bells, low_info = eq_mod.lowend_plan(fine, settings.lowend_strength)
            eq_on = settings.enable_eq and settings.eq_amount > 0
            out.append(
                eq_mod.apply_cuts(
                    c, sr, cuts, settings.eq_amount if eq_on else 0.0,
                    settings.warmth_db if eq_on else 0.0,
                    ((settings.soften_strength if settings.enable_soften else "base") if (settings.hf_tame and eq_on) else False),
                    low_shelf_db=shelf, bells=bells,
                )
            )
        chans = out
        used["eq_cuts_db"] = {int(round(f)): round(g, 1) for f, g in cuts_used.items()}
        used.update(low_info)
        stage("Corrective EQ + low-end", 0.65)

    y = np.stack(chans, axis=1)

    if settings.enable_lowend:
        y, bass_stats = bass_mod.tighten(y, sr, settings.lowend_strength)
        used.update(bass_stats)
        stage("Tightening bass", 0.68)

    if settings.enable_debreath:
        reduction = debreath_mod.STRENGTH_DB.get(settings.debreath_strength, 10.0)
        y, events = debreath_mod.debreath(y, sr, reduction_db=reduction)
        used["breaths_reduced"] = len(events)
        used["breath_reduction_db"] = round(float(np.mean([e.reduction_db for e in events])), 1) if events else 0.0
        stage("De-breath", 0.70)

    if settings.enable_compress:
        if settings.compressor_override:
            threshold_db, ratio = settings.compressor_override
        else:
            act = activity_mod.analyze_activity(np.max(np.abs(y), axis=1), sr) if y.shape[1] > 1 else activity_mod.analyze_activity(y[:, 0], sr)
            crest = float(np.percentile(act.level_db, 95) - act.speech_db)
            threshold_db, ratio = compressor.auto_settings(act.speech_db, crest)
        used["compressor_threshold_db"] = round(threshold_db, 1)
        used["compressor_ratio"] = ratio
        y = compressor.compress(y, sr, threshold_db=threshold_db, ratio=ratio)
        stage("Compression", 0.75)

    if settings.enable_soften:
        y, stats = soften_mod.soften(y, sr, settings.soften_strength)
        used.update(stats)
        stage("Softening harsh presence", 0.82)

    if settings.enable_deess:
        y, stats = deess_mod.deess(y, sr, strength=settings.deess_strength)
        used.update(stats)
        stage("De-essing", 0.85)

    if settings.enable_loudness or settings.enable_limiter:
        y = loudness.normalize_and_limit(
            y,
            sr,
            target_lufs=settings.target_lufs,
            ceiling_dbfs=settings.peak_ceiling_dbfs,
            apply_gain=settings.enable_loudness,
            apply_limit=settings.enable_limiter,
        )
        stage("Loudness + true-peak limiting", 0.95)

    if settings.enable_roomtone or settings.enable_roomtone_floor:
        out = []
        rt_stats: dict = {}
        for c in range(y.shape[1]):
            ch = y[:, c]
            act = activity_mod.analyze_activity(ch, sr)
            if settings.enable_roomtone_floor:
                ch, st = roomtone_mod.raise_outlier_gaps(ch, sr, act, settings.roomtone_floor_below_typical_db)
                rt_stats.update(st)
                if st.get("roomtone_floor_raised_pct"):
                    act = activity_mod.analyze_activity(ch, sr)  # refresh before harvesting the pad bed
            if settings.enable_roomtone:
                ch, st = roomtone_mod.pad_head_tail(ch, sr, act, settings.roomtone_head_s, settings.roomtone_tail_s)
                rt_stats.update(st)
            out.append(ch)
        y = np.stack(out, axis=1)
        used.update(rt_stats)
        stage("Room tone (head/tail + gap smoothing)", 1.0)

    y = y[:, 0] if data.ndim == 1 else y
    return y.astype(np.float32, copy=False), used


def process_file(
    input_path: str,
    output_path: str,
    settings: Optional[PipelineSettings] = None,
    on_stage: Optional[StageCallback] = None,
) -> ProcessResult:
    settings = settings or PipelineSettings()

    data, sr = io_utils.load_audio(input_path)
    channels = 1 if data.ndim == 1 else data.shape[1]

    if on_stage:
        on_stage("Analyzing", 0.02)
    pre = analyze(io_utils.to_mono(data), sr, channels=channels)

    processed, used = process_array(data, sr, settings, pre, on_stage)
    io_utils.save_audio(output_path, processed, sr)

    post = analyze(io_utils.to_mono(processed), sr, channels=channels)
    return ProcessResult(
        pre_analysis=pre,
        post_analysis=post,
        settings_used=used,
        clicks_fixed=used.get("clicks_fixed", 0),
        output_path=output_path,
    )
