"""Orchestrates analysis -> automatic parameter selection -> DSP chain.

Chain order follows standard restoration/mastering practice (iZotope RX
dialogue workflow, Auphonic): rumble/DC removal, declick, de-hum, HF de-hiss, spectral denoise,
subtractive EQ + voice polish + low-end (mud/bass), de-breath, compression, tone match, harshness control, de-esser, then a single loudness gain into a true-peak
limiter. Denoise comes BEFORE any dynamics/gain so nothing ever amplifies
the noise floor; loudness and limiting come last and only once.

Multi-channel files: denoise/declick/EQ run per channel (each learns its own
room tone); compression, loudness and limiting use a single linked gain so
the stereo image never shifts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

import os

from . import acx, auto, io_utils
from .analysis import Analysis, analyze
from .dsp import (
    activity as activity_mod,
    compressor,
    declick as declick_mod,
    dereverb as dereverb_mod,
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
    mic: str = "generic"               # auto.MIC_PROFILES key: "pd70" | "stellar_x2" | "generic"
    smart_auto: bool = True            # analyze each file and choose every stage's strength (auto.decide)
    enable_roomfix: bool = True        # small-booth fix: low-end tone match + room-mode notches + de-reverb
    dereverb_strength: str = "auto"    # "auto" (from the measured T60) | "off" | "light" | "medium" | "strong"
    room_t60_s: Optional[float] = None        # measured by Smart Auto; estimated in-stage otherwise
    enable_tonematch: bool = True      # match the voice's long-term balance to eq.TONE_TARGET_REL_1K
    tone_boost_scale: float = 1.0      # how much of eq.TONE_LIMITS' boost the tone match may use (mic profile)
    eq_low: str = "off"                # eq.EQ_PRESETS keys, applied on top of everything automatic
    eq_mid: str = "off"
    eq_high: str = "off"
    declick_strength: str = "normal"   # "normal" | "strong"
    noise_region_s: Optional[tuple[float, float]] = None   # user-marked room tone to learn the noise profile from
    export_mp3: bool = False           # also write an ACX-spec MP3 (192 kbps CBR, 44.1 kHz)
    acx_enforce: bool = True           # final RMS trim into ACX's window (only with loudness on)
    enable_highpass: bool = True
    enable_declick: bool = True
    enable_dehum: bool = True
    enable_dehum_under_speech: bool = False   # experimental: narrow-notch detected hum harmonics under speech too, not just gaps (see dsp.dehum.notch_under_speech)
    dehum_under_speech_mix: float = dehum_mod.NOTCH_MIX
    enable_dehiss: bool = True
    enable_neural: bool = field(default_factory=lambda: neural_mod.installed())   # DeepFilterNet3 cleanup pass; on when the neural env exists
    neural_atten_limit_db: Optional[float] = 12.0   # gentler default (~-90 dBFS in gaps): unlimited (None, ~-95 dBFS) pushes the neural
    # model harder toward silence, which is the classic cause of faint tonal/"bell" artifacts between words. Set to None to re-enable it.
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
    roomtone_head_s: float = roomtone_mod.DEFAULT_PAD_S   # ACX: 0.5-1 s at the head
    roomtone_tail_s: float = 2.0                          # ACX: 1-5 s at the tail
    enable_roomtone_floor: bool = True   # smooth over isolated gaps far quieter than this file's own typical gap
    roomtone_floor_below_typical_db: float = roomtone_mod.DEFAULT_FLOOR_BELOW_TYPICAL_DB


@dataclass
class ProcessResult:
    pre_analysis: Analysis
    post_analysis: Analysis
    settings_used: dict
    clicks_fixed: int
    output_path: str
    acx_pre: list = field(default_factory=list)    # acx.check() of the raw input
    acx_post: list = field(default_factory=list)   # ... of the master (WAV)
    acx_mp3: list = field(default_factory=list)    # ... of the decoded MP3, if exported
    mp3_path: str = ""
    diagnosis: list = field(default_factory=list)


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
            c, n = declick_mod.declick(c, sr, settings.declick_strength)
            fixed.append(c)
            clicks_fixed += n
        chans = fixed
        stage("De-clicking (mouth noise)", 0.20)
    used["clicks_fixed"] = clicks_fixed

    # Denoise learns its noise profile from the pre-de-hum audio, so hum stays part of the
    # "noise" it treats under speech. Learn it now and drop the reference (saves a full copy).
    profiles = None
    region_floor_db = None
    if settings.enable_denoise and settings.noise_region_s:
        a, b = (int(round(t * sr)) for t in settings.noise_region_s)
        profiles = [denoise_mod.noise_psd_from_audio(c[a:b], sr) for c in chans]
        seg = chans[0][a:b].astype(np.float64)
        region_floor_db = float(10 * np.log10(np.mean(seg * seg) + 1e-12))
        used["noise_profile"] = f"user selection {settings.noise_region_s[0]:.2f}-{settings.noise_region_s[1]:.2f} s"
    elif settings.enable_denoise and settings.enable_dehum:
        profiles = [
            denoise_mod.learn_noise_psd(c, sr, activity_mod.analyze_activity(c, sr)) for c in chans
        ]
    if settings.enable_dehum:
        out = []
        for c in chans:
            act_c = activity_mod.analyze_activity(c, sr)
            info = dehum_mod.detect(c, sr, act_c)
            c, hum_stats = dehum_mod.dehum(c, sr, act_c, info)
            if info and settings.enable_dehum_under_speech:
                c, notch_stats = dehum_mod.notch_under_speech(
                    c, sr, info, act_c.floor_db, act_c.speech_db, mix=settings.dehum_under_speech_mix
                )
                hum_stats.update(notch_stats)
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
            floor = region_floor_db if region_floor_db is not None else pre.noise_floor_dbfs
            reduction = denoise_mod.auto_reduction_db(floor, gain_needed)
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

    strength = settings.dereverb_strength if settings.enable_roomfix else "off"
    if strength != "off":
        t60 = settings.room_t60_s or dereverb_mod.estimate_t60(chans[0], sr)
        if strength == "auto":
            strength = dereverb_mod.auto_strength(t60) or "off"
        if strength != "off":
            t60 = t60 or 0.25   # forced on a file too dry to measure: assume a small booth
            st: dict = {}
            chans = [dereverb_mod.dereverb(c, sr, t60, strength, st) for c in chans]
            used["dereverb"] = f"{strength} (T60 {t60:.2f} s)"
            used.update(st)
            stage(f"De-reverb ({strength})", 0.60)

    eq_on = settings.enable_eq and settings.eq_amount > 0
    tone_on = settings.enable_tonematch and (eq_on or settings.enable_roomfix)
    tone_owns_lows = tone_on and settings.enable_roomfix
    if eq_on or settings.enable_lowend or settings.enable_roomfix:
        out, cuts_used, low_info = [], {}, {}
        for c in chans:
            act = activity_mod.analyze_activity(c, sr)
            cuts, shelf, bells = {}, 0.0, []
            if eq_on:
                cuts = eq_mod.compute_cuts(eq_mod.measure_speech_bands(c, sr, act))
                if settings.enable_lowend:  # the low-end stage owns <600 Hz; don't stack cuts
                    cuts = {f: g for f, g in cuts.items() if f >= eq_mod.LOWEND_OWNS_BELOW_HZ}
                cuts_used = cuts or cuts_used
            if settings.enable_lowend:
                fine = eq_mod.measure_speech_bands(c, sr, act, eq_mod.FINE_CENTERS, octave_frac=6.0)
                shelf, bells, low_info = eq_mod.lowend_plan(fine, settings.lowend_strength)
                if tone_owns_lows:   # the final tone match sets the broad low balance from a real measurement
                    shelf, bells = 0.0, bells[1:]
                    low_info = {k: v for k, v in low_info.items() if k == "mud_peaks"}
            shapes = []
            if settings.enable_roomfix:
                res, res_info = eq_mod.find_resonances(c, sr, act)
                if res:   # a narrow room-mode notch replaces the low-end stage's broader mud bell at that frequency
                    keep = lambda hz: all(abs(np.log2(hz / r[1])) > 0.25 for r in res)
                    first = 0 if tone_owns_lows else 1   # bells[0] is the broad dip unless the tone match owns lows
                    bells = bells[:first] + [b for b in bells[first:] if keep(b[0])]
                    if "mud_peaks" in low_info:
                        low_info["mud_peaks"] = [m for m in low_info["mud_peaks"] if keep(m["hz"])]
                shapes += res
                used.update(res_info)
            kw = dict(
                cuts=cuts, amount=settings.eq_amount if eq_on else 0.0,
                warmth_db=settings.warmth_db if eq_on else 0.0,
                hf_tame=((settings.soften_strength if settings.enable_soften else "base") if (settings.hf_tame and eq_on) else False),
                low_shelf_db=shelf, bells=bells, low_shelf_top_hz=low_info.get("low_shelf_top_hz", 320.0),
            )
            out.append(eq_mod.apply_curve(c, sr, lambda f: eq_mod.curve_db(f, shapes=shapes, **kw)))
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

    user_shapes = eq_mod.preset_shapes(settings.eq_low, settings.eq_mid, settings.eq_high)
    if tone_on or user_shapes:
        # Tonal balance, measured AFTER bass control and compression (both change the long-term spectrum),
        # so it corrects what is really there. User presets ride on top. Harshness control and de-essing
        # come AFTER it: run before, the tone match's presence/air lift re-added the sibilance they removed.
        out = []
        for c in range(y.shape[1]):
            ch = y[:, c]
            applied: dict[int, float] = {}
            for p in range(eq_mod.TONE_PASSES if tone_on else 1):
                shapes = []
                if tone_on:   # each pass re-measures and corrects what is still off target
                    act = activity_mod.analyze_activity(ch, sr)
                    bands = eq_mod.measure_speech_bands(ch, sr, act, eq_mod.TONE_CENTERS, octave_frac=3.0)
                    gaps = replace(act, speech_mask=~act.speech_mask)
                    snr = bands - eq_mod.measure_speech_bands(ch, sr, gaps, eq_mod.TONE_CENTERS, octave_frac=3.0)
                    shapes, tm_info = eq_mod.tone_match(
                        bands, np.zeros(len(bands)), strength=settings.eq_amount if eq_on else 1.0,
                        max_boost_scale=settings.tone_boost_scale if eq_on else 0.0,
                        min_hz=0.0 if settings.enable_roomfix else eq_mod.LOWEND_OWNS_BELOW_HZ, applied=applied, snr_db=snr,
                    )
                    if not eq_on:   # room fix alone: only its low/low-mid part
                        shapes = [sh for sh in shapes if sh[1] < eq_mod.LOWEND_OWNS_BELOW_HZ]
                    for sh in shapes:
                        applied[int(sh[1])] = applied.get(int(sh[1]), 0.0) + sh[2]
                if p == eq_mod.TONE_PASSES - 1 or not tone_on:
                    shapes = shapes + user_shapes
                ch = eq_mod.apply_curve(ch, sr, lambda f: eq_mod.shapes_db(np.log2(np.maximum(f, 1.0)), shapes))
            if tone_on:
                used["tone_match_db"] = {k: round(v, 1) for k, v in sorted(applied.items())}
            out.append(ch)
        y = np.stack(out, axis=1)
        if user_shapes:
            used["eq_presets"] = {"low": settings.eq_low, "mid": settings.eq_mid, "high": settings.eq_high}
        stage("Tone match + EQ presets", 0.88)

    if settings.enable_soften:
        y, stats = soften_mod.soften(y, sr, settings.soften_strength)
        used.update(stats)
        stage("Softening harsh presence", 0.90)

    if settings.enable_deess:
        y, stats = deess_mod.deess(y, sr, strength=settings.deess_strength)
        used.update(stats)
        stage("De-essing", 0.92)

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
        bounds = acx.speech_bounds(y[:, 0] if y.shape[1] == 1 else np.max(np.abs(y), axis=1), sr)   # shared: channels stay aligned
        for c in range(y.shape[1]):
            ch = y[:, c]
            act = activity_mod.analyze_activity(ch, sr)
            if settings.enable_roomtone_floor:
                ch, st = roomtone_mod.raise_outlier_gaps(ch, sr, act, settings.roomtone_floor_below_typical_db)
                rt_stats.update(st)
                if st.get("roomtone_floor_raised_pct"):
                    act = activity_mod.analyze_activity(ch, sr)  # refresh before harvesting the pad bed
            if settings.enable_roomtone and bounds:
                ch, st = roomtone_mod.fit_head_tail(ch, sr, act, bounds, settings.roomtone_head_s, settings.roomtone_tail_s)
                rt_stats.update(st)
            out.append(ch)
        n_min = min(len(o) for o in out)
        y = np.stack([o[:n_min] for o in out], axis=1)
        used.update(rt_stats)
        stage("Room tone (exact ACX head/tail + gap smoothing)", 0.97)

    if settings.acx_enforce and settings.enable_loudness:
        y, st = acx.enforce_rms(y[:, 0] if y.shape[1] == 1 else y, sr, settings.peak_ceiling_dbfs)
        y = y[:, None] if y.ndim == 1 else y
        used.update(st)
        stage("ACX compliance check", 1.0)

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
    mono = io_utils.to_mono(data)
    pre = analyze(mono, sr, channels=channels)
    if settings.smart_auto:
        if on_stage:
            on_stage("Smart Auto: deciding what this file needs", 0.05)
        settings, diagnosis = auto.decide(settings, mono, sr, pre)
    else:
        settings, diagnosis = auto.apply_mic(settings), ["Smart Auto off: using your settings"]
    del mono

    processed, used = process_array(data, sr, settings, pre, on_stage)
    io_utils.save_audio(output_path, processed, sr)
    used["diagnosis"] = diagnosis

    acx_mp3, mp3_path = [], ""
    if settings.export_mp3:
        mp3_path = os.path.splitext(output_path)[0] + ".mp3"
        io_utils.save_mp3(mp3_path, processed, sr, kbps=acx.MP3_KBPS, out_sr=acx.MP3_SR)
        dec, dsr = io_utils.load_audio(mp3_path)   # check what ACX will actually receive
        acx_mp3 = acx.check(acx.measure(dec, dsr), mp3=True)

    post = analyze(io_utils.to_mono(processed), sr, channels=channels)
    return ProcessResult(
        pre_analysis=pre,
        post_analysis=post,
        settings_used=used,
        clicks_fixed=used.get("clicks_fixed", 0),
        output_path=output_path,
        acx_pre=acx.check(acx.measure(data, sr)),
        acx_post=acx.check(acx.measure(processed, sr)),
        acx_mp3=acx_mp3,
        mp3_path=mp3_path,
        diagnosis=diagnosis,
    )


def diagnose(input_path: str, settings: Optional[PipelineSettings] = None) -> tuple[list, list[str]]:
    """Preview without processing: (ACX check of the raw file, what Smart Auto would do)."""
    settings = settings or PipelineSettings()
    data, sr = io_utils.load_audio(input_path)
    mono = io_utils.to_mono(data)
    pre = analyze(mono, sr, channels=1 if data.ndim == 1 else data.shape[1])
    notes = auto.decide(settings, mono, sr, pre)[1] if settings.smart_auto else ["Smart Auto off: your settings will be used"]
    return acx.check(acx.measure(data, sr)), list(pre.notes) + notes
