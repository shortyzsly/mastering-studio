"""Spectral denoise: learned room-tone profile + MMSE-LSA gain with
decision-directed a-priori SNR (Ephraim & Malah 1984/85).

This is the same family of algorithm as iZotope RX Spectral De-noise
(learn a noise profile, subtract it, smooth to avoid "musical noise") and
what the speech-enhancement literature uses. The parts that matter:

  * The noise profile is learned from detected room tone only (see
    dsp.activity), never from the whole clip.
  * The gain is driven by the *a priori* SNR, estimated decision-directed
    with alpha=0.98. That recursion carries the previous frame's clean
    estimate forward, which is what stops isolated noise bins flickering in
    and out -- the "warble"/"watery" artifact of plain spectral gating.
  * Reduction is capped in dB (a gain floor). Pro guidance is roughly 6-15
    dB (plus extra at the noise-only extremes); a floor also keeps natural-sounding room tone rather than digital
    silence (which ACX rejects anyway).
"""
from __future__ import annotations

import numpy as np
from scipy.special import exp1

from . import activity as act_mod

# Presets map to a *reduction in dB*, the unit RX / Auphonic expose.
STRENGTH_PRESETS_DB = {"gentle": 6.0, "moderate": 10.0, "aggressive": 18.0}

# Extra reduction where there is (almost) no speech: below ~120 Hz (rumble) and
# above ~6-12 kHz (hiss; speech has very little energy past 12 kHz). Speech-
# dominated bins keep G~1 regardless, so this only lowers residual noise.
HF_EXTRA_DB = 5.0
LF_EXTRA_DB = 6.0

ALPHA = 0.98          # decision-directed smoothing
OVERSUB = 1.0         # noise-PSD over-subtraction factor (1.0 = trust the learned profile)
# Decision-directed estimation carries the previous frame's clean-speech estimate
# forward with weight ALPHA, i.e. a memory of ~50 frames (~0.5 s). Right after a
# word that memory keeps the gain near 1, so hiss in the short gaps between words
# gets almost no reduction. When a frame is clearly just room tone (level within
# RESET_ABOVE_FLOOR_DB of the floor, and the next ~40 ms too, so a word onset is
# never pre-empted) the memory is reset to the a-priori-SNR floor.
GAP_RESET = True
RESET_ABOVE_FLOOR_DB = 8.0
RESET_LOOKAHEAD_FRAMES = 4
XI_MIN_DB = -25.0     # a-priori SNR floor
MAX_NOISE_FRAMES = 4000


def _n_fft(sr: int) -> int:
    return 1 << int(round(np.log2(0.0427 * sr)))  # ~43 ms: 2048 @ 48k


def _window(n_fft: int) -> np.ndarray:
    # sqrt-Hann analysis + synthesis at 75% overlap reconstructs perfectly.
    return np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)


def auto_reduction_db(noise_floor_dbfs: float, gain_to_target_db: float) -> float:
    """Pick the reduction from where the noise will *end up* after loudness
    gain, not from where it is now: the gain stage lifts the noise with the
    speech, so what matters is the final floor vs the ACX -60 dB line (we
    aim for -75 for a clearly clean floor), clamped to 14-16 dB. Raised from 12-14 once
    de-hum and the gap reset (below) removed the two artifacts that made deeper
    reduction look bad; the warble proxy stays at natural-noise level to 18 dB.
    HF/LF get extra on top (see HF_EXTRA_DB)."""
    final_floor = noise_floor_dbfs + max(gain_to_target_db, 0.0)
    need = final_floor - (-75.0)
    return float(np.clip(need, 14.0, 16.0))


def _reduction_curve_db(sr: int, n_fft: int, reduction_db: float) -> np.ndarray:
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    hf = np.clip((np.log2(np.maximum(f, 1.0)) - np.log2(6000.0)) / (np.log2(12000.0) - np.log2(6000.0)), 0.0, 1.0)
    lf = np.clip((np.log2(160.0) - np.log2(np.maximum(f, 1.0))) / (np.log2(160.0) - np.log2(80.0)), 0.0, 1.0)
    return reduction_db + HF_EXTRA_DB * hf + LF_EXTRA_DB * lf


def learn_noise_psd(x: np.ndarray, sr: int, activity: act_mod.Activity) -> np.ndarray:
    """Mean power spectrum (per FFT bin) of the detected room tone."""
    n_fft = _n_fft(sr)
    win = _window(n_fft)
    hop = n_fft // 2
    frames = []
    for a, b in act_mod.frame_ranges(activity.noise_mask, activity.hop, activity.frame, len(x)):
        for s in range(a, b - n_fft, hop):
            frames.append(x[s : s + n_fft])
    if not frames:
        raise ValueError("no room tone found to learn a noise profile from")
    if len(frames) > MAX_NOISE_FRAMES:
        pick = np.linspace(0, len(frames) - 1, MAX_NOISE_FRAMES).astype(int)
        frames = [frames[i] for i in pick]
    spec = np.fft.rfft(np.stack(frames) * win, axis=1)
    psd = np.mean(np.abs(spec) ** 2, axis=0)
    # Smooth over ~5 bins: a noise estimate is a statistic, not a fingerprint;
    # over-fitted profile detail is another source of tonal artifacts.
    k = np.ones(5) / 5.0
    padded = np.pad(psd, 2, mode="edge")
    return np.maximum(np.convolve(padded, k, mode="valid"), 1e-14)


EXPOSED_GAMMA = 20.0   # a-posteriori SNR below which noise is considered audible (unmasked) in that bin
STAT_BANDS_HZ = ((80, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 10000), (10000, 20000))


def denoise(
    x: np.ndarray,
    sr: int,
    reduction_db: float = 10.0,
    activity: act_mod.Activity | None = None,
    noise_psd: np.ndarray | None = None,
    stats: dict | None = None,
) -> np.ndarray:
    """If `stats` is a dict it is filled with diagnostics computed from the gains
    actually applied: expected noise attenuation (dB, per band) in speech frames
    and in room-tone frames, plus a speech-damage proxy and gain roughness."""
    if reduction_db <= 0.0:
        return x
    x = x.astype(np.float32, copy=False)
    if noise_psd is None:
        activity = activity or act_mod.analyze_activity(x, sr)
        noise_psd = learn_noise_psd(x, sr, activity)

    n_fft = _n_fft(sr)
    hop = n_fft // 4
    win = _window(n_fft)
    g_min = 10 ** (-_reduction_curve_db(sr, n_fft, reduction_db) / 20.0)  # per bin
    xi_min = 10 ** (XI_MIN_DB / 10.0)
    lam = noise_psd * OVERSUB
    lam_true = noise_psd

    n = len(x)
    n_frames = (n + n_fft) // hop + 1
    padded_len = (n_frames - 1) * hop + n_fft
    xp = np.zeros(padded_len, dtype=np.float32)
    xp[n_fft // 2 : n_fft // 2 + n] = x
    out = np.zeros(padded_len, dtype=np.float32)

    gap = np.zeros(n_frames, dtype=bool)
    if GAP_RESET:
        act = activity or act_mod.analyze_activity(x, sr)
        quiet = act.level_db <= act.floor_db + RESET_ABOVE_FLOOR_DB
        run = quiet.copy()
        for s_ in range(1, RESET_LOOKAHEAD_FRAMES + 1):
            run[:-s_] &= quiet[s_:]
            run[-s_:] = False
        # STFT frame i is centred on original sample i*hop; activity frame j on j*ahop + aframe/2
        j = np.clip(np.round((np.arange(n_frames) * hop - act.frame / 2) / act.hop).astype(int), 0, len(run) - 1)
        gap = run[j]

    speech_f = np.zeros(n_frames, dtype=bool)
    if stats is not None:
        act = activity or act_mod.analyze_activity(x, sr)
        thr = act.floor_db + max(15.0, 0.4 * (np.percentile(act.level_db, 95) - act.floor_db))
        jj = np.clip(np.round((np.arange(n_frames) * hop - act.frame / 2) / act.hop).astype(int), 0, len(act.level_db) - 1)
        speech_f = act.level_db[jj] >= thr
        fbin = np.fft.rfftfreq(n_fft, 1.0 / sr)
        bandm = np.stack([((fbin >= lo) & (fbin < hi)).astype(np.float64) for lo, hi in STAT_BANDS_HZ], axis=1)  # (bins, nb)
        acc = {k: np.zeros(len(STAT_BANDS_HZ)) for k in ("sp_num", "sp_den", "gp_num", "gp_den", "ex_num", "ex_den", "ex_cnt", "sp_cnt")}
        dmg_num = dmg_den = 0.0
        rough_sum = rough_n = 0.0
        prev_g = None

    prev = np.ones_like(lam)  # previous frame's (G^2 * gamma) = A^2 / lambda
    batch = 512
    for f0 in range(0, n_frames, batch):
        f1 = min(n_frames, f0 + batch)
        idx = (np.arange(f0, f1)[:, None] * hop) + np.arange(n_fft)[None, :]
        spec = np.fft.rfft(xp[idx] * win, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float64)
        gains = np.empty_like(power)
        for i in range(f1 - f0):
            if gap[f0 + i]:
                prev = np.full_like(prev, xi_min)
            gamma = power[i] / lam
            xi = ALPHA * prev + (1.0 - ALPHA) * np.maximum(gamma - 1.0, 0.0)
            xi = np.maximum(xi, xi_min)
            nu = xi / (1.0 + xi)
            v = np.maximum(nu * gamma, 1e-8)
            g = np.minimum(nu * np.exp(0.5 * exp1(v)), 1.0)
            g = np.maximum(g, g_min)
            gains[i] = g
            prev = g * g * gamma
            if stats is not None:
                fi = f0 + i
                w = g * g * lam_true
                key = "sp" if speech_f[fi] else "gp"
                acc[key + "_num"] += w @ bandm
                acc[key + "_den"] += lam_true @ bandm
                if speech_f[fi]:
                    exposed = (gamma < EXPOSED_GAMMA * OVERSUB).astype(np.float64)   # noise within ~13 dB of the bin's signal: not masked
                    acc["ex_num"] += (w * exposed) @ bandm
                    acc["ex_den"] += (lam_true * exposed) @ bandm
                    acc["ex_cnt"] += exposed @ bandm
                    acc["sp_cnt"] += bandm.sum(axis=0)
                    clear = gamma > 30.0 * OVERSUB                        # bins that are clearly speech
                    dmg_num += float(np.sum(power[i][clear] * (1.0 - g[clear] ** 2)))
                    dmg_den += float(np.sum(power[i][clear]))
                    if prev_g is not None:
                        mid = (gamma > 2.0) & (gamma < 12.0)             # ambiguous bins: where musical noise lives
                        if mid.any():
                            rough_sum += float(np.mean(np.abs(20 * np.log10(g[mid] / np.maximum(prev_g[mid], 1e-6)))))
                            rough_n += 1
                prev_g = g
        frames = np.fft.irfft(spec * gains, n=n_fft, axis=1).astype(np.float32) * win
        for i in range(f1 - f0):
            s = (f0 + i) * hop
            out[s : s + n_fft] += frames[i]
    if stats is not None:
        na = lambda a, b: [round(float(10 * np.log10(max(n_ / max(d_, 1e-30), 1e-12))), 1) for n_, d_ in zip(acc[a], acc[b])]
        stats["noise_atten_in_speech_db"] = dict(zip([f"{lo}-{hi}" for lo, hi in STAT_BANDS_HZ], na("sp_num", "sp_den")))
        stats["noise_atten_in_gaps_db"] = dict(zip([f"{lo}-{hi}" for lo, hi in STAT_BANDS_HZ], na("gp_num", "gp_den")))
        stats["noise_atten_exposed_db"] = dict(zip([f"{lo}-{hi}" for lo, hi in STAT_BANDS_HZ], na("ex_num", "ex_den")))
        stats["exposed_bins_pct"] = dict(zip([f"{lo}-{hi}" for lo, hi in STAT_BANDS_HZ], [round(100 * float(c / max(t, 1))) for c, t in zip(acc["ex_cnt"], acc["sp_cnt"])]))
        stats["speech_damage_db"] = round(float(10 * np.log10(max(1.0 - dmg_num / max(dmg_den, 1e-30), 1e-6))), 2)
        stats["gain_roughness_db"] = round(rough_sum / max(rough_n, 1), 2)
    out *= 0.5  # sum of sqrt-Hann^2 at 75% overlap is 2.0
    return out[n_fft // 2 : n_fft // 2 + n]
