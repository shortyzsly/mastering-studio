"""Room tone: keeps ACX-natural room tone at the file's head/tail, and smooths
over any mid-file gap that dips far below this file's own usual quiet level.

ACX wants 0.5-1 s of room tone -- not digital silence -- at the start and end
of the file, and rejects unnaturally deep drop-outs as much as it rejects too
much noise. Two things can leave a file without that:
  * the source recording simply starts or ends on speech, with no pause at
    the edge to begin with;
  * aggressive noise reduction (especially the neural cleanup pass at its
    unlimited setting) can occasionally push one isolated gap far below the
    file's typical quiet level -- an audible "hole" rather than steady tone.

Nothing here invents noise. Both fixes HARVEST this file's own cleanest quiet
stretches (per channel, at final output level, after every other stage) and
loop them, crossfaded, to whatever length is needed -- so head/tail padding
and any raised dip sound like more of the same recording, not synthetic hiss.

Two independent, separately-toggleable effects:
  * Head/tail pad: always brings the file up to `pad_s` (default 0.75 s,
    inside ACX's 0.5-1 s guidance) of room tone at each end, crossfaded into
    the existing start/end so there is no seam.
  * Gap-floor smoothing: raises a gap frame only if it falls more than
    `floor_below_typical_db` (default 10 dB) under this file's own MEDIAN gap
    level -- i.e. only genuine outlier dips, not the ordinary noise floor a
    file already has. On a file whose gaps are already even this typically
    finds nothing to do.
"""
from __future__ import annotations

import numpy as np

from . import activity as act_mod

DEFAULT_PAD_S = 0.75          # ACX asks for 0.5-1 s
DEFAULT_XFADE_MS = 40.0
DEFAULT_FLOOR_BELOW_TYPICAL_DB = 10.0
MIN_OUTLIER_RUN_S = 0.08
BED_TARGET_S = 3.0            # harvested bed length before it has to repeat; longer = less audible periodicity


def _crossfade_join(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """a followed by b, with the last n samples of a equal-power-crossfaded
    into the first n samples of b (a and b must each have >= n samples)."""
    n = min(n, len(a), len(b))
    if n < 2:
        return np.concatenate([a, b])
    w = np.linspace(0.0, 1.0, n)
    seam = a[-n:] * (1.0 - w) + b[:n] * w
    return np.concatenate([a[:-n], seam, b[n:]])


def harvest_bed(x: np.ndarray, sr: int, act: act_mod.Activity, target_s: float = BED_TARGET_S) -> np.ndarray | None:
    """A representative stretch of this channel's own quiet room tone, built
    by crossfade-joining its longest quiet stretches until `target_s` is
    reached (or they run out). None if the channel has no usable quiet audio
    at all (essentially only possible for a file that never pauses)."""
    ranges = act_mod.frame_ranges(act.noise_mask, act.hop, act.frame, len(x))
    if not ranges:
        # Fall back to the quietest 5% of 20 ms frames -- coarser, but this
        # only triggers when the file has no run long/flat enough to count
        # as clean room tone by the normal detector.
        quiet = act.level_db <= np.percentile(act.level_db, 5)
        ranges = act_mod.frame_ranges(quiet, act.hop, act.frame, len(x))
    if not ranges:
        return None
    xfade_n = max(8, int(DEFAULT_XFADE_MS / 1000 * sr))
    ranges = sorted(ranges, key=lambda r: -(r[1] - r[0]))
    bed = None
    for s, e in ranges:
        seg = np.asarray(x[s:e], dtype=np.float64)
        bed = seg if bed is None else _crossfade_join(bed, seg, xfade_n)
        if len(bed) >= target_s * sr:
            break
    return bed.astype(np.float32) if bed is not None else None


def _loopify(bed: np.ndarray, xfade_n: int) -> np.ndarray:
    """Blend `bed`'s own tail into its own head so tiling it end-to-end has
    no seam at the wrap point."""
    xfade_n = min(xfade_n, len(bed) // 3)
    if xfade_n < 8:
        return bed
    w = np.linspace(0.0, 1.0, xfade_n)
    out = bed.copy()
    out[:xfade_n] = bed[-xfade_n:] * (1.0 - w) + bed[:xfade_n] * w
    return out


def tone(bed: np.ndarray, n: int, sr: int) -> np.ndarray:
    """`n` samples of room tone built by tiling `bed` (looped seamlessly)."""
    xfade_n = max(8, int(DEFAULT_XFADE_MS / 1000 * sr))
    loop = _loopify(bed, xfade_n)
    if len(loop) == 0:
        return np.zeros(n, dtype=np.float32)
    reps = -(-n // len(loop))
    return np.tile(loop, reps)[:n].astype(np.float32)


def pad_head_tail(
    x: np.ndarray,
    sr: int,
    act: act_mod.Activity,
    head_s: float = DEFAULT_PAD_S,
    tail_s: float = DEFAULT_PAD_S,
    bed: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Prepend/append `head_s`/`tail_s` seconds of this channel's own room
    tone, crossfaded into the existing start/end. 0 for either disables that
    side. Returns (audio, stats)."""
    if head_s <= 0 and tail_s <= 0:
        return x, {}
    bed = bed if bed is not None else harvest_bed(x, sr, act)
    if bed is None:
        return x, {"roomtone_pad": "skipped - no quiet audio found to harvest"}

    xfade_n = max(8, int(DEFAULT_XFADE_MS / 1000 * sr))
    out = x
    if head_s > 0:
        head = tone(bed, int(round(head_s * sr)) + xfade_n, sr)
        out = _crossfade_join(head, out, xfade_n)
    if tail_s > 0:
        tail = tone(bed, int(round(tail_s * sr)) + xfade_n, sr)
        out = _crossfade_join(out, tail, xfade_n)
    return out.astype(np.float32), {
        "roomtone_head_s": head_s,
        "roomtone_tail_s": tail_s,
    }


def raise_outlier_gaps(
    x: np.ndarray,
    sr: int,
    act: act_mod.Activity,
    below_typical_db: float = DEFAULT_FLOOR_BELOW_TYPICAL_DB,
    bed: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Blend harvested room tone into any gap stretch whose level falls more
    than `below_typical_db` under this file's own median gap level. Ordinary
    gaps (the file's normal noise floor) are left bit-identical."""
    gap = act.noise_mask
    if gap.sum() < 20:
        return x, {}
    typical_db = float(np.median(act.level_db[gap]))
    thresh_db = typical_db - below_typical_db
    outlier = gap & (act.level_db < thresh_db)
    min_run = max(1, int(round(MIN_OUTLIER_RUN_S * sr / act.hop)))
    padded = np.concatenate(([False], outlier, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_run]
    if not runs:
        return x, {"roomtone_floor_typical_db": round(typical_db, 1), "roomtone_floor_raised_pct": 0.0}

    bed = bed if bed is not None else harvest_bed(x, sr, act)
    if bed is None:
        return x, {}

    # Needed extra RMS (linear amplitude, added on top of what's already there) to
    # bring EACH FRAME up to the threshold, not one average over the whole run --
    # a run that mixes a deep dip with shallower edges would otherwise dilute the
    # correction the deep part actually needs. act.level_db is power-referenced
    # (10*log10(power)), matching how it's built in dsp.activity.
    n_frames = len(act.level_db)
    thr_pow = 10 ** (thresh_db / 10.0)
    need_amp = np.zeros(n_frames)
    for a, b in runs:
        need_amp[a:b] = np.sqrt(np.maximum(0.0, thr_pow - 10 ** (act.level_db[a:b] / 10.0)))
    need_amp = np.convolve(need_amp, np.ones(5) / 5.0, mode="same")  # ~50 ms smoothing, no frame-to-frame steps
    centers = act.frame / 2 + np.arange(n_frames) * act.hop
    need_amp_s = np.interp(np.arange(len(x)), centers, need_amp).astype(np.float32)

    out = x.astype(np.float32, copy=True)
    xfade_n = max(4, int(15.0 / 1000 * sr))  # short fade: these are brief, isolated dips
    total_raised = 0
    for a, b in runs:
        s, e = a * act.hop, min(len(x), b * act.hop + act.frame)
        n = e - s
        if n <= 0:
            continue
        fill = tone(bed, n, sr)
        fill_rms = float(np.sqrt(np.mean(fill.astype(np.float64) ** 2)) + 1e-12)
        # Amplitude-modulate the (unit-RMS) harvested tone by the per-sample need,
        # so a dip in the middle of a run is corrected fully even if its edges need less.
        added = (fill / fill_rms) * need_amp_s[s:e]
        blended = x[s:e] + added
        f = min(xfade_n, n // 2)
        if f >= 2:
            w = np.linspace(0.0, 1.0, f)
            blended[:f] = x[s : s + f] * (1 - w) + blended[:f] * w
            blended[-f:] = x[e - f : e] * w[::-1] + blended[-f:] * (1 - w[::-1])
        out[s:e] = blended
        total_raised += n
    return out, {
        "roomtone_floor_typical_db": round(typical_db, 1),
        "roomtone_floor_threshold_db": round(thresh_db, 1),
        "roomtone_floor_raised_pct": round(100.0 * total_raised / len(x), 2),
        "roomtone_floor_events": len(runs),
    }
