"""DC-offset removal and rumble high-pass filter.

Always-safe, always-on stage: removing sub-audible rumble (HVAC, mic
handling, traffic) never audibly harms speech and lets every downstream
stage work on a cleaner signal.

2nd-order Butterworth run zero-phase (forward+backward): effective slope is
24 dB/oct with no phase smear, -6 dB at the cutoff and about -1 dB by 1.4x
the cutoff. 80 Hz tightens rumble/mud while a warmth bell in dsp.eq gives
back the 100-250 Hz body.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import blocks


def remove_dc_and_rumble(x: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return blocks.sosfiltfilt_blocks(sos, x, sr, offset=float(np.mean(x, dtype=np.float64)))
