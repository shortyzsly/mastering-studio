"""Presence / harshness control: a dynamic 2.2-4.4 kHz split-band compressor.

The 2-5 kHz region is where the ear is most sensitive, and where a voice sounds
"harsh" or "aggressive" when it spikes: emphasised words, bright vowels, mic
presence peaks. A static EQ cut here dulls every word equally; this compresses
just that band, only when it is loud relative to this speaker's own level in it
(threshold = a percentile of the band level over speech frames), so ordinary
speech is left alone and the edgy moments are pulled back.

Same split-band mechanism as the de-esser (isolate band, reduce, subtract), with
slower time constants (4 ms attack, 70 ms release) because this is vowel-scale
energy, not a fast fricative. Sits below the de-esser's band so the two do not
double up.

Strengths (threshold percentile / ratio / max reduction):
  light 88 / 1.8:1 / 3 dB    medium 84 / 2:1 / 4 dB    strong 76 / 2.5:1 / 6 dB
"""
from __future__ import annotations

import numpy as np

from . import deess

STRENGTHS = {
    "light": (88.0, 1.8, 3.0),
    "medium": (84.0, 2.0, 4.0),
    "strong": (76.0, 2.5, 6.0),
}
BAND_HZ = (2200.0, 4400.0)


def soften(x: np.ndarray, sr: int, strength: str = "medium") -> tuple[np.ndarray, dict]:
    return deess.deess(
        x, sr, strength, lo_hz=BAND_HZ[0], hi_hz=BAND_HZ[1], table=STRENGTHS,
        attack_ms=4.0, release_ms=70.0, prefix="presence",
    )
