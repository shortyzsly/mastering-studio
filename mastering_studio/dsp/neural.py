"""Optional neural denoiser stage (DeepFilterNet3), run as a helper process.

Why it exists: a Wiener-type filter cannot remove noise that shares a
time-frequency bin with the voice. A neural model can partly, because it
resynthesises speech. Measured on this narrator's recording (noise injected at a
known level, see README): under the softest words it removes 1-3 dB more of the
added noise than the Wiener filter at 3-16 kHz, and 1 dB more under soft-mid words,
while preserving speech far better than RNNoise (word onsets -0.7 dB vs -24.5 dB,
0.3% vs 4% of speech frames dropped). It does not remove low-frequency room noise
in gaps (-1.4 dB vs -14.7 dB for the Wiener filter), so the pipeline runs the Wiener
denoiser after it at reduced depth.

Isolation: the model needs PyTorch (~1 GB on disk), which is kept out of the app's
own environment and out of git (it's platform/arch-specific, so a Linux venv would
not run on Mac/Windows anyway). It runs in a separate virtualenv, created by
`scripts/setup_neural_denoiser.py`, at $MASTERING_STUDIO_NEURAL_PYTHON if set,
else ~/.mastering-studio/neural-venv (OS-appropriate bin/python vs.
Scripts/python.exe), run via `_dfn_worker.py`. If that environment is missing,
`available()` is False and the GUI/CLI option is disabled; nothing else is
affected.

Temporary files go to ~/.cache/mastering-studio, NOT /tmp: /tmp is RAM-backed on this
machine (3.9 GB) and a 20-minute file is 230 MB per copy.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy import signal

def _venv_python_path(venv_dir: str) -> str:
    """The interpreter path inside a venv, matching how `python -m venv` and
    `scripts/setup_neural_denoiser.py` lay it out on each OS."""
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


DEFAULT_VENV_DIR = os.path.expanduser(os.path.join("~", ".mastering-studio", "neural-venv"))
DEFAULT_PYTHON = _venv_python_path(DEFAULT_VENV_DIR)
CACHE_DIR = os.path.expanduser("~/.cache/mastering-studio")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dfn_worker.py")
MODEL_SR = 48000
EXTRA_GB = 1.2   # peak extra memory while the helper runs (torch + model + chunk buffers)

_available_cache: dict[str, bool] = {}


def python_path() -> str:
    return os.environ.get("MASTERING_STUDIO_NEURAL_PYTHON", DEFAULT_PYTHON)


def installed() -> bool:
    """Fast check (no subprocess): does the neural environment's Python exist?"""
    py = python_path()
    return os.path.isfile(py) and os.access(py, os.X_OK)


def available() -> bool:
    py = python_path()
    if py in _available_cache:
        return _available_cache[py]
    ok = False
    if os.path.isfile(py) and os.access(py, os.X_OK):
        try:
            r = subprocess.run([py, "-c", "import torch, soundfile, scipy; import importlib.util as u; assert u.find_spec('df')"],
                               capture_output=True, timeout=120)
            ok = r.returncode == 0
        except Exception:
            ok = False
    _available_cache[py] = ok
    return ok


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    f = Fraction(sr_out, sr_in).limit_denominator(1000)
    return signal.resample_poly(x, f.numerator, f.denominator).astype(np.float32)


def enhance(x: np.ndarray, sr: int, atten_lim_db: float | None = None, threads: int = 2) -> np.ndarray:
    """x: mono float32 (n,). Returns float32 of the same length."""
    if not available():
        raise RuntimeError(f"neural denoiser unavailable (no usable Python with torch+deepfilternet at {python_path()})")
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="dfn_", dir=CACHE_DIR)
    try:
        src, dst = os.path.join(tmp, "in.wav"), os.path.join(tmp, "out.wav")
        sf.write(src, _resample(np.asarray(x, dtype=np.float32), sr, MODEL_SR), MODEL_SR, subtype="FLOAT")
        env = dict(os.environ, MS_NEURAL_THREADS=str(threads))
        cmd = [python_path(), WORKER, src, dst, "none" if atten_lim_db is None else str(atten_lim_db)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise RuntimeError("neural denoiser failed:\n" + (r.stderr or r.stdout)[-1500:])
        y, _ = sf.read(dst, dtype="float32")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    y = _resample(y, MODEL_SR, sr)
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y[: len(x)]
