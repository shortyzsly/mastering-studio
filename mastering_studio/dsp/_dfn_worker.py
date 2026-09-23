"""DeepFilterNet3 worker. RUN WITH THE NEURAL ENVIRONMENT'S PYTHON, not the app's:

    <neural-venv>/bin/python _dfn_worker.py in.wav out.wav [atten_lim_db]

Reads a mono float WAV at 48 kHz, enhances it in 30 s chunks (with 2 s of context
each side, discarded, so the recurrent state is warm at every chunk boundary), and
writes a float WAV of the same length.

The upstream `deepfilternet` package needs `libdf`, a Rust extension. Where a real
wheel installed for this interpreter (e.g. Python 3.10-3.13 on a platform PyPI
has one for), it's used as-is -- it's the genuine, better-tested implementation.
Only where none is installed (this had no wheel at all for Python 3.14, which is
why this fallback exists) does `_libdf_shim.py`, a NumPy replacement for the few
functions used at inference (STFT analysis/synthesis, ERB features, exponential
normalisation; STFT round-trip exact to -162 dBFS), stand in for it. A stub for
`torchaudio.backend.common` is also provided because current torchaudio removed
it and deepfilternet imports it only for a type annotation.
"""
import dataclasses
import importlib.util
import os
import sys
import types

if importlib.util.find_spec("libdf") is None:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _here)
    import _libdf_shim  # noqa: E402

    sys.modules["libdf"] = _libdf_shim

_backend = types.ModuleType("torchaudio.backend")
_common = types.ModuleType("torchaudio.backend.common")


@dataclasses.dataclass
class AudioMetaData:
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0
    bits_per_sample: int = 0
    encoding: str = ""


_common.AudioMetaData = AudioMetaData
sys.modules.setdefault("torchaudio.backend", _backend)
sys.modules.setdefault("torchaudio.backend.common", _common)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("MS_NEURAL_THREADS", "2")))
from df.enhance import enhance, init_df  # noqa: E402

CHUNK_S = 30
CONTEXT_S = 2


def main(path_in: str, path_out: str, atten_lim_db) -> None:
    x, sr = sf.read(path_in, dtype="float32")
    if sr != 48000:
        raise SystemExit(f"expected 48 kHz, got {sr}")
    if x.ndim != 1:
        raise SystemExit("mono input expected")
    model, df_state, _ = init_df(log_level="WARNING")
    y = np.empty_like(x)
    chunk, ctx = CHUNK_S * sr, CONTEXT_S * sr
    for s in range(0, len(x), chunk):
        e = min(len(x), s + chunk)
        a, b = max(0, s - ctx), min(len(x), e + ctx)
        seg = torch.from_numpy(np.ascontiguousarray(x[a:b])[None, :])
        o = enhance(model, df_state, seg, pad=True, atten_lim_db=atten_lim_db)[0].numpy()
        y[s:e] = o[s - a : s - a + (e - s)]
    sf.write(path_out, y, sr, subtype="FLOAT")


if __name__ == "__main__":
    lim = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "none" else None
    main(sys.argv[1], sys.argv[2], lim)
