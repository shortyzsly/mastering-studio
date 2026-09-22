"""Pure-NumPy stand-in for DeepFilterNet's Rust `libdf` (STFT analysis/synthesis, ERB features,
exponential normalisation). Only the pieces needed for inference."""
import math
import numpy as np
from scipy.signal import lfilter

def _freq2erb(f): return 9.265 * math.log1p(f / (24.7 * 9.265))
def _erb2freq(n): return 24.7 * 9.265 * (math.exp(n / 9.265) - 1.0)

def _erb_widths(sr, fft_size, nb_bands, min_nb_freqs):
    nyq = sr // 2; fw = sr / fft_size
    lo, hi = _freq2erb(0.0), _freq2erb(nyq); step = (hi - lo) / nb_bands
    erb = [0] * nb_bands; prev = 0; over = 0
    for i in range(1, nb_bands + 1):
        f = _erb2freq(i * step + lo); fb = int(round(f / fw)); n = fb - prev - over
        if n < min_nb_freqs: over = min_nb_freqs - n; n = min_nb_freqs
        else: over = 0
        erb[i - 1] = n; prev = fb
    erb[-1] += 1
    too_large = sum(erb) - (fft_size // 2 + 1)
    if too_large > 0: erb[-1] -= too_large
    assert sum(erb) == fft_size // 2 + 1, (sum(erb), fft_size)
    return erb

class DF:
    def __init__(self, sr, fft_size, hop_size, nb_bands, min_nb_erb_freqs=2, **kw):
        self._sr, self._n, self._h, self._nb = sr, fft_size, hop_size, nb_bands
        self._widths = _erb_widths(sr, fft_size, nb_bands, min_nb_erb_freqs)
        i = np.arange(fft_size)
        self._win = np.sin(0.5 * np.pi * np.sin(np.pi * (i + 0.5) / fft_size) ** 2).astype(np.float64)   # Vorbis window
        self._wnorm = 1.0 / (fft_size ** 2 / (2 * hop_size))
    def sr(self): return self._sr
    def fft_size(self): return self._n
    def hop_size(self): return self._h
    def nb_erb(self): return self._nb
    def erb_widths(self): return np.array(self._widths, dtype=np.uintp)
    def analysis(self, audio):
        a = np.asarray(audio, dtype=np.float64); C, T = a.shape; H, N = self._h, self._n; nf = T // H
        out = np.empty((C, nf, N // 2 + 1), dtype=np.complex64)
        for c in range(C):
            buf = np.concatenate([np.zeros(N - H), a[c, : nf * H]])          # analysis memory starts at zeros
            idx = np.arange(nf)[:, None] * H + np.arange(N)[None, :]
            out[c] = (np.fft.rfft(buf[idx] * self._win[None, :], axis=1) * self._wnorm).astype(np.complex64)
        return out
    def synthesis(self, spec):
        s = np.asarray(spec); C, nf, F = s.shape; H, N = self._h, self._n
        out = np.zeros((C, nf * H + (N - H)), dtype=np.float64)
        for c in range(C):
            fr = np.fft.irfft(s[c].astype(np.complex128), n=N, axis=1) * N * self._win[None, :]
            for t in range(nf): out[c, t * H : t * H + N] += fr[t]
        return out[:, : nf * H].astype(np.float32)                              # block t = input samples [(t-1)H, tH): delay = one hop

def erb(spec, erb_widths, db=True):
    s = np.asarray(spec); w = [int(x) for x in erb_widths]; p = (s.real ** 2 + s.imag ** 2).astype(np.float64)
    out = np.empty(p.shape[:-1] + (len(w),), dtype=np.float32); k = 0
    for i, width in enumerate(w):
        out[..., i] = p[..., k : k + width].mean(axis=-1); k += width
    return (10.0 * np.log10(out + 1e-10)).astype(np.float32) if db else out

def erb_inv(erb_feat, erb_widths):
    e = np.asarray(erb_feat); w = [int(x) for x in erb_widths]
    return np.concatenate([np.repeat(e[..., i : i + 1], width, axis=-1) for i, width in enumerate(w)], axis=-1)

def erb_norm(erb_feat, alpha):
    x = np.asarray(erb_feat, dtype=np.float64); B = x.shape[-1]; init = np.linspace(-60.0, -90.0, B)
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        st, _ = lfilter([1 - alpha], [1, -alpha], x[c], axis=0, zi=(alpha * init)[None, :])
        out[c] = (x[c] - st) / 40.0
    return out.astype(np.float32)

def unit_norm_init(n): return np.linspace(0.001, 0.0001, n).astype(np.float32)

def unit_norm(spec, alpha):
    x = np.asarray(spec); F = x.shape[-1]; init = unit_norm_init(F).astype(np.float64); out = np.empty_like(x)
    for c in range(x.shape[0]):
        st, _ = lfilter([1 - alpha], [1, -alpha], np.abs(x[c]).astype(np.float64), axis=0, zi=(alpha * init)[None, :])
        out[c] = (x[c] / np.sqrt(st)).astype(x.dtype)
    return out
