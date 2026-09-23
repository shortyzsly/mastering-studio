# Mastering Studio

Native audio mastering app for voice/audiobook recordings. PySide6 GUI plus a
CLI; batch-processes several files in parallel (separate OS processes), sizing
the parallelism from the files and the free RAM.

## Setup

Needs Python 3.10+ (the reference machine happens to run 3.14; nothing here
requires that specific version except as noted under Neural denoiser below).

**Linux / macOS:**
```
git clone <this repo's URL>
cd mastering-studio
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -m mastering_studio.gui
.venv/bin/python -m mastering_studio.cli in.wav -o out/ --target-lufs -23 --peak -3
```

**Windows (PowerShell):**
```
git clone <this repo's URL>
cd mastering-studio
py -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\python -m mastering_studio.gui
.venv\Scripts\python -m mastering_studio.cli in.wav -o out\ --target-lufs -23 --peak -3
```

PySide6/numpy/scipy/soundfile/noisereduce/pyloudnorm all ship prebuilt wheels
for Mac and Windows, so `pip install -e .` should just work. The DSP code
itself (pure NumPy/SciPy) has no OS-specific assumptions. What's untested on
Mac/Windows: the app has only actually been run on Linux so far -- if
something breaks, it's most likely a path assumption (see below) or a Qt/GUI
quirk, not the audio processing.

The optional neural denoiser (see below) is off until you run its setup
script; its default location, `~/.mastering-studio/neural-venv`, and its temp
files under `~/.cache/mastering-studio`, both resolve correctly on Windows too
(under `%USERPROFILE%`) via `os.path.expanduser`, but haven't actually been
tested there.

## Signal chain (and why it's built this way)

Modeled on how iZotope RX, Auphonic and broadcast mastering chains work.

1. **Rumble/DC removal** -- 80 Hz high-pass (2nd-order Butterworth, zero-phase =
   24 dB/oct effective), tightens the low end.
2. **De-click** -- finds *isolated* 1-3 ms spikes (>3 kHz energy 20 dB above the
   local median), repairs with LPC interpolation. Leaves speech, plosives and
   fricatives alone.
3. **De-hum** (`dsp/dehum.py`) -- finds mains hum/buzz (a comb at multiples of
   50 or 60 Hz) in the room tone at 0.7 Hz resolution and subtracts it in the
   gaps by fitting each line's amplitude/phase locally (a sinusoidal model), not
   with notch filters. Outside gaps the audio is bit-identical. Reports "no hum"
   and touches nothing if there is no comb. GUI: Auto / Off; CLI `--dehum`.
4. **HF de-hiss** (`dsp/dehiss.py`) -- a downward expander on two top-octave
   bands (5.5-9.5 kHz, 9.5 kHz-Nyquist). Threshold = that band's own room-tone
   level + 12/16/20 dB (Light/Medium/Strong), 1.7/2.2/3:1, capped at 6/10/14 dB.
   Runs BEFORE the denoiser so the room-tone level it measures is the true hiss
   level. Only touches frames whose top-octave level is close to the hiss (soft
   syllables, word edges, gaps); loud consonants are 0.0 dB. This is the only
   stage that lowers hiss *under words*, see "Design rules". CLI `--dehiss`.
5. **Spectral denoise** -- learns the noise profile from detected room tone,
   then MMSE-LSA gain with decision-directed a-priori SNR (Ephraim-Malah).
   Reduction is capped in dB: auto picks 14-16 dB (presets: Gentle 6 /
   Moderate 10 / Aggressive 18), plus extra reduction below 160 Hz and above
   6 kHz where there is only rumble/hiss and no speech. The speech-memory of the
   estimator is reset in frames that are clearly just room tone, so short gaps
   between words are cleaned too, not only long pauses. `denoise(stats={})`
   reports the noise attenuation it actually applies, per band, in speech / gap
   / exposed (unmasked) bins.
5b. **Neural denoise (`dsp/neural.py`, on by default when the neural env exists)** --
   DeepFilterNet3 run as a helper process in a separate virtualenv, applied as a
   cleanup pass AFTER the Wiener denoiser, attenuation UNLIMITED (the user preferred
   this by ear over the 12 dB cap; gaps end up near -95 dBFS). GUI: Neural denoiser
   On/Off; CLI `--neural auto|on|off`. See "Neural denoiser" below.
6. **EQ / voice polish** (one linear-phase FIR, so gains are exact) --
   measured cuts (max 2.5 dB, >=600 Hz only) for bumps >3 dB above the voice's
   own spectral trend, plus a top-end curve chosen by the Harshness setting
   (`eq.HF_CURVES`; Medium: -0.8 dB @ 3 kHz, -1.5 @ 4, -3.2 @ 6.3, -4.5 @ 8,
   -5.5 @ 10, -8 @ 14 kHz). GUI has Auto / Half strength / Off. An optional
   warmth bell (`warmth_db`, off by default) exists but adds low-end mass.
7. **Low end (mud & bass)** -- Light/Medium/Strong/Off, cuts only, in two parts:
   * *Static* (in the same FIR): a gentle low shelf below ~150 Hz (-0.5/-1/-2 dB),
     a broad dip centred at 320 Hz (-1/-2/-3.5 dB), and up to two narrow bells
     on the **mud peaks found in this voice** (the bands 180-520 Hz that stand
     furthest above their local +/-0.75 octave median; each cut removes ~85% of
     the excess, capped 2.5/3.5/5 dB). On the reference narrator these are ~252
     and ~400 Hz, stable across the whole chapter and NOT pitch harmonics (F0
     ~100-140 Hz puts harmonics in the valleys).
   * *Dynamic* (`dsp/bass.py`): a 70-280 Hz split-band compressor (280, not 200: the voice's 2nd harmonic, where the mud peak sits, swells with the fundamental), threshold =
     70th percentile of the speaker's own bass level in speech, 3:1, max 5 dB
     (Light 80th/2:1/3 dB, Strong 60th/4:1/8 dB). Pulls back boomy surges
     (proximity effect, plosive thump) while leaving steady body alone; nothing
     above ~400 Hz changes. CLI: `--lowend off|light|medium|strong`.
8. **De-breath** -- finds inhales and ducks them 6/10/14 dB (Light/Medium/
   Strong), never removing them and never below the room tone. A breath must be
   unvoiced (low autocorrelation), 12-34 dB below speech, 0.24-1.5 s long, start
   out of a pause, and not sibilant (<40% of energy above 5 kHz). Detection is
   seeded on clearly-qualifying frames and grown through the soft onset/decay
   ramps. Gain ramps end *inside* the event, so nothing outside a detected
   breath changes (verified: 0 samples). CLI: `--debreath off|light|medium|strong`.
9. **Compression** -- 1.6-2.5:1, threshold *relative to measured speech level*,
   downward-only (cannot raise the noise floor), gain-smoothed.
10. **Harshness / presence** (`dsp/soften.py`) -- dynamic 2.2-4.4 kHz split-band
    compressor (threshold = 88/84/76th percentile of the speaker's own level in
    that band, 1.8/2/2.5:1, max 3/4/6 dB, 4 ms attack / 70 ms release). Together
    with the static top-end curve in step 6 this is the "Harshness" setting.
    Compresses edgy moments, leaves ordinary speech alone. CLI `--soften`.
11. **De-esser** -- split-band, centred on the speaker's own sibilance peak
    (auto-detected, typically 4-9 kHz), threshold = a percentile of that band's
    level in speech (Light 95 / Medium 92 / Strong 88), 2-3:1, max 4/6/9 dB,
    speech frames only. CLI: `--deess off|light|medium|strong`.
12. **Loudness + true-peak limiter** -- one gain to the target integrated LUFS,
    then a 4x-oversampled lookahead limiter; gain is re-trimmed so the file
    lands on target *after* limiting.
13. **Room tone** (`dsp/roomtone.py`, runs LAST, on the final samples) -- ACX
    wants 0.5-1 s of room tone (not digital silence) at the head and tail, and
    dislikes an unnaturally deep drop-out as much as too much noise. Nothing
    is synthesized: this HARVESTS the file's own cleanest quiet stretches (per
    channel, crossfade-joined into a several-second bed) and loops that,
    crossfaded, wherever it's needed. Two independent effects:
    * Pad head/tail to `roomtone_head_s`/`roomtone_tail_s` (default 0.75 s
      each), crossfaded into the existing start/end.
    * Smooth over a gap that dips more than `roomtone_floor_below_typical_db`
      (default 10 dB) under this file's own MEDIAN gap level -- an isolated
      outlier, not the file's ordinary noise floor -- by adding just enough of
      the harvested tone, per-frame, to bring that stretch back up near its
      neighbours. A file with even gaps typically has nothing to smooth.
    GUI: On / Pad only / Smooth only / Off. CLI `--roomtone on|pad-only|floor-only|off`,
    `--roomtone-pad <seconds>`.

| Preset | Integrated | True-peak ceiling |
|---|---|---|
| ACX submission (default) | -21.5 LUFS (whole-file RMS ~ -21.9) | -3 dBTP |
| Audiobook | -23 LUFS (RMS -23.4: 0.4 dB UNDER ACX's floor) | -3 dBTP |
| Podcast | -16 LUFS | -1 dBTP |

ACX wants whole-file RMS in -23..-18 dB, peak <= -3 dB and noise floor <= -60 dB.
On this narration RMS is ~0.35 dB below LUFS, so -23 LUFS misses ACX's window; the ACX
preset lands ~1.1 dB inside it.

## Design rules learned the hard way

* **Never let early stages shrink the speech.** v1 lost ~10 dB of speech level
  across declick/denoise/EQ, then the loudness stage added +16 dB, which
  dragged the hiss up ~15 dB. Every stage now preserves speech level (check
  `settings_used` and the pre/post analysis).
* **Learn noise from room tone, not the whole clip.**
* **Denoise before any gain or dynamics.**
* **Filters run once.** `sosfiltfilt` squares a filter's gain (4 dB -> 8 dB).
* **Look at a spectrogram before tuning a detector.** The de-breath detector's
  first version looked like it failed a false-positive test (flagging 4-6
  "events"/min in real speech); spectrograms showed most were real quiet
  inhales, and the actual false positives were word-final decay tails. Tightening
  thresholds blindly would have deleted the true positives instead.
* **Scale synthetic test signals by RMS, not peak.** Gaussian noise has ~12 dB
  crest factor; peak-scaled test breaths were 12 dB fainter than intended.
* **Check whether a "peak" is the voice before cutting it.** Low-mid peaks can
  be pitch harmonics (cutting them thins the voice) or resonances (mud). Measure
  F0 first: harmonics of F0 must line up with the peaks to be harmonics.
* **Don't stack EQ stages on the same region.** The generic auto-cut and the
  low-end stage both hit 400 Hz and produced a -9 dB hole until the low-end
  stage was made the sole owner of <600 Hz.
* **Cutting bass lowers the voice, and the loudness stage makes it back up,
  which raises the hiss by the same amount** (~2 dB at Medium). Keep denoise in
  step when raising low-end strength.
* **A broadband denoiser cannot remove tonal hum.** Its FFT bins (23 Hz) and
  smoothed noise profile blur a 50 Hz comb into a flat floor and lower the lines
  by the same amount as the bed, so they stay 10-19 dB proud of it -- and tones
  are far more audible than the same energy as hiss. Measure the room-tone
  spectrum at ~1 Hz resolution when someone says "there's still noise".
* **Don't notch-filter hum out of speech.** High-Q notches ring for 100+ ms,
  zero-phase they ring *before* loud sounds too, and they sit on the voice's own
  harmonics. Subtract locally fitted sinusoids inside the gaps instead.
* **Look at where in time the noise is, not just how much.** Decision-directed
  denoising remembers "speech was here" for ~0.5 s, so noise right after words
  was barely reduced (SNR gain ~0 dB vs 7 dB in long pauses) until the memory
  was reset in room-tone frames.
* **A Wiener-type denoiser cannot remove noise under speech, and no setting
  changes that.** In a bin where the voice is 10 dB above the noise the optimal
  gain is ~0.9; cutting it further cuts the voice. Measured (`denoise(stats=)`):
  noise attenuation under speech is -0.1 to -0.6 dB below 10 kHz, and stays at
  about -2 dB in *exposed* bins whatever the depth (14->22 dB), over-subtraction
  (x1->x4) or estimator memory (alpha 0.98->0.80). Under words the noise is
  masked (SNR 38-55 dB per band) except in soft syllables at the top octaves
  (SNR 11 dB at 6-10 kHz, 4 dB at 10-16 kHz). What helps there is a level-keyed
  HF expander (dsp.dehiss) and a lower top end; going further needs a neural
  denoiser (resynthesis), which is a separate decision (dependencies, artifacts).
* **Place a level-keyed stage where its reference is true.** The de-hiss
  threshold is set from room-tone level; measured after denoising that level is
  ~13 dB below the hiss still under the words, so nothing was expanded. Run it
  before the denoiser.
* **Measure noise removal by injecting known noise.** Band-energy change in speech
  frames cannot see noise removal under speech (a band 15 dB above the noise barely
  moves if all the noise is removed). Add real room tone at a known level, denoise,
  and measure how much of the *added* noise survives, per frame class and band.
* **Stages that share a reference must be tested together.** The neural model beat
  the Wiener filter in isolation and lost in the chain.
* **`/tmp` is a 3.9 GB RAM-backed tmpfs here.** Keep big virtualenvs and test WAVs
  on the disk; when it filled, the harness shell itself stopped working.
* **Judge tone changes relative to the mids, not in absolute band levels.** The
  loudness stage renormalises after every change, so cutting bass adds makeup gain and
  makes absolute band levels above it *rise*; measured against 500-2 kHz the top end
  was unchanged. Also level-match any A/B clips, or the louder one wins.
* **Watch the band the problem lives in.** The bass controller watched 70-200 Hz while
  the voice's swelling 2nd harmonic sat at 200-280; widening to 70-280 cut the swell
  from 7.8 to 6.6 dB.
* **Compute a correction per-frame, not once over a whole flagged run.** The
  first version of the outlier-gap smoothing computed one average level over
  an entire flagged stretch; a run that mixed a deep dip with shallower edges
  diluted the correction the deep part actually needed (it landed ~9 dB short
  of the target). Per-frame need, amplitude-modulated onto the harvested tone,
  fixed it exactly.
* **Never rectify a signal you will analyse spectrally** (`abs(x)`).
* **Never operate on the whole file at once.** Full-length float64 temporaries
  (zero-phase filters, FFT convolution, `np.interp`, `scipy.signal.welch`,
  pyloudnorm) cost 1-3 GB each on a 20-minute file. `dsp/blocks.py` has
  chunked equivalents that match the whole-file results below -140 dBFS.
* **Verify by ear before batch-running.** Numbers cannot detect warble or dull
  EQ. Process one 60 s clip first and listen.

## Neural denoiser (optional)

Tried 2026-09-22. Two models, tested as pure denoisers and then in the full chain on
the reference narrator (noise injected at a known level to measure what survives):

| | Wiener (default) | RNNoise | DeepFilterNet3 |
|---|---|---|---|
| Noise left under the softest words, 6-10k / 10-16k | -3.4 / -7.7 dB | (not pursued) | -5.3 / -10.4 dB |
| Phrase onsets, first 60 ms | -0.2 dB | **-24.5 dB** (worst frame -36) | -0.7 dB |
| Speech frames dropped > 6 dB | 0.01% | **4.1%** | 0.3% |
| Gap floor | natural | -30..-49 dB (near-silence) | natural, but -1.4 dB below 300 Hz |

* **RNNoise is unusable for narration**: its VAD gating eats phrase onsets and tails.
* **DeepFilterNet3 preserves speech but only removes ~1-3 dB more noise under the
  softest words** than the Wiener filter, ~0.3 dB more under soft-mid words, and
  nothing measurable under loud words (where the noise is masked anyway).
* **Placement matters.** BEFORE the Wiener stage it made hiss under soft words 4.8 dB
  *worse* than the current default: the expander -> Wiener chain works because the
  Wiener stage reuses a noise profile from before the expander, so it drives the
  expander-lowered bins to its floor; a neural model in between re-normalises the
  signal and breaks that. AFTER the Wiener stage it is safe (0.1% of speech frames
  > 6 dB quieter) but adds only ~1-1.7 dB in the softest words.
* Its main real effect is deeper gaps (-89.6 dBFS with the 12 dB cap, -95 unlimited,
  vs -81 default): approaching the dead-air sound of gated audio, which ACX dislikes.
* Cost: ~+60 s and ~1.1 GB extra per 12 min file, ~1.4 GB of environment on disk.

Verdict at the time: modest measured gain (~1-2 dB under the softest words) at real
cost. Later the user listened to the AFTER-Wiener, unlimited variant and found it
"excellent" (minimal noise floor), so it is now the default when installed. Measured
numbers said modest; ears said worth it -- the by-ear judgement wins.

**Setup** (optional; the app runs fine without it, just without this stage):

```
python3 scripts/setup_neural_denoiser.py
```

Run with any Python 3.10+ interpreter -- it doesn't need to be, and normally
isn't, the one running the main app. It creates a **separate** virtualenv (not
committed to git: PyTorch is ~1 GB and platform/arch-specific, so a venv built
on this machine wouldn't run on another OS anyway) at
`~/.mastering-studio/neural-venv` by default, installs PyTorch (CPU wheels) and
`deepfilternet` there, and verifies the result. Pass `--dest PATH` to put it
somewhere else, in which case also set `$MASTERING_STUDIO_NEURAL_PYTHON` to
that venv's `python` (`python.exe` on Windows) so the app finds it; the default
location is found automatically.

`deepfilternet` needs `libdf`, a Rust extension. Where a real wheel exists for
this Python (more likely on 3.10-3.13 than on 3.14, which had none at all when
this was built), the script installs and uses it as-is. Otherwise it falls back
to `dsp/_libdf_shim.py`, a NumPy replacement for the parts used at inference
(STFT round-trip exact to -162 dBFS), which `dsp/_dfn_worker.py` only loads if
a real `libdf` isn't found -- either way this is transparent to the app.

Weights (8 MB) download to `~/.cache/DeepFilterNet` on first use. Temp files go
to `~/.cache/mastering-studio`.

## Room tone and file length

The room tone stage runs LAST, after loudness/limiting, so it adds real seconds
to the file (default +1.5 s total: 0.75 s at each end) -- expected, since ACX
explicitly wants that pad. Disable head/tail padding (`--roomtone floor-only`
or the GUI's "Smooth outlier gaps only") if you need the output to stay exactly
the same length as the input.

## Memory

Peak resident memory is ~45 bytes per sample per channel: about 2.5 GB for a
20-minute mono 48 kHz file, ~1.6 GB for 12.5 minutes (was 4.5 GB before the
chunked rewrite). `batch.plan_workers` estimates this per file, reads
`MemAvailable` from `/proc/meminfo`, and runs only as many files at once as fit
in 75% of free RAM (and never more than the CPU count or the GUI setting). On a
7 GB / 4-core machine that means long chapters run one at a time.

## Known limits

* Denoise assumes fairly stationary noise (hiss, room tone, fans). It cannot
  remove non-stationary noise (traffic, voices) and, by construction, none of
  the noise sitting in the same time-frequency bin as the voice. A neural denoiser
  (torch + DeepFilterNet / RNNoise wheels exist for Python 3.14) is the next step
  for that; it resynthesises speech, so it needs its own by-ear validation. De-hum handles only 50/60 Hz mains combs (>= 5 lines).
* Hum is left in place under speech and in gaps shorter than 150 ms (masked by
  the voice); it is removed in gaps of 150 ms or more.
* De-click deliberately misses clicks inside loud speech.
* De-breath only handles breaths that start out of a pause (inhales before a
  phrase, or mid-pause). It ignores exhales/sighs after speech, since those are
  indistinguishable from word decay tails. Tuned by spectrogram on two chapters
  of one narrator; the thresholds have not been validated by ear or on other
  voices.
* Nothing in this chain has been validated by ear by the author; the numbers
  and spectrograms say the artifacts are absent, the listener decides.
