#!/usr/bin/env python3
"""Sets up the optional neural denoiser (DeepFilterNet3) in its own virtualenv,
kept separate from the app's own dependencies -- see mastering_studio/dsp/neural.py
for why: PyTorch is ~1 GB and platform/arch-specific, so it can't be committed to
git or shared into the app's own portable .venv.

Usage:
    python3 scripts/setup_neural_denoiser.py [--dest PATH] [--force]

Run with any Python 3.10+ interpreter -- it does not need to be, and normally
won't be, the one running the main app. Creates a venv at PATH (default: the
same place mastering_studio.dsp.neural looks for it automatically,
~/.mastering-studio/neural-venv), installs PyTorch (CPU wheels), deepfilternet,
and its remaining runtime dependencies, then verifies the result the same way
the app does before using it.

If you use --dest to put it somewhere else, also set the app's
$MASTERING_STUDIO_NEURAL_PYTHON environment variable to that venv's python (or
python.exe on Windows) so it's found.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv


def default_dest() -> str:
    return os.path.expanduser(os.path.join("~", ".mastering-studio", "neural-venv"))


def venv_python(dest: str) -> str:
    # Matches mastering_studio.dsp.neural._venv_python_path -- keep in sync.
    if sys.platform == "win32":
        return os.path.join(dest, "Scripts", "python.exe")
    return os.path.join(dest, "bin", "python")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, **kw)


def pip_install(py: str, *args: str) -> bool:
    return run([py, "-m", "pip", "install", "-q", *args]).returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=default_dest(), help="where to create the neural denoiser's virtualenv")
    p.add_argument("--force", action="store_true", help="recreate the venv even if one already exists there")
    args = p.parse_args()

    dest = os.path.abspath(os.path.expanduser(args.dest))
    py = venv_python(dest)

    if os.path.isfile(py) and not args.force:
        print(f"A venv already exists at {dest} (pass --force to recreate it). Just verifying it still works...")
    else:
        print(f"Creating virtualenv at {dest} ...")
        venv.EnvBuilder(with_pip=True, clear=args.force).create(dest)

        run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"])

        print("\nInstalling PyTorch (CPU)...")
        if sys.platform == "darwin":
            ok = pip_install(py, "torch", "torchaudio")  # PyPI's Mac wheels are CPU/MPS already; no special index needed
        else:
            ok = pip_install(py, "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu")
        if not ok:
            print("torch/torchaudio install FAILED.", file=sys.stderr)
            return 1

        print("\nInstalling deepfilternet's other runtime dependencies...")
        if not pip_install(py, "numpy", "scipy", "soundfile", "loguru"):
            print("dependency install FAILED.", file=sys.stderr)
            return 1

        print("\nInstalling deepfilternet (trying a normal install first, in case a real")
        print("libdf wheel exists for this Python/platform)...")
        if not pip_install(py, "deepfilternet"):
            print("  ...no full install available; falling back to --no-deps")
            print("  (mastering_studio ships a pure-Python stand-in for libdf for this case).")
            if not pip_install(py, "--no-deps", "deepfilternet"):
                print("deepfilternet install FAILED.", file=sys.stderr)
                return 1

    print("\nVerifying...")
    check = (
        "import torch, soundfile, scipy; import importlib.util as u; "
        "assert u.find_spec('df'); "
        "print('OK, libdf is', 'the real Rust extension' if u.find_spec('libdf') else 'the bundled NumPy stand-in')"
    )
    r = run([py, "-c", check], capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        print("Verification FAILED.", file=sys.stderr)
        return 1

    print(f"\nDone. The app will use this automatically (checked at: {py}).")
    if dest != os.path.abspath(default_dest()):
        print(f"This isn't the default location, so also set:\n  export MASTERING_STUDIO_NEURAL_PYTHON={py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
