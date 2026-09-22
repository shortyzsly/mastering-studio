"""Command-line entry point for batch processing without the GUI."""
from __future__ import annotations

import argparse
import os
import sys

from .batch import BatchJob, run_batch
from .pipeline import PipelineSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch master audiobook/voice recordings.")
    parser.add_argument("inputs", nargs="+", help="Input audio files")
    parser.add_argument("-o", "--output-dir", help="Output directory (default: next to each input file)")
    parser.add_argument("--denoise", choices=["auto", "gentle", "moderate", "aggressive", "off"], default="auto")
    parser.add_argument("--target-lufs", type=float, default=-21.5, help="integrated loudness: -21.5 ACX (default), -23 audiobook, -16 podcast")
    parser.add_argument("--peak", type=float, default=-3.0, help="true-peak ceiling in dBTP")
    parser.add_argument("--neural", choices=["auto", "off", "on"], default="auto", help="DeepFilterNet3 cleanup pass: auto = on if the neural env is installed")
    parser.add_argument("--dehum", choices=["auto", "off"], default="auto", help="remove 50/60 Hz mains hum/buzz")
    parser.add_argument("--lowend", choices=["off", "light", "medium", "strong"], default="medium", help="mud cut + bass tightening")
    parser.add_argument("--soften", choices=["off", "light", "medium", "strong"], default="medium", help="harshness control (top-end curve + 2.2-4.4 kHz presence)")
    parser.add_argument("--dehiss", choices=["off", "light", "medium", "strong"], default="medium", help="HF hiss under soft words")
    parser.add_argument("--deess", choices=["off", "light", "medium", "strong"], default="medium")
    parser.add_argument("--debreath", choices=["off", "light", "medium", "strong"], default="medium")
    parser.add_argument("--roomtone", choices=["on", "pad-only", "floor-only", "off"], default="on",
                        help="ACX head/tail room tone pad + outlier gap smoothing")
    parser.add_argument("--roomtone-pad", type=float, default=0.75, help="seconds of room tone at head/tail (ACX wants 0.5-1)")
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()

    settings = PipelineSettings(target_lufs=args.target_lufs, peak_ceiling_dbfs=args.peak)
    if args.denoise == "off":
        settings.enable_denoise = False
    elif args.denoise != "auto":
        settings.denoise_strength = args.denoise

    if args.neural != "auto":
        settings.enable_neural = args.neural == "on"
    settings.enable_dehum = args.dehum != "off"
    settings.enable_lowend = args.lowend != "off"
    if settings.enable_lowend:
        settings.lowend_strength = args.lowend
    settings.enable_soften = args.soften != "off"
    if settings.enable_soften:
        settings.soften_strength = args.soften
    settings.enable_dehiss = args.dehiss != "off"
    if settings.enable_dehiss:
        settings.dehiss_strength = args.dehiss
    settings.enable_deess = args.deess != "off"
    if settings.enable_deess:
        settings.deess_strength = args.deess
    settings.enable_debreath = args.debreath != "off"
    if settings.enable_debreath:
        settings.debreath_strength = args.debreath
    settings.enable_roomtone = args.roomtone in ("on", "pad-only")
    settings.enable_roomtone_floor = args.roomtone in ("on", "floor-only")
    settings.roomtone_head_s = settings.roomtone_tail_s = args.roomtone_pad

    jobs = []
    for path in args.inputs:
        base, ext = os.path.splitext(os.path.basename(path))
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(path))
        out_path = os.path.join(out_dir, f"{base}_mastered{ext}")
        jobs.append(BatchJob(job_id=path, input_path=path, output_path=out_path, settings=settings))

    def on_event(kind: str, job_id: str, msg: str, pct: float) -> None:
        print(f"[{pct:5.0%}] {os.path.basename(job_id)}: {msg}")

    results = run_batch(jobs, on_event, max_workers=args.max_workers)
    failures = 0
    for job_id, (result, err) in results.items():
        if err:
            failures += 1
            print(f"FAILED: {job_id}\n{err}", file=sys.stderr)
        else:
            print(f"OK: {result.output_path}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
