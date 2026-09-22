"""Batch processing: up to 10 files processed concurrently across separate
worker processes (true parallelism, not just threads -- the DSP chain is
CPU-bound and partly pure-Python, so process-based parallelism is what
actually uses multiple cores).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import soundfile as sf
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from .pipeline import PipelineSettings, ProcessResult, process_file

MAX_CONCURRENT = 10

# Measured peak resident memory of the whole pipeline: ~45 bytes per sample
# (2.5 GB for a 20-minute mono 48 kHz file = 56.7 M samples), plus ~0.4 GB of
# interpreter/library baseline per worker process. Stereo scales with channels.
BYTES_PER_SAMPLE_PEAK = 48
BASELINE_GB = 0.4
RAM_FRACTION_USABLE = 0.75


def estimate_job_gb(job: "BatchJob") -> float:
    try:
        info = sf.info(job.input_path)
        gb = info.frames * info.channels * BYTES_PER_SAMPLE_PEAK / 1e9 + BASELINE_GB
    except Exception:
        gb = 1.0
    if getattr(job.settings, "enable_neural", False):
        gb += 1.2   # neural helper process (torch + model) runs while this process holds the audio
    return gb


def available_ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def plan_workers(jobs: list["BatchJob"], requested: int) -> tuple[int, str]:
    """How many files to process at once without running the machine out of RAM."""
    cap = min(requested, MAX_CONCURRENT, max(1, os.cpu_count() or 1), max(1, len(jobs)))
    avail = available_ram_gb()
    if avail is None or not jobs:
        return cap, ""
    biggest = max(estimate_job_gb(j) for j in jobs)
    fit = max(1, int(RAM_FRACTION_USABLE * avail / biggest))
    n = min(cap, fit)
    note = f"Running {n} file(s) at a time (largest needs ~{biggest:.1f} GB, {avail:.1f} GB RAM free)"
    if n < cap:
        note += f" -- limited by memory; {cap} would have fit by CPU count"
    if biggest > avail:
        note += ". WARNING: this file may exceed free memory; close other applications"
    return n, note


@dataclass
class BatchJob:
    job_id: str
    input_path: str
    output_path: str
    settings: PipelineSettings = field(default_factory=PipelineSettings)


def _worker(job: BatchJob, progress_queue) -> tuple[str, Optional[ProcessResult], Optional[str]]:
    try:
        def on_stage(name: str, pct: float) -> None:
            progress_queue.put(("progress", job.job_id, name, pct))

        result = process_file(job.input_path, job.output_path, job.settings, on_stage=on_stage)
        progress_queue.put(("done", job.job_id, "Done", 1.0))
        return job.job_id, result, None
    except Exception:
        err = traceback.format_exc()
        progress_queue.put(("error", job.job_id, err, 0.0))
        return job.job_id, None, err


def run_batch(
    jobs: list[BatchJob],
    on_event: Callable[[str, str, str, float], None],
    max_workers: Optional[int] = None,
) -> dict[str, tuple[Optional[ProcessResult], Optional[str]]]:
    """Runs jobs (up to MAX_CONCURRENT at a time) and streams progress via
    on_event(kind, job_id, message, fraction). Blocks until all jobs finish;
    intended to be called from a background QThread, not the GUI thread.
    """
    max_workers, note = plan_workers(jobs, max_workers or MAX_CONCURRENT)
    if note:
        on_event("info", "", note, 0.0)
    manager = mp.Manager()
    progress_queue = manager.Queue()
    results: dict[str, tuple[Optional[ProcessResult], Optional[str]]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, job, progress_queue) for job in jobs]
        pending = len(futures)
        while pending > 0:
            # Drain progress events without blocking job completion.
            try:
                while True:
                    kind, job_id, msg, pct = progress_queue.get(timeout=0.1)
                    on_event(kind, job_id, msg, pct)
            except Exception:
                pass
            for f in list(futures):
                if f.done():
                    futures.remove(f)
                    pending -= 1
                    job_id, result, err = f.result()
                    results[job_id] = (result, err)

    return results
