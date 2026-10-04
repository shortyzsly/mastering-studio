from __future__ import annotations

import html
import os
import sys
import uuid
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QDragEnterEvent, QDropEvent, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import acx
from .auto import MIC_PROFILES
from .batch import MAX_CONCURRENT, BatchJob, run_batch
from .dsp import activity as activity_mod, neural as neural_mod
from .dsp.eq import EQ_PRESETS
from .io_utils import SUPPORTED_EXTENSIONS, load_audio, to_mono
from .pipeline import PipelineSettings, diagnose

COL_FILE, COL_STATUS, COL_PROGRESS, COL_ACX, COL_NOTES = range(5)

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "MASTERING STUDIO OUTPUT")

# Widget attribute names (all QComboBox) whose selection is remembered across launches.
PERSISTED_COMBOS = [
    "mic_combo", "auto_combo", "roomfix_combo", "dereverb_combo", "eq_low_combo", "eq_mid_combo", "eq_high_combo",
    "export_combo", "declick_strength_combo",
    "dehum_combo", "dehum_under_speech_combo", "neural_combo", "neural_atten_combo", "denoise_combo",
    "eq_combo", "lowend_combo", "declick_combo", "soften_combo", "dehiss_combo",
    "deess_combo", "debreath_combo", "compress_combo", "loudness_combo", "roomtone_combo",
]
MIN_NOISE_SELECTION_S = 0.2


class BatchWorker(QThread):
    progress = Signal(str, str, float)  # job_id, stage/message, fraction
    job_done = Signal(str, object, object)  # job_id, ProcessResult|None, error|None
    all_done = Signal()
    info = Signal(str)  # planner / status text for the log

    def __init__(self, jobs: list[BatchJob], max_workers: int):
        super().__init__()
        self.jobs = jobs
        self.max_workers = max_workers

    def run(self) -> None:
        def on_event(kind: str, job_id: str, msg: str, pct: float) -> None:
            if kind in ("progress", "done", "error"):
                self.progress.emit(job_id, msg, pct)
            elif kind == "info":
                self.info.emit(msg)

        results = run_batch(self.jobs, on_event, max_workers=self.max_workers)
        for job in self.jobs:
            result, err = results.get(job.job_id, (None, "No result"))
            self.job_done.emit(job.job_id, result, err)
        self.all_done.emit()


class FnWorker(QThread):
    """Runs fn() off the GUI thread; emits (result, error)."""
    finished_with = Signal(object, object)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            self.finished_with.emit(self.fn(), None)
        except Exception as e:   # shown to the user, never swallowed
            self.finished_with.emit(None, e)


def load_waveform(path: str, buckets: int = 4000) -> dict:
    """Min/max envelope for drawing, plus the auto-detected room-tone regions (seconds)."""
    data, sr = load_audio(path)
    mono = to_mono(data)
    n = len(mono)
    per = max(1, n // buckets)
    m = mono[: (n // per) * per].reshape(-1, per)
    act = activity_mod.analyze_activity(mono, sr)
    regions = [(a / sr, b / sr) for a, b in activity_mod.frame_ranges(act.noise_mask, act.hop, act.frame, n)]
    return {"mins": m.min(axis=1), "maxs": m.max(axis=1), "duration": n / sr, "regions": regions}


class Waveform(QWidget):
    """Waveform with click-to-seek and drag-to-select. Shades the auto-detected room tone (green),
    the user's noise sample (orange) and the current selection (blue)."""
    seek = Signal(float)
    selected = Signal(float, float)

    def __init__(self, title: str, selectable: bool = True):
        super().__init__()
        self.title = title
        self.selectable = selectable
        self.wave: dict | None = None
        self.selection: tuple[float, float] | None = None
        self.noise_region: tuple[float, float] | None = None
        self.playhead: float | None = None
        self._drag_from: float | None = None
        self.setMinimumHeight(90)

    def set_wave(self, wave: dict | None) -> None:
        self.wave, self.selection, self.playhead = wave, None, None
        self.update()

    def _t(self, x: float) -> float:
        d = self.wave["duration"] if self.wave else 0.0
        return float(np.clip(x / max(self.width(), 1), 0, 1)) * d

    def _x(self, t: float) -> float:
        d = self.wave["duration"] if self.wave else 1.0
        return t / max(d, 1e-9) * self.width()

    def mousePressEvent(self, e) -> None:
        if self.wave:
            self._drag_from = self._t(e.position().x())

    def mouseMoveEvent(self, e) -> None:
        if self.wave and self._drag_from is not None and self.selectable:
            t = self._t(e.position().x())
            self.selection = (min(self._drag_from, t), max(self._drag_from, t))
            self.update()

    def mouseReleaseEvent(self, e) -> None:
        if not self.wave or self._drag_from is None:
            return
        t = self._t(e.position().x())
        if abs(self._x(t) - self._x(self._drag_from)) < 4 or not self.selectable:
            self.selection = None
            self.seek.emit(t)
        else:
            self.selected.emit(*self.selection)
        self._drag_from = None
        self.update()

    def paintEvent(self, _) -> None:
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(24, 26, 30))
        if not self.wave:
            p.setPen(QColor(140, 140, 140))
            p.drawText(self.rect(), Qt.AlignCenter, f"{self.title}: select a file in the queue")
            return
        for a, b in self.wave.get("regions", []):
            p.fillRect(int(self._x(a)), 0, max(1, int(self._x(b) - self._x(a))), h, QColor(40, 90, 50, 110))
        for reg, col in ((self.noise_region, QColor(230, 140, 30, 110)), (self.selection, QColor(70, 130, 230, 110))):
            if reg:
                p.fillRect(int(self._x(reg[0])), 0, max(1, int(self._x(reg[1]) - self._x(reg[0]))), h, col)
        mins, maxs = self.wave["mins"], self.wave["maxs"]
        idx = np.linspace(0, len(mins), w + 1).astype(int)
        p.setPen(QPen(QColor(120, 200, 255)))
        mid = h / 2
        for x in range(w):
            a, b = idx[x], max(idx[x + 1], idx[x] + 1)
            lo, hi = float(mins[a:b].min()), float(maxs[a:b].max())
            p.drawLine(x, int(mid - hi * mid), x, int(mid - lo * mid))
        if self.playhead is not None:
            p.setPen(QPen(QColor(255, 80, 80), 2))
            x = int(self._x(self.playhead))
            p.drawLine(x, 0, x, h)
        p.setPen(QColor(200, 200, 200))
        p.drawText(6, 14, self.title)


class DropTableWidget(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self, rows=0, cols=0, parent=None):
        super().__init__(rows, cols, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


def _combo(items: list[str], tip: str = "") -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    if tip:
        c.setToolTip(tip)
    return c


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mastering Studio -- Automated Audiobook Mastering (ACX)")
        self.resize(1320, 860)

        self.jobs: dict[str, BatchJob] = {}
        self.row_for_job: dict[str, int] = {}
        self.results: dict[str, object] = {}
        self.previews: dict[str, tuple] = {}
        self.noise_regions: dict[str, tuple[float, float]] = {}
        self.worker: BatchWorker | None = None
        self.bg: list[QThread] = []   # keep background workers alive until they finish
        self.output_dir: str | None = None
        self.current_job: str | None = None
        self.settings_store = QSettings("MasteringStudio", "MasteringStudio")

        self.players = {}
        for key in ("orig", "master"):
            player, out = QMediaPlayer(self), QAudioOutput(self)
            player.setAudioOutput(out)
            player.positionChanged.connect(lambda ms, k=key: self.on_position(k, ms))
            self.players[key] = (player, out)
        self.active_player: str | None = None
        self.play_end: float | None = None

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addWidget(self._build_toolbar())

        left = QSplitter(Qt.Vertical)
        left.addWidget(self._build_queue_panel())
        left.addWidget(self._build_editor_panel())
        left.addWidget(self._build_report_panel())
        left.setStretchFactor(0, 2)
        left.setStretchFactor(1, 3)
        left.setStretchFactor(2, 2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._build_settings_panel())
        scroll.setMinimumWidth(430)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

        self._load_settings()
        for name in PERSISTED_COMBOS:
            getattr(self, name).currentIndexChanged.connect(self._save_settings)
        self.concurrency_spin.valueChanged.connect(self._save_settings)
        self.auto_combo.currentIndexChanged.connect(self._sync_auto_mode)
        self._sync_auto_mode()

    # -- persisted settings --------------------------------------------------

    def _load_settings(self) -> None:
        s = self.settings_store
        for name in PERSISTED_COMBOS:
            combo: QComboBox = getattr(self, name)
            saved = s.value(f"combo/{name}", None)
            if saved is not None:
                idx = combo.findText(saved)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        self.concurrency_spin.setValue(int(s.value("concurrency", self.concurrency_spin.value())))
        self.output_dir = s.value("output_dir", DEFAULT_OUTPUT_DIR)
        self.output_label.setText(self.output_dir)

    def _save_settings(self) -> None:
        s = self.settings_store
        for name in PERSISTED_COMBOS:
            s.setValue(f"combo/{name}", getattr(self, name).currentText())
        s.setValue("concurrency", self.concurrency_spin.value())
        s.setValue("output_dir", self.output_dir)

    def _sync_auto_mode(self) -> None:
        self.manual_box.setEnabled(not self.auto_combo.currentText().startswith("On"))

    # -- UI construction -----------------------------------------------------

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        for label, slot in (("Add Files...", self.on_add_files_clicked), ("Remove Selected", self.on_remove_selected),
                            ("Clear Queue", self.on_clear_queue)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            layout.addWidget(b)
        self.analyze_btn = QPushButton("Analyze Selected")
        self.analyze_btn.setToolTip("Preview what Smart Auto finds and decides for this file, and its raw ACX check, without processing")
        self.analyze_btn.clicked.connect(self.on_analyze_clicked)
        layout.addWidget(self.analyze_btn)
        layout.addStretch(1)
        out_btn = QPushButton("Output Folder...")
        out_btn.clicked.connect(self.on_choose_output_dir)
        layout.addWidget(out_btn)
        self.output_label = QLabel(DEFAULT_OUTPUT_DIR)
        layout.addWidget(self.output_label)
        layout.addStretch(1)
        self.process_btn = QPushButton("Master Queue")
        self.process_btn.setStyleSheet("font-weight: bold; padding: 4px 14px;")
        self.process_btn.clicked.connect(self.on_process_clicked)
        layout.addWidget(self.process_btn)
        return bar

    def _build_queue_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Drag and drop audio files here, or use Add Files. Select a row to see, play and mark its audio.")
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)
        self.table = DropTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Progress", "ACX", "Notes"])
        self.table.horizontalHeader().setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(COL_NOTES, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.files_dropped.connect(self.add_files)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        layout.addWidget(self.table)
        return panel

    def _build_editor_panel(self) -> QWidget:
        box = QGroupBox("Audio")
        layout = QVBoxLayout(box)
        self.wave_orig = Waveform("Original  (drag = select, click = seek; green = auto-detected room tone)")
        self.wave_master = Waveform("Mastered", selectable=False)
        self.wave_orig.seek.connect(lambda t: self.seek("orig", t))
        self.wave_master.seek.connect(lambda t: self.seek("master", t))
        self.wave_orig.selected.connect(self.on_selection)
        layout.addWidget(self.wave_orig, stretch=1)
        layout.addWidget(self.wave_master, stretch=1)

        row = QHBoxLayout()
        self.play_orig_btn = QPushButton("Play Original")
        self.play_master_btn = QPushButton("Play Mastered")
        self.ab_btn = QPushButton("A/B Switch")
        self.ab_btn.setToolTip("Switch between original and mastered at the same moment of the performance")
        stop_btn = QPushButton("Stop")
        self.play_orig_btn.clicked.connect(lambda: self.play("orig"))
        self.play_master_btn.clicked.connect(lambda: self.play("master"))
        self.ab_btn.clicked.connect(self.on_ab)
        stop_btn.clicked.connect(self.stop)
        for b in (self.play_orig_btn, self.play_master_btn, self.ab_btn, stop_btn):
            row.addWidget(b)
        row.addSpacing(20)
        self.noise_btn = QPushButton("Use Selection as Noise Sample")
        self.noise_btn.setToolTip("Mark a stretch of pure room tone (no breaths, no words). Spectral denoise learns its noise profile\n"
                                  "from exactly this instead of the auto-detected room tone.")
        self.noise_btn.clicked.connect(self.on_use_noise_sample)
        clear_noise_btn = QPushButton("Auto Noise Sample")
        clear_noise_btn.clicked.connect(self.on_clear_noise_sample)
        row.addWidget(self.noise_btn)
        row.addWidget(clear_noise_btn)
        row.addStretch(1)
        self.time_label = QLabel("")
        row.addWidget(self.time_label)
        layout.addLayout(row)
        return box

    def _build_report_panel(self) -> QWidget:
        tabs = QTabWidget()
        self.report = QTextBrowser()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        tabs.addTab(self.report, "Report (selected file)")
        tabs.addTab(self.log, "Log")
        return tabs

    def _build_settings_panel(self) -> QWidget:
        panel = QWidget()
        outer = QVBoxLayout(panel)

        top = QGroupBox("Automation")
        form = QFormLayout(top)
        self.mic_combo = _combo([p["label"] for p in MIC_PROFILES.values()],
                                "Sets the starting point for this mic type: high-pass, bass/boom control,\n"
                                "harshness and de-ess floors, room sensitivity, how much top end may be lifted.")
        form.addRow("Microphone:", self.mic_combo)
        self.auto_combo = _combo(["On -- analyze each file and decide (recommended)", "Off -- use my stage settings"],
                                 "Smart Auto measures each file (tone vs a finished-narration target, noise, clicks,\n"
                                 "room decay) and picks every stage's strength. Click 'Analyze Selected' to preview.")
        form.addRow("Smart Auto:", self.auto_combo)
        self.roomfix_combo = _combo(["On (recommended)", "Off"],
                                    "Small-booth fix: measured low-end/mud tone match, narrow notches on room modes\n"
                                    "(needs ~40 s of speech to tell them from pitch), and de-reverb for reflections.")
        form.addRow("Room && mud fix:", self.roomfix_combo)
        self.dereverb_combo = _combo(["Auto (from measured room decay)", "Off", "Light", "Medium", "Strong"])
        form.addRow("De-reverb (reflections):", self.dereverb_combo)
        self.loudness_combo = _combo([
            "ACX submission (-21.5 LUFS, RMS checked -23..-18, -3 dBTP)",
            "Audiobook (-23 LUFS, -3 dBTP; not ACX-checked)",
            "Podcast (-16 LUFS, -1 dBTP)",
            "Off",
        ])
        form.addRow("Loudness target:", self.loudness_combo)
        self.export_combo = _combo(["WAV (24-bit) + ACX MP3 (192 kbps, 44.1 kHz)", "WAV only"])
        form.addRow("Export:", self.export_combo)
        self.roomtone_combo = _combo([
            "On (exact 0.75 s head / 2 s tail + smooth outlier gaps)",
            "Head/tail only",
            "Smooth outlier gaps only",
            "Off",
        ])
        form.addRow("Room tone (ACX):", self.roomtone_combo)
        self.neural_combo = _combo(["On (DeepFilterNet3)", "Off"])
        if not neural_mod.installed():
            self.neural_combo.setEnabled(False)
            self.neural_combo.setToolTip("Neural environment not found. Run scripts/setup_neural_denoiser.py (see README).")
        form.addRow("Neural denoiser:", self.neural_combo)
        outer.addWidget(top)

        eqbox = QGroupBox("EQ presets (on top of the automatic EQ)")
        form = QFormLayout(eqbox)
        self.eq_low_combo = _combo([k.capitalize() if k != "off" else "Off" for k in EQ_PRESETS["low"]])
        self.eq_mid_combo = _combo([k.capitalize() if k != "off" else "Off" for k in EQ_PRESETS["mid"]])
        self.eq_high_combo = _combo([k.capitalize() if k != "off" else "Off" for k in EQ_PRESETS["high"]])
        form.addRow("Low end:", self.eq_low_combo)
        form.addRow("Mids:", self.eq_mid_combo)
        form.addRow("High end:", self.eq_high_combo)
        outer.addWidget(eqbox)

        self.manual_box = QGroupBox("Stage settings (used when Smart Auto is off)")
        form = QFormLayout(self.manual_box)
        self.dehum_combo = _combo(["Auto (recommended)", "Off"])
        form.addRow("De-hum (50/60 Hz buzz):", self.dehum_combo)
        self.dehum_under_speech_combo = _combo(["Off (default)", "On (experimental)"],
                                               "Narrow-notch detected hum harmonics under speech too, not just in pauses.")
        form.addRow("De-hum under speech:", self.dehum_under_speech_combo)
        self.neural_atten_combo = _combo(["Limited (12 dB, gentler -- recommended)", "Unlimited (deepest cleanup)"])
        form.addRow("Neural attenuation:", self.neural_atten_combo)
        self.denoise_combo = _combo(["Auto (recommended)", "Gentle", "Moderate", "Aggressive", "Maximum", "Off"])
        form.addRow("Spectral denoise:", self.denoise_combo)
        self.eq_combo = _combo(["Auto (tone match + corrective cuts)", "Half strength", "Off"])
        form.addRow("Corrective EQ:", self.eq_combo)
        self.lowend_combo = _combo(["Medium (recommended)", "Light", "Strong", "Max (small/untreated room)", "Off"])
        form.addRow("Low end (mud && bass):", self.lowend_combo)
        self.declick_combo = _combo(["On (recommended)", "Off"])
        form.addRow("Mouth de-click:", self.declick_combo)
        self.declick_strength_combo = _combo(["Normal", "Strong (more lip smacks in pauses)"])
        form.addRow("De-click sensitivity:", self.declick_strength_combo)
        self.soften_combo = _combo(["Strong (recommended)", "Medium", "Light", "Off"])
        form.addRow("Harshness (top end):", self.soften_combo)
        self.dehiss_combo = _combo(["Medium (recommended)", "Light", "Strong", "Off"])
        form.addRow("Hiss under soft words:", self.dehiss_combo)
        self.deess_combo = _combo(["Medium (recommended)", "Light", "Strong", "Off"])
        form.addRow("De-esser (sibilance):", self.deess_combo)
        self.debreath_combo = _combo(["Medium (-10 dB)", "Light (-6 dB)", "Strong (-14 dB)", "Off"])
        form.addRow("De-breath:", self.debreath_combo)
        self.compress_combo = _combo(["Auto (recommended)", "Off"])
        form.addRow("Compression:", self.compress_combo)
        outer.addWidget(self.manual_box)

        misc = QGroupBox("Batch")
        form = QFormLayout(misc)
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, MAX_CONCURRENT)
        self.concurrency_spin.setValue(min(4, MAX_CONCURRENT))
        form.addRow("Max files at once:", self.concurrency_spin)
        note = QLabel("Listen to the first finished chapter before batch-running a whole book: "
                      "numbers can't hear warble or a dull top end.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        outer.addWidget(misc)
        outer.addStretch(1)
        return panel

    # -- queue management ----------------------------------------------------

    def on_add_files_clicked(self) -> None:
        exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(self, "Select audio files", "", f"Audio Files ({exts})")
        if paths:
            self.add_files(paths)

    def add_files(self, paths: list[str]) -> None:
        added = 0
        for path in paths:
            if not os.path.isfile(path):
                continue
            if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
                self.log_msg(f"Skipping unsupported file type: {path}")
                continue
            job_id = str(uuid.uuid4())
            self.jobs[job_id] = BatchJob(job_id=job_id, input_path=path, output_path="")
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, COL_FILE, QTableWidgetItem(os.path.basename(path)))
            self.table.item(row, COL_FILE).setData(Qt.UserRole, job_id)
            self.table.setItem(row, COL_STATUS, QTableWidgetItem("Queued"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            self.table.setCellWidget(row, COL_PROGRESS, bar)
            self.table.setItem(row, COL_ACX, QTableWidgetItem(""))
            self.table.setItem(row, COL_NOTES, QTableWidgetItem(""))
            added += 1
        self._reindex_rows()
        if added:
            self.log_msg(f"Added {added} file(s) to queue.")
            if self.current_job is None:
                self.table.selectRow(self.table.rowCount() - added)

    def on_remove_selected(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            job_id = self.table.item(row, COL_FILE).data(Qt.UserRole)
            for d in (self.jobs, self.results, self.previews, self.noise_regions):
                d.pop(job_id, None)
            self.table.removeRow(row)
        self._reindex_rows()

    def on_clear_queue(self) -> None:
        self.stop()
        self.table.setRowCount(0)
        for d in (self.jobs, self.row_for_job, self.results, self.previews, self.noise_regions):
            d.clear()
        self.current_job = None
        self.wave_orig.set_wave(None)
        self.wave_master.set_wave(None)
        self.report.clear()

    def on_choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if directory:
            self.output_dir = directory
            self.output_label.setText(directory)
            self._save_settings()

    def _reindex_rows(self) -> None:
        self.row_for_job = {self.table.item(r, COL_FILE).data(Qt.UserRole): r for r in range(self.table.rowCount())}

    # -- waveform, selection, playback ------------------------------------------

    def on_row_selected(self) -> None:
        rows = {i.row() for i in self.table.selectedIndexes()}
        if len(rows) != 1:
            return
        job_id = self.table.item(rows.pop(), COL_FILE).data(Qt.UserRole)
        if job_id == self.current_job:
            return
        self.stop()
        self.current_job = job_id
        job = self.jobs[job_id]
        self.wave_orig.set_wave(None)
        self.wave_orig.noise_region = self.noise_regions.get(job_id)
        self.players["orig"][0].setSource(QUrl.fromLocalFile(job.input_path))
        self._run_bg(lambda p=job.input_path: load_waveform(p), lambda w, err, j=job_id: self._show_wave(self.wave_orig, j, w, err))
        self._load_master_view(job_id)
        self._render_report(job_id)

    def _load_master_view(self, job_id: str) -> None:
        self.wave_master.set_wave(None)
        result = self.results.get(job_id)
        if result:
            self.players["master"][0].setSource(QUrl.fromLocalFile(result.output_path))
            self._run_bg(lambda p=result.output_path: load_waveform(p),
                         lambda w, err, j=job_id: self._show_wave(self.wave_master, j, w, err))
        else:
            self.players["master"][0].setSource(QUrl())

    def _show_wave(self, widget: Waveform, job_id: str, wave, err) -> None:
        if err:
            self.log_msg(f"Could not load waveform: {err}")
        elif job_id == self.current_job:
            widget.set_wave(wave)

    def _run_bg(self, fn, done) -> None:
        w = FnWorker(fn)
        w.finished_with.connect(done)
        w.finished.connect(lambda w=w: self.bg.remove(w) if w in self.bg else None)
        self.bg.append(w)
        w.start()

    def on_selection(self, t0: float, t1: float) -> None:
        self.time_label.setText(f"Selection {t0:.2f}-{t1:.2f} s ({t1 - t0:.2f} s)")

    def on_use_noise_sample(self) -> None:
        sel = self.wave_orig.selection
        if not self.current_job or not sel:
            QMessageBox.information(self, "No selection", "Drag across a stretch of pure room tone in the Original waveform first.")
            return
        if sel[1] - sel[0] < MIN_NOISE_SELECTION_S:
            QMessageBox.information(self, "Too short", f"Select at least {MIN_NOISE_SELECTION_S} s of room tone.")
            return
        self.noise_regions[self.current_job] = sel
        self.wave_orig.noise_region = sel
        self.wave_orig.update()
        self.log_msg(f"{os.path.basename(self.jobs[self.current_job].input_path)}: noise sample {sel[0]:.2f}-{sel[1]:.2f} s")

    def on_clear_noise_sample(self) -> None:
        if self.current_job:
            self.noise_regions.pop(self.current_job, None)
            self.wave_orig.noise_region = None
            self.wave_orig.update()

    def _shift(self) -> float:
        r = self.results.get(self.current_job)
        return float(r.settings_used.get("roomtone_shift_s", 0.0)) if r else 0.0

    def play(self, key: str, at: float | None = None) -> None:
        if key == "master" and self.current_job not in self.results:
            QMessageBox.information(self, "Not mastered yet", "Master this file first (Master Queue).")
            return
        self.stop()
        player = self.players[key][0]
        sel = self.wave_orig.selection
        if at is None and sel:   # play the selection (mapped onto the master's timeline)
            at = sel[0] + (self._shift() if key == "master" else 0.0)
            self.play_end = sel[1] + (self._shift() if key == "master" else 0.0)
        if at is not None:
            player.setPosition(int(at * 1000))
        self.active_player = key
        player.play()

    def stop(self) -> None:
        for player, _ in self.players.values():
            player.pause()
        self.active_player, self.play_end = None, None

    def seek(self, key: str, t: float) -> None:
        self.play(key, at=t)

    def on_ab(self) -> None:
        if self.active_player is None:
            return
        cur = self.players[self.active_player][0].position() / 1000
        if self.active_player == "orig":
            self.play("master", at=cur + self._shift())
        else:
            self.play("orig", at=max(0.0, cur - self._shift()))

    def on_position(self, key: str, ms: int) -> None:
        if key != self.active_player:
            return
        t = ms / 1000
        if self.play_end is not None and t >= self.play_end:
            self.stop()
        orig_t = t if key == "orig" else t - self._shift()
        self.wave_orig.playhead = orig_t
        self.wave_master.playhead = t if key == "master" else t + self._shift()
        self.wave_orig.update()
        self.wave_master.update()
        self.time_label.setText(f"{'Original' if key == 'orig' else 'Mastered'}  {int(t // 60)}:{t % 60:05.2f}")

    # -- settings -> PipelineSettings ------------------------------------------

    @staticmethod
    def _choice(combo: QComboBox) -> str:
        return combo.currentText().split(" (")[0].split(" --")[0].strip().lower()

    def _current_settings(self) -> PipelineSettings:
        s = PipelineSettings()
        s.mic = list(MIC_PROFILES)[self.mic_combo.currentIndex()]
        s.smart_auto = self.auto_combo.currentText().startswith("On")
        s.enable_roomfix = self.roomfix_combo.currentText().startswith("On")
        s.dereverb_strength = self._choice(self.dereverb_combo)
        s.eq_low, s.eq_mid, s.eq_high = (list(EQ_PRESETS[band])[c.currentIndex()] for band, c in
                                         (("low", self.eq_low_combo), ("mid", self.eq_mid_combo), ("high", self.eq_high_combo)))
        s.export_mp3 = "MP3" in self.export_combo.currentText()
        s.enable_neural = self.neural_combo.currentText().startswith("On") and neural_mod.installed()

        rt = self.roomtone_combo.currentText()
        if rt == "Off":
            s.enable_roomtone = s.enable_roomtone_floor = False
        elif rt.startswith("Head"):
            s.enable_roomtone_floor = False
        elif rt.startswith("Smooth"):
            s.enable_roomtone = False

        loud = self.loudness_combo.currentText()
        s.acx_enforce = loud.startswith("ACX")
        if loud.startswith("Off"):
            s.enable_loudness = s.enable_limiter = False
        elif loud.startswith("Podcast"):
            s.target_lufs, s.peak_ceiling_dbfs = -16.0, -1.0
        elif loud.startswith("Audiobook"):
            s.target_lufs, s.peak_ceiling_dbfs = -23.0, -3.0
        else:
            s.target_lufs, s.peak_ceiling_dbfs = -21.5, -3.0

        if s.smart_auto:
            return s   # strengths are decided per file; stage on/off defaults stay on

        s.enable_dehum = self.dehum_combo.currentText() != "Off"
        s.enable_dehum_under_speech = self.dehum_under_speech_combo.currentText().startswith("On")
        s.neural_atten_limit_db = None if self.neural_atten_combo.currentText().startswith("Unlimited") else 12.0
        d = self._choice(self.denoise_combo)
        if d == "off":
            s.enable_denoise = False
        elif d != "auto":
            s.denoise_strength = d
        e = self.eq_combo.currentText()
        if e == "Off":
            s.enable_eq = False
        elif e.startswith("Half"):
            s.eq_amount = 0.5
        for attr_on, attr_strength, combo in (
            ("enable_lowend", "lowend_strength", self.lowend_combo),
            ("enable_soften", "soften_strength", self.soften_combo),
            ("enable_dehiss", "dehiss_strength", self.dehiss_combo),
            ("enable_deess", "deess_strength", self.deess_combo),
            ("enable_debreath", "debreath_strength", self.debreath_combo),
        ):
            choice = combo.currentText().split()[0].lower()
            setattr(s, attr_on, choice != "off")
            if choice != "off":
                setattr(s, attr_strength, choice)
        s.enable_declick = self.declick_combo.currentText() != "Off"
        s.declick_strength = "strong" if self.declick_strength_combo.currentText().startswith("Strong") else "normal"
        s.enable_compress = self.compress_combo.currentText() != "Off"
        return s

    # -- analysis preview ------------------------------------------------------------

    def on_analyze_clicked(self) -> None:
        if not self.current_job:
            QMessageBox.information(self, "Nothing selected", "Select a file in the queue first.")
            return
        job_id = self.current_job
        settings = replace(self._current_settings(), noise_region_s=self.noise_regions.get(job_id))
        path = self.jobs[job_id].input_path
        self.analyze_btn.setEnabled(False)
        self.report.setHtml("<i>Analyzing...</i>")

        def done(res, err, j=job_id):
            self.analyze_btn.setEnabled(True)
            if err:
                self.log_msg(f"Analyze failed: {err}")
                self.report.setPlainText(f"Analyze failed: {err}")
                return
            self.previews[j] = res
            if j == self.current_job:
                self._render_report(j)

        self._run_bg(lambda: diagnose(path, settings), done)

    # -- processing ---------------------------------------------------------------

    def on_process_clicked(self) -> None:
        if not self.jobs:
            QMessageBox.information(self, "Nothing to do", "Add some audio files to the queue first.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "A batch is already running.")
            return
        self.stop()
        settings = self._current_settings()
        out_dir = self.output_dir or DEFAULT_OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        jobs = []
        for job_id, job in self.jobs.items():
            base = os.path.splitext(os.path.basename(job.input_path))[0]
            out_path = os.path.join(out_dir, f"{base}_mastered.wav")
            jobs.append(replace(job, output_path=out_path,
                                settings=replace(settings, noise_region_s=self.noise_regions.get(job_id))))
            self.table.item(self.row_for_job[job_id], COL_STATUS).setText("Queued")
        self.process_btn.setEnabled(False)
        self.worker = BatchWorker(jobs, self.concurrency_spin.value())
        self.worker.progress.connect(self.on_progress)
        self.worker.job_done.connect(self.on_job_done)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.info.connect(self.log_msg)
        self.log_msg(f"Mastering {len(jobs)} file(s), up to {self.concurrency_spin.value()} at a time...")
        self.worker.start()

    def on_progress(self, job_id: str, message: str, pct: float) -> None:
        row = self.row_for_job.get(job_id)
        if row is None:
            return
        self.table.item(row, COL_STATUS).setText(message.strip().splitlines()[-1] if message.strip() else "")
        bar: QProgressBar = self.table.cellWidget(row, COL_PROGRESS)
        if bar:
            bar.setValue(int(pct * 100))

    def on_job_done(self, job_id: str, result, error) -> None:
        row = self.row_for_job.get(job_id)
        if row is None:
            return
        if error:
            self.table.item(row, COL_STATUS).setText("Error")
            self.table.item(row, COL_NOTES).setText(str(error).strip().splitlines()[-1])
            self.log_msg(f"ERROR: {os.path.basename(self.jobs[job_id].input_path)}:\n{error}")
            return
        self.results[job_id] = result
        self.table.item(row, COL_STATUS).setText("Done")
        final = result.acx_mp3 or result.acx_post
        ok = acx.passed(final)
        item = self.table.item(row, COL_ACX)
        item.setText("PASS" if ok else "FAIL")
        item.setForeground(QColor(40, 170, 70) if ok else QColor(220, 60, 60))
        pre, post = result.pre_analysis, result.post_analysis
        summary = (f"Noise {pre.noise_floor_dbfs:.0f}->{post.noise_floor_dbfs:.0f} dBFS, RMS {pre.rms_dbfs:.1f}->"
                   f"{post.rms_dbfs:.1f} dB, clicks fixed {result.clicks_fixed}")
        self.table.item(row, COL_NOTES).setText(summary)
        self.log_msg(f"{os.path.basename(result.output_path)}: {acx.summary(final)}; {summary}")
        if job_id == self.current_job:
            self._load_master_view(job_id)
            self._render_report(job_id)

    def on_all_done(self) -> None:
        self.process_btn.setEnabled(True)
        self.log_msg("Batch complete.")
        outputs = [r.output_path for r in self.results.values() if os.path.exists(r.output_path)]
        if outputs:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(outputs[0])))

    # -- report ------------------------------------------------------------------

    def _render_report(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        result, preview = self.results.get(job_id), self.previews.get(job_id)
        h = [f"<h3>{html.escape(os.path.basename(job.input_path))}</h3>"]
        if not result and not preview:
            h.append("<p style='color:gray'>Not analyzed yet. Click <b>Analyze Selected</b> to preview Smart Auto's "
                     "decisions and the raw ACX check, or <b>Master Queue</b> to process.</p>")
            self.report.setHtml("".join(h))
            return
        checks = [("Raw", result.acx_pre if result else preview[0])]
        if result:
            checks.append(("Master WAV", result.acx_post))
            if result.acx_mp3:
                checks.append(("ACX MP3", result.acx_mp3))
        names = [n for n, _, _ in max((c for _, c in checks), key=len)]
        h.append("<table cellpadding=4 border=1 style='border-collapse:collapse'><tr><th>ACX requirement</th>"
                 + "".join(f"<th>{t}</th>" for t, _ in checks) + "</tr>")
        for n in names:
            cells = []
            for _, items in checks:
                hit = next((it for it in items if it[0] == n), None)
                cells.append("<td></td>" if not hit else
                             f"<td style='color:{'#2a2' if hit[1] else '#d33'}'>{'&#10003;' if hit[1] else '&#10007;'} {html.escape(hit[2])}</td>")
            h.append(f"<tr><td>{html.escape(n)}</td>{''.join(cells)}</tr>")
        h.append("</table>")
        if result:
            h.append(f"<p><b>{html.escape(acx.summary(checks[-1][1]))}</b></p>")
            edges_failed = any(not ok and n.startswith(("Head", "Tail")) for n, ok, _ in checks[-1][1])
            if edges_failed and "roomtone_head_s" not in result.settings_used:
                h.append("<p style='color:#d33'>Room tone (ACX) is Off, so the head/tail were left as recorded. "
                         "Turn it On to set them to exact ACX lengths.</p>")
        notes = result.diagnosis if result else preview[1]
        h.append("<h4>What Smart Auto found and did</h4><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul>")
        if result:
            u = result.settings_used
            rows = []
            if u.get("tone_match_db"):
                rows.append("Tone match (dB @ Hz): " + ", ".join(f"{v:+.1f}@{k}" for k, v in u["tone_match_db"].items()))
            if isinstance(u.get("resonances"), list):
                rows.append("Room modes notched: " + (", ".join(f"{r['hz']} Hz ({r['cut_db']} dB, Q {r['q']})" for r in u["resonances"]) or "none found"))
            elif u.get("resonances"):
                rows.append(f"Room modes: {u['resonances']}")
            if u.get("mud_peaks"):
                rows.append("Mud peaks cut: " + ", ".join(f"{p['hz']} Hz ({p['cut_db']} dB)" for p in u["mud_peaks"]))
            for key, label in (("dereverb", "De-reverb"), ("noise_profile", "Noise profile"), ("neural", "Neural denoise"),
                               ("denoise_reduction_db", "Spectral denoise depth (dB)"), ("clicks_fixed", "Mouth clicks repaired"),
                               ("breaths_reduced", "Breaths reduced"), ("deess_max_db", "De-ess max (dB)"),
                               ("eq_presets", "EQ presets"), ("acx_rms_trim_db", "ACX RMS trim (dB)")):
                if key in u:
                    rows.append(f"{label}: {u[key]}")
            h.append("<h4>Processing</h4><ul>" + "".join(f"<li>{html.escape(str(r))}</li>" for r in rows) + "</ul>")
            h.append(f"<p style='color:gray'>{html.escape(result.output_path)}"
                     + (f"<br>{html.escape(result.mp3_path)}" if result.mp3_path else "") + "</p>")
        self.report.setHtml("".join(h))

    def log_msg(self, text: str) -> None:
        self.log.appendPlainText(text)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
