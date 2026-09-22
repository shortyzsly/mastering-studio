from __future__ import annotations

import os
import sys
import uuid
from dataclasses import replace

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
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
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .batch import MAX_CONCURRENT, BatchJob, run_batch
from .io_utils import SUPPORTED_EXTENSIONS
from .dsp import neural as neural_mod
from .pipeline import PipelineSettings

COL_FILE, COL_STATUS, COL_PROGRESS, COL_NOTES = range(4)


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
            if kind == "progress":
                self.progress.emit(job_id, msg, pct)
            elif kind in ("done", "error"):
                self.progress.emit(job_id, msg, pct)
            elif kind == "info":
                self.info.emit(msg)

        results = run_batch(self.jobs, on_event, max_workers=self.max_workers)
        for job in self.jobs:
            result, err = results.get(job.job_id, (None, "No result"))
            self.job_done.emit(job.job_id, result, err)
        self.all_done.emit()


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
        paths = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mastering Studio -- Audiobook Voice Cleanup")
        self.resize(1000, 640)

        self.jobs: dict[str, BatchJob] = {}
        self.row_for_job: dict[str, int] = {}
        self.worker: BatchWorker | None = None
        self.output_dir: str | None = None

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        outer.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_queue_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, stretch=1)

        outer.addWidget(self._build_log_panel())

    # -- UI construction -------------------------------------------------

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        add_btn = QPushButton("Add Files...")
        add_btn.clicked.connect(self.on_add_files_clicked)
        layout.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.on_remove_selected)
        layout.addWidget(remove_btn)

        clear_btn = QPushButton("Clear Queue")
        clear_btn.clicked.connect(self.on_clear_queue)
        layout.addWidget(clear_btn)

        layout.addStretch(1)

        out_btn = QPushButton("Output Folder...")
        out_btn.clicked.connect(self.on_choose_output_dir)
        layout.addWidget(out_btn)
        self.output_label = QLabel("(defaults next to each source file)")
        layout.addWidget(self.output_label)

        layout.addStretch(1)

        self.process_btn = QPushButton(f"Process Queue (up to {MAX_CONCURRENT} at a time)")
        self.process_btn.clicked.connect(self.on_process_clicked)
        layout.addWidget(self.process_btn)

        return bar

    def _build_queue_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        hint = QLabel("Drag and drop audio files here, or use Add Files.")
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        self.table = DropTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["File", "Status", "Progress", "Notes"])
        self.table.horizontalHeader().setSectionResizeMode(COL_FILE, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(COL_NOTES, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.files_dropped.connect(self.add_files)
        layout.addWidget(self.table)
        return panel

    def _build_settings_panel(self) -> QWidget:
        box = QGroupBox("Processing Settings")
        form = QFormLayout(box)

        self.dehum_combo = QComboBox()
        self.dehum_combo.addItems(["Auto (recommended)", "Off"])
        form.addRow("De-hum (50/60 Hz buzz):", self.dehum_combo)

        self.neural_combo = QComboBox()
        self.neural_combo.addItems(["On (DeepFilterNet3)", "Off"])
        if not neural_mod.installed():
            self.neural_combo.setEnabled(False)
            self.neural_combo.setToolTip("Neural environment not found. Set MASTERING_STUDIO_NEURAL_PYTHON or see README.")
        form.addRow("Neural denoiser:", self.neural_combo)

        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["Auto (recommended)", "Gentle", "Moderate", "Aggressive", "Off"])
        form.addRow("Denoise strength:", self.denoise_combo)

        self.eq_combo = QComboBox()
        self.eq_combo.addItems(["Auto (tighten low end, ease harshness)", "Half strength", "Off"])
        form.addRow("Corrective EQ:", self.eq_combo)

        self.lowend_combo = QComboBox()
        self.lowend_combo.addItems(["Medium (recommended)", "Light", "Strong", "Off"])
        form.addRow("Low end (mud & bass):", self.lowend_combo)

        self.declick_combo = QComboBox()
        self.declick_combo.addItems(["On (recommended)", "Off"])
        form.addRow("De-click:", self.declick_combo)

        self.soften_combo = QComboBox()
        self.soften_combo.addItems(["Strong (recommended)", "Medium", "Light", "Off"])
        form.addRow("Harshness (top end):", self.soften_combo)

        self.dehiss_combo = QComboBox()
        self.dehiss_combo.addItems(["Medium (recommended)", "Light", "Strong", "Off"])
        form.addRow("Hiss under soft words:", self.dehiss_combo)

        self.deess_combo = QComboBox()
        self.deess_combo.addItems(["Medium (recommended)", "Light", "Strong", "Off"])
        form.addRow("De-esser (sibilance):", self.deess_combo)

        self.debreath_combo = QComboBox()
        self.debreath_combo.addItems(["Medium (-10 dB)", "Light (-6 dB)", "Strong (-14 dB)", "Off"])
        form.addRow("De-breath:", self.debreath_combo)

        self.compress_combo = QComboBox()
        self.compress_combo.addItems(["Auto (recommended)", "Off"])
        form.addRow("Compression:", self.compress_combo)

        self.loudness_combo = QComboBox()
        self.loudness_combo.addItems([
            "ACX submission (-21.5 LUFS, RMS ~-22, -3 dBTP)",
            "Audiobook (-23 LUFS, -3 dBTP; RMS -23.4, under ACX's floor)",
            "Podcast (-16 LUFS, -1 dBTP)",
            "Off",
        ])
        form.addRow("Loudness target:", self.loudness_combo)

        self.roomtone_combo = QComboBox()
        self.roomtone_combo.addItems([
            "On (0.75 s pad + smooth outlier gaps)",
            "Pad head/tail only",
            "Smooth outlier gaps only",
            "Off",
        ])
        form.addRow("Room tone (ACX head/tail):", self.roomtone_combo)

        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, MAX_CONCURRENT)
        self.concurrency_spin.setValue(MAX_CONCURRENT)
        form.addRow("Max concurrent (batch):", self.concurrency_spin)

        note = QLabel(
            "Analysis picks denoise/compression strength per file automatically.\n"
            "Always listen to a result before trusting the numbers -- aggressive\n"
            "settings can look clean on meters while sounding worse by ear."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        form.addRow(note)

        return box

    def _build_log_panel(self) -> QWidget:
        box = QGroupBox("Log")
        layout = QVBoxLayout(box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)
        return box

    # -- queue management --------------------------------------------------

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
            ext = os.path.splitext(path)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                self.log_msg(f"Skipping unsupported file type: {path}")
                continue
            job_id = str(uuid.uuid4())
            job = BatchJob(job_id=job_id, input_path=path, output_path="")
            self.jobs[job_id] = job
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, COL_FILE, QTableWidgetItem(os.path.basename(path)))
            self.table.setItem(row, COL_STATUS, QTableWidgetItem("Queued"))
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            self.table.setCellWidget(row, COL_PROGRESS, progress_bar)
            self.table.setItem(row, COL_NOTES, QTableWidgetItem(""))
            self.row_for_job[job_id] = row
            added += 1
        if added:
            self.log_msg(f"Added {added} file(s) to queue.")

    def on_remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            job_id = self._job_id_for_row(row)
            if job_id:
                del self.jobs[job_id]
                del self.row_for_job[job_id]
            self.table.removeRow(row)
        self._reindex_rows()

    def on_clear_queue(self) -> None:
        self.table.setRowCount(0)
        self.jobs.clear()
        self.row_for_job.clear()

    def on_choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if directory:
            self.output_dir = directory
            self.output_label.setText(directory)

    def _job_id_for_row(self, row: int) -> str | None:
        for job_id, r in self.row_for_job.items():
            if r == row:
                return job_id
        return None

    def _reindex_rows(self) -> None:
        self.row_for_job = {}
        for row in range(self.table.rowCount()):
            filename = self.table.item(row, COL_FILE).text()
            for job_id, job in self.jobs.items():
                if os.path.basename(job.input_path) == filename and job_id not in self.row_for_job.values():
                    self.row_for_job[job_id] = row
                    break

    # -- settings -> PipelineSettings --------------------------------------

    def _current_settings(self) -> PipelineSettings:
        settings = PipelineSettings()

        if self.dehum_combo.currentText() == "Off":
            settings.enable_dehum = False

        settings.enable_neural = self.neural_combo.currentText().startswith("On") and neural_mod.installed()

        denoise_choice = self.denoise_combo.currentText()
        if denoise_choice == "Off":
            settings.enable_denoise = False
        elif denoise_choice != "Auto (recommended)":
            settings.denoise_strength = denoise_choice.lower()

        eq_choice = self.eq_combo.currentText()
        if eq_choice == "Off":
            settings.enable_eq = False
        elif eq_choice.startswith("Half"):
            settings.eq_amount = 0.5

        lowend_choice = self.lowend_combo.currentText()
        if lowend_choice == "Off":
            settings.enable_lowend = False
        else:
            settings.lowend_strength = lowend_choice.split()[0].lower()

        soften_choice = self.soften_combo.currentText()
        if soften_choice == "Off":
            settings.enable_soften = False
        else:
            settings.soften_strength = soften_choice.split()[0].lower()

        dehiss_choice = self.dehiss_combo.currentText()
        if dehiss_choice == "Off":
            settings.enable_dehiss = False
        else:
            settings.dehiss_strength = dehiss_choice.split()[0].lower()

        deess_choice = self.deess_combo.currentText()
        if deess_choice == "Off":
            settings.enable_deess = False
        else:
            settings.deess_strength = deess_choice.split()[0].lower()

        debreath_choice = self.debreath_combo.currentText()
        if debreath_choice == "Off":
            settings.enable_debreath = False
        else:
            settings.debreath_strength = debreath_choice.split()[0].lower()

        roomtone_choice = self.roomtone_combo.currentText()
        if roomtone_choice == "Off":
            settings.enable_roomtone = False
            settings.enable_roomtone_floor = False
        elif roomtone_choice.startswith("Pad"):
            settings.enable_roomtone_floor = False
        elif roomtone_choice.startswith("Smooth"):
            settings.enable_roomtone = False

        if self.declick_combo.currentText() == "Off":
            settings.enable_declick = False

        if self.compress_combo.currentText() == "Off":
            settings.enable_compress = False

        loudness_choice = self.loudness_combo.currentText()
        if loudness_choice.startswith("Off"):
            settings.enable_loudness = False
            settings.enable_limiter = False
        elif loudness_choice.startswith("Podcast"):
            settings.target_lufs = -16.0
            settings.peak_ceiling_dbfs = -1.0
        elif loudness_choice.startswith("Audiobook"):
            settings.target_lufs = -23.0
            settings.peak_ceiling_dbfs = -3.0
        else:
            settings.target_lufs = -21.5
            settings.peak_ceiling_dbfs = -3.0

        return settings

    # -- processing ---------------------------------------------------------

    def on_process_clicked(self) -> None:
        if not self.jobs:
            QMessageBox.information(self, "Nothing to do", "Add some audio files to the queue first.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "A batch is already running.")
            return

        settings = self._current_settings()
        jobs = []
        for job_id, job in self.jobs.items():
            in_path = job.input_path
            base, ext = os.path.splitext(os.path.basename(in_path))
            out_dir = self.output_dir or os.path.dirname(in_path)
            out_path = os.path.join(out_dir, f"{base}_mastered{ext}")
            jobs.append(replace(job, output_path=out_path, settings=settings))
            row = self.row_for_job[job_id]
            self.table.item(row, COL_STATUS).setText("Queued")

        self.process_btn.setEnabled(False)
        max_workers = self.concurrency_spin.value()
        self.worker = BatchWorker(jobs, max_workers)
        self.worker.progress.connect(self.on_progress)
        self.worker.job_done.connect(self.on_job_done)
        self.worker.all_done.connect(self.on_all_done)
        self.worker.info.connect(self.log_msg)
        self.log_msg(f"Processing {len(jobs)} file(s), up to {max_workers} at a time...")
        self.worker.start()

    def on_progress(self, job_id: str, message: str, pct: float) -> None:
        row = self.row_for_job.get(job_id)
        if row is None:
            return
        self.table.item(row, COL_STATUS).setText(message)
        bar: QProgressBar = self.table.cellWidget(row, COL_PROGRESS)
        if bar:
            bar.setValue(int(pct * 100))

    def on_job_done(self, job_id: str, result, error) -> None:
        row = self.row_for_job.get(job_id)
        if row is None:
            return
        if error:
            self.table.item(row, COL_STATUS).setText("Error")
            self.table.item(row, COL_NOTES).setText(str(error).splitlines()[-1])
            self.log_msg(f"ERROR processing job: {error}")
            return
        self.table.item(row, COL_STATUS).setText("Done")
        pre, post = result.pre_analysis, result.post_analysis
        summary = (
            f"Noise floor {pre.noise_floor_dbfs:.0f}->{post.noise_floor_dbfs:.0f} dBFS, "
            f"LUFS {pre.integrated_lufs:.1f}->{post.integrated_lufs:.1f}, "
            f"clicks fixed: {result.clicks_fixed}"
        )
        used = result.settings_used
        if "breaths_reduced" in used:
            summary += f", breaths reduced: {used['breaths_reduced']}"
        if "hum_f0_hz" in used:
            summary += f", hum removed ({used['hum_f0_hz']:.0f} Hz, {used['hum_lines']} lines)"
        if used.get("mud_peaks"):
            summary += ", mud cut @" + "/".join(str(p["hz"]) for p in used["mud_peaks"]) + " Hz"
        if "deess_max_db" in used:
            summary += f", de-ess up to {used['deess_max_db']} dB"
        if used.get("roomtone_head_s") or used.get("roomtone_tail_s"):
            summary += f", room tone padded +{used.get('roomtone_head_s', 0):.2f}/+{used.get('roomtone_tail_s', 0):.2f}s"
        if used.get("roomtone_floor_events"):
            summary += f", smoothed {used['roomtone_floor_events']} quiet outlier(s)"
        self.table.item(row, COL_NOTES).setText(summary)
        self.log_msg(f"{os.path.basename(result.output_path)}: {summary}")

    def on_all_done(self) -> None:
        self.process_btn.setEnabled(True)
        self.log_msg("Batch complete.")
        outputs = [job.output_path for job in self.jobs.values() if os.path.exists(job.output_path)]
        if outputs:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(outputs[0])))

    def log_msg(self, text: str) -> None:
        self.log.appendPlainText(text)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
