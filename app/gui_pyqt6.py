#!/usr/bin/env python3
# DentWise -- PyQt5/PyQt6 real-time GUI for live_unified.py
# Usage: python gui_pyqt6.py --model path/to/model.h5 --device 0 --out results/
# Install: pip install PyQt5  (recommended for conda/TF envs)
#          pip install PyQt6  (alternative -- if DLL errors, switch to PyQt5)

import sys, os, argparse, time, re
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

# Auto-detect PyQt5 vs PyQt6 -- tries PyQt5 first (more stable with TensorFlow)
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
        QProgressBar, QListWidget, QListWidgetItem, QTabWidget,
        QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
        QComboBox, QGroupBox, QSizePolicy, QStatusBar,
        QMessageBox, QLineEdit, QSplitter, QScrollArea, QShortcut,
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
    from PyQt5.QtGui import QImage, QPixmap, QColor, QKeySequence
    print("[Qt] PyQt5")
except ImportError:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
        QProgressBar, QListWidget, QListWidgetItem, QTabWidget,
        QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox,
        QComboBox, QGroupBox, QSizePolicy, QStatusBar,
        QMessageBox, QLineEdit, QSplitter, QScrollArea,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QMutexLocker
    from PyQt6.QtGui import QImage, QPixmap, QColor, QKeySequence, QShortcut, QShortcut
    Qt.AlignCenter = Qt.AlignmentFlag.AlignCenter
    Qt.AlignRight  = Qt.AlignmentFlag.AlignRight
    Qt.KeepAspectRatio = Qt.AspectRatioMode.KeepAspectRatio
    Qt.SmoothTransformation = Qt.TransformationMode.SmoothTransformation
    QImage.Format_RGB888 = QImage.Format.Format_RGB888
    QSizePolicy.Expanding = QSizePolicy.Policy.Expanding
    QMessageBox.Yes = QMessageBox.StandardButton.Yes
    QMessageBox.No  = QMessageBox.StandardButton.No
    print("[Qt] PyQt6")

# Colours
TEAL    = "#1D9E75"; TEAL_BG = "#0a2a1e"
BLUE    = "#378ADD"; BBRIGHT = "#5ba3f5"
AMBER   = "#EF9F27"; RED     = "#E24B4A"
GREEN   = "#639922"; DARK_BG = "#0d1117"
PANEL   = "#161b22"; SURFACE = "#21262d"
BORDER  = "#30363d"; TEXT    = "#e6edf3"
MUTED   = "#8b949e"; HINT    = "#484f58"

# Build style string without f-strings (avoids Python 3.12 f-string parser issues)
def _s(*parts): return "".join(parts)
STYLE = _s(
    "QMainWindow,QWidget{background:", DARK_BG, ";color:", TEXT, ";font-family:Inter,Segoe UI,SF Pro Display,-apple-system,Helvetica Neue,sans-serif;font-size:28px;}",
    "QFrame{border:none;}",
    "QLabel{color:", TEXT, ";}",
    "QPushButton{background:", SURFACE, ";color:", TEXT, ";border:1px solid ", BORDER, ";border-radius:6px;padding:8px 14px;font-size:26px;font-weight:500;}",
    "QPushButton:hover{background:#2d333b;border-color:", MUTED, ";}",
    "QPushButton:pressed{background:", BORDER, ";}",
    "QPushButton:disabled{color:", HINT, ";border-color:", HINT, ";}",
    "QProgressBar{border:1px solid ", BORDER, ";border-radius:3px;background:", SURFACE, ";height:6px;}",
    "QProgressBar::chunk{background:", TEAL, ";border-radius:3px;}",
    "QListWidget{background:", PANEL, ";border:1px solid ", BORDER, ";border-radius:6px;color:", TEXT, ";font-size:24px;}",
    "QListWidget::item{padding:3px 8px;}",
    "QListWidget::item:selected{background:", SURFACE, ";}",
    "QTabWidget::pane{border:1px solid ", BORDER, ";border-radius:6px;background:", PANEL, ";}",
    "QTabBar::tab{background:", SURFACE, ";color:", MUTED, ";border:1px solid ", BORDER, ";border-bottom:none;border-radius:6px 6px 0 0;padding:6px 16px;font-size:26px;}",
    "QTabBar::tab:selected{background:", PANEL, ";color:", TEXT, ";}",
    "QGroupBox{border:1px solid ", BORDER, ";border-radius:6px;margin-top:22px;padding-top:8px;font-size:18px;color:", MUTED, ";font-weight:700;letter-spacing:0.05em;}",
    "QGroupBox::title{subcontrol-origin:margin;top:-2px;left:12px;padding:2px 6px;background:transparent;}",
    "QSpinBox,QDoubleSpinBox,QComboBox,QLineEdit{background:", SURFACE, ";border:1px solid ", BORDER, ";border-radius:4px;color:", TEXT, ";padding:4px 8px;font-size:26px;}",
    "QSpinBox:focus,QDoubleSpinBox:focus,QComboBox:focus,QLineEdit:focus{border-color:", TEAL, ";}",
    "QCheckBox{color:", TEXT, ";spacing:6px;font-size:26px;}",
    "QCheckBox::indicator{width:16px;height:16px;border:1px solid ", BORDER, ";border-radius:3px;background:", SURFACE, ";}",
    "QCheckBox::indicator:checked{background:", TEAL, ";border-color:", TEAL, ";}",
    "QScrollBar:vertical{background:", SURFACE, ";width:7px;border-radius:3px;}",
    "QScrollBar::handle:vertical{background:", BORDER, ";border-radius:3px;min-height:20px;}",
    "QStatusBar{background:", PANEL, ";color:", MUTED, ";font-size:24px;border-top:1px solid ", BORDER, ";}",
)

BTN_PRIMARY = "QPushButton{background:#1a3a6e;color:" + BLUE  + ";border:1px solid " + BLUE  + ";border-radius:6px;padding:8px 14px;font-size:26px;font-weight:600;}"
BTN_SUCCESS = "QPushButton{background:#1a3a20;color:" + GREEN + ";border:1px solid " + GREEN + ";border-radius:6px;padding:8px 14px;font-size:26px;font-weight:600;}"
BTN_DANGER  = "QPushButton{background:#3a1a1a;color:" + RED   + ";border:1px solid " + RED   + ";border-radius:6px;padding:8px 14px;font-size:26px;font-weight:600;}"
BTN_AMBER   = "QPushButton{background:#3a2a10;color:" + AMBER + ";border:1px solid " + AMBER + ";border-radius:6px;padding:8px 14px;font-size:26px;font-weight:600;}"
BTN_TEAL    = "QPushButton{background:" + TEAL_BG + ";color:" + TEAL + ";border:1px solid " + TEAL + ";border-radius:6px;padding:8px 14px;font-size:26px;font-weight:600;}"
BTN_REC     = "QPushButton{background:#3a1a1a;color:" + RED   + ";border:1px solid " + RED   + ";border-radius:6px;padding:8px 14px;font-size:26px;font-weight:600;}"


# =============================================================================
#  InferenceThread
# =============================================================================

class InferenceThread(QThread):
    frame_ready   = pyqtSignal(QImage)
    metrics_ready = pyqtSignal(dict)
    error         = pyqtSignal(str)

    def __init__(self, capture, pano_system, chart_annotator, chart_system, voice, parent=None):
        super().__init__(parent)
        self.capture         = capture
        self.pano_system     = pano_system
        self.chart_annotator = chart_annotator
        self.chart_system    = chart_system
        self.voice           = voice
        self._running        = True
        self._mode           = "stitching"
        self._mutex          = QMutex()
        self._fps_t0         = time.time()
        self._fps_n          = 0
        self._fps            = 0.
        self._paused         = False   # unified pause flag

    def set_mode(self, m):
        with QMutexLocker(self._mutex):
            self._mode = m

    def set_paused(self, paused: bool):
        """Pause/resume frame processing. Camera keeps running, inference stops."""
        with QMutexLocker(self._mutex):
            self._paused = paused

    def stop(self):
        self._running = False

    def run(self):
        skip = 0
        imu_conf = 0.
        while self._running:
            frame = self.capture.get_frame()
            if frame is None:
                skip += 1
                if skip > 100: self.error.emit("HD60X lost"); break
                self.msleep(10); continue
            skip = 0

            with QMutexLocker(self._mutex):
                mode = self._mode

            imu = self.pano_system.imu.get_latest()
            self._fps_n += 1
            if time.time() - self._fps_t0 >= 1.0:
                self._fps   = self._fps_n / (time.time() - self._fps_t0)
                self._fps_n = 0; self._fps_t0 = time.time()

            ann = frame
            try:
                if mode == "stitching":
                    ann, imu_conf = self._run_stitch(frame, imu, imu_conf)
                elif mode == "charting":
                    ann = self._run_chart(frame)
            except Exception as e:
                pass

            try:
                rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                if rgb is None or rgb.size == 0: continue
                h, w, ch = rgb.shape
                img = QImage(bytes(rgb.data), w, h, ch * w, QImage.Format_RGB888)
                self.frame_ready.emit(img.copy())
            except Exception:
                pass

    def _run_stitch(self, frame, imu_now, imu_conf):
        import live_unified as U
        ps = self.pano_system
        result = ps.proc.process(frame)
        if result is None:
            return frame, imu_conf

        gm_x  = result["gm_x_mm"]; gm_y = result["gm_y_mm"]
        q     = result["quality_score"]
        gc    = result.get("gate_comp", {"combined": q})
        gp    = result.get("gate_pass", q >= 0.15)

        if ps.recording and not ps.voice_paused and gp:
            gx_f, imu_conf = ps.cf.update(gm_x, imu_now, q)
        else:
            gx_f = gm_x; imu_conf = 0.

        ps._detect_sweep_dir(gx_f); ps._update_jitter(gm_x)
        da = True
        if ps.recording and not ps.voice_paused:
            da, _, _ = ps._sweep_tracker.update(gx_f)

        imu_used = ps.imu.connected and imu_now.valid and abs(imu_now.gyro_z) > 0.1

        if ps.recording and not ps.voice_paused and gp and da:
            col0   = ps._compute_col0(gx_f)
            gm_row = gm_y / U.PIXEL_SPACING_MM
            ps.canvas.add_frame(result["frame_256"], col0, gm_row, q, imu_used=imu_used)
            dy = result.get("gm_dy_px", 0.)
            kf_ok = ((ps._last_keyframe_col0 is None or
                      abs(col0 - ps._last_keyframe_col0) >= U.KEYFRAME_MIN_DIST_PX)
                     and abs(dy) <= U.KEYFRAME_JITTER_PX and q >= U.KEYFRAME_QUALITY)
            if kf_ok:
                ps.sharp_canvas.add_frame(result["frame_256"], col0, gm_row, q, imu_used=imu_used)
                ps._last_keyframe_col0 = col0
            ps._session_frames.append((result["frame_256"] * 255).astype(np.uint8))
            coord = {"frame_n": len(ps._session_coords), "ts": time.time(),
                     "gm_x_mm": round(gm_x,4), "gm_y_mm": round(gm_y,4),
                     "col0_px": round(col0,2), "gm_row_frame": round(gm_row,2),
                     "gm_x_fused": round(gx_f,4), "quality_score": round(q,4),
                     "gate_score": round(gc.get("combined",q),4),
                     "imu_yaw": round(imu_now.yaw,3), "imu_pitch": round(imu_now.pitch,3),
                     "imu_roll": round(imu_now.roll,3), "imu_gyro_z": round(imu_now.gyro_z,3),
                     "imu_conf": round(imu_conf,3), "imu_connected": imu_now.valid,
                     "imu_receiving": ps.imu.connected, "imu_valid": imu_now.valid,
                     "is_anchor": False, "is_bad": False}
            ps._session_coords.append(coord)
            if ps._pano_metrics:
                ti = int(col0 // 256)
                tl = ps.canvas._tiles.get(ti, {}).get("level", 0)
                ps._pano_metrics.log(result, col0_px=col0, imu_now=imu_now,
                    imu_conf=imu_conf, imu_alpha=imu_conf,
                    canvas=ps.canvas, tile_idx=ti, tile_level=tl,
                    gm_x_fused=gx_f, frame_bgr=frame)
            ps.audit.log(frame_n=len(ps._session_coords), ts=coord["ts"],
                gm_x=gm_x, gm_y=gm_y, col0_px=col0, quality=q,
                gater_components=gc, imu_yaw=imu_now.yaw,
                imu_conf=imu_conf, imu_valid=imu_now.valid, anchor=False, bad=False)
            ps.stats["frames_good"] += 1
        ps.stats["frames_total"] += 1

        ann = result["annotated"]
        ann = ps._draw_gate_overlay(ann, gc, gp)
        if ps.show_imu:
            ann = ps._draw_imu_overlay(ann, imu_now, imu_conf)
        ann = ps._draw_recording_overlay(ann)

        jit = getattr(ps, "_jitter_gm_x", 0.) * 1000
        self.metrics_ready.emit({
            "mode": "stitching", "quality": q, "gate": gc.get("combined", q),
            "fps": round(self._fps, 1), "frames": ps.stats.get("frames_good", 0),
            "kf": ps.sharp_canvas.frame_count, "jitter_x": round(jit, 1),
            "mem_mb": round(ps.canvas.memory_mb(), 2),
            "imu_valid": imu_now.valid, "imu_alpha": round(imu_conf, 3),
            "imu_yaw": round(imu_now.yaw, 2), "recording": ps.recording,
        })
        return ann, imu_conf

    def _run_chart(self, frame):
        ca = self.chart_annotator; cs = self.chart_system
        t0 = time.time()
        batch = ca.preprocess_frame_fast(frame)
        if batch is None:
            return frame
        preds = ca.predict_single_frame(batch)
        inf_ms = (time.time() - t0) * 1000.
        h, w  = frame.shape[:2]
        proc  = ca.post_process_predictions(preds, (h, w))
        meas  = ca.get_measurements_throttled(preds)
        ann   = ca.create_annotated_frame(frame, proc)
        # Update performance_stats so add_performance_overlay shows real values
        ca.performance_stats['display_fps']     = round(self._fps, 1)
        ca.performance_stats['inference_time']  = round(inf_ms, 1)
        ca.performance_stats['avg_latency']     = round(inf_ms, 1)
        ca.performance_stats['total_frames']    = cs.tooth_buffer.__len__() if cs.scanning_active else 0
        ann   = ca.add_performance_overlay(ann)
        if cs.scanning_active and meas and not self._paused:
            cs.add_frame(ann, meas)   # paused = do not add to tooth buffer
        self.metrics_ready.emit({
            "mode": "charting", "quality": ca._last_quality,
            "fps": round(self._fps, 1),
            "tooth": getattr(cs, "current_tooth", None) or "--",
            "scanning": cs.scanning_active, "meas": meas or {},
        })
        return ann


# =============================================================================
#  VoiceThread
# =============================================================================

class VoiceThread(QThread):
    command_received = pyqtSignal(str, str)
    mode_changed     = pyqtSignal(str)
    log_entry        = pyqtSignal(str, str)

    def __init__(self, voice, parent=None):
        super().__init__(parent)
        self._voice = voice; self._running = True

    def stop(self):
        self._running = False
        if self._voice: self._voice.stop()

    def run(self):
        while self._running:
            if not self._voice: self.sleep(1); continue
            m = self._voice.get_mode_event()
            if m: self.mode_changed.emit(m)
            # Chart commands (go/stop/end) — active in charting mode
            c = self._voice.get_chart_command()
            if c: self.command_received.emit(c[0], c[1]); self.log_entry.emit(str(c), c[1].upper())
            # Pano commands (on/off/save etc) — active in stitching mode
            p = self._voice.get_pano_command()
            if p: self.command_received.emit("pano", p); self.log_entry.emit(p, p.upper())
            # Tooth/surface commands — active in BOTH modes so stitching
            # can label subfolders by voice (e.g. say "14 buccal" during sweep)
            t = self._voice.get_tooth_command()
            if t: self.command_received.emit("tooth", t); self.log_entry.emit(t, "TOOTH " + t)
            self.msleep(30)



# =============================================================================
#  VoiceTuneThread  —  runs enrollment recording OR fine-tuning in background
# =============================================================================

class VoiceTuneThread(QThread):
    """
    Background thread for enrollment recording and Whisper LoRA fine-tuning.
    Emits log lines in real time so VoiceTuneDialog can show progress.
    """
    log_line   = pyqtSignal(str)   # one text line to append to the log widget
    progress   = pyqtSignal(int)   # 0-100 for progress bar
    finished_ok= pyqtSignal(str)   # success message
    error_sig  = pyqtSignal(str)   # error message

    def __init__(self, mode: str, enrollment_dir: str, user: str,
                 whisper_model: str, mic_device, parent=None):
        super().__init__(parent)
        self.mode           = mode            # "enroll" or "finetune"
        self.enrollment_dir = enrollment_dir
        self.user           = user
        self.whisper_model  = whisper_model
        self.mic_device     = mic_device
        self._stop          = False

    def run(self):
        import sys, io, contextlib
        from pathlib import Path

        # Redirect stdout/stderr so we capture print() from finetune_whisper
        class _Tee:
            def __init__(self, signal):
                self._sig = signal; self._buf = ""
            def write(self, s):
                self._buf += s
                nl = chr(10)
                while nl in self._buf:
                    line, self._buf = self._buf.split(nl, 1)
                    if line.strip():
                        self._sig.emit(line)
            def flush(self): pass

        tee = _Tee(self.log_line)
        try:
            import finetune_whisper as FT
        except ImportError as e:
            self.error_sig.emit(f"Cannot import finetune_whisper.py: {e}")
            return

        try:
            user_id = self.user.strip().lower().replace(" ","_") if self.user else None
            root    = Path(self.enrollment_dir)
            enr_dir = root / user_id if user_id else root
            enr_dir.mkdir(parents=True, exist_ok=True)
            phrases = FT.STANDARD_PHRASES

            old_stdout = sys.stdout; old_stderr = sys.stderr
            sys.stdout  = tee;       sys.stderr  = tee

            if self.mode == "enroll":
                self.log_line.emit(f"Starting enrollment for '{self.user or 'default'}'")
                self.log_line.emit(f"Folder: {enr_dir}")
                self.log_line.emit(f"Phrases to record: {len(phrases)}")
                self.log_line.emit("─" * 50)
                self.log_line.emit("IMPORTANT: Speak clearly when prompted.")
                self.log_line.emit("Press ENTER before each phrase, then say it.")
                self.log_line.emit("─" * 50)
                # Run enrollment — this is interactive but stdout is captured
                FT.record_enrollment(enr_dir, phrases, mic_device=self.mic_device)
                sys.stdout = old_stdout; sys.stderr = old_stderr
                self.finished_ok.emit(
                    f"Enrollment complete for '{self.user}'.  "
                    f"Click Fine-Tune Whisper to train the adapted model.")
            else:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.log_line.emit(f"Fine-tuning Whisper '{self.whisper_model}' for '{self.user or 'default'}'")
                self.log_line.emit(f"Device: {device}  |  Phrases: {len(phrases)}")
                self.log_line.emit("─" * 50)
                out_path = enr_dir / "adapted_model.pt"
                summary = FT.finetune_one_user(
                    enrollment_dir = enr_dir,
                    user_label     = self.user or "default",
                    model_size     = self.whisper_model,
                    device         = device,
                    phrases        = phrases,
                    lora_rank      = FT.LORA_RANK,
                    lora_alpha     = FT.LORA_ALPHA,
                    lora_dropout   = FT.LORA_DROPOUT,
                    lr             = FT.LEARNING_RATE,
                    max_epochs     = FT.MAX_EPOCHS,
                    patience       = FT.EARLY_STOP_PAT,
                    aug_factor     = FT.AUG_FACTOR,
                    eval_split     = 0.15,
                    output_path    = out_path,
                )
                sys.stdout = old_stdout; sys.stderr = old_stderr
                wer_b = summary.get("baseline_wer", 0)
                wer_f = summary.get("final_wer", 0)
                self.finished_ok.emit(
                    f"Fine-tuning complete!  "
                    f"WER: {wer_b:.1%} → {wer_f:.1%}  |  "
                    f"Model saved: {out_path}")
        except Exception as e:
            try:
                sys.stdout = old_stdout; sys.stderr = old_stderr
            except: pass
            self.error_sig.emit(f"Error: {e}")


# =============================================================================
#  VoiceTuneDialog  —  in-app enrollment + fine-tune window
# =============================================================================

class VoiceTuneDialog(QDialog):
    """
    Interactive in-app voice enrollment + Whisper LoRA fine-tuning dialog.

    Enrollment flow:
      - Shows each phrase with RECORD / SKIP / PREV / NEXT buttons
      - User clicks RECORD, speaks the phrase, recording auto-stops
      - Can navigate freely: jump to any phrase, re-record any phrase
      - Progress bar shows overall completion
      - Fine-tune runs in background with live log output
    """
    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args       = args
        self._thread    = None
        self._phrases   = None   # loaded lazily
        self._cur_idx   = 0
        self._recording = False
        self._pa        = None
        self._stream    = None
        self._rec_frames= []
        self._rec_timer = QTimer(self)
        self._rec_timer.timeout.connect(self._on_rec_tick)

        self.setWindowTitle("DentWise — Voice Personalisation")
        self.setMinimumSize(860, 680)
        self.setStyleSheet(STYLE)

        lay = QVBoxLayout(self); lay.setSpacing(8)

        # ── Title ───────────────────────────────────────────────────────────
        t = QLabel("Voice Personalisation")
        t.setStyleSheet("font-size:30px;font-weight:700;color:" + TEAL + ";padding:6px 0;")
        lay.addWidget(t)

        # ── Config row ──────────────────────────────────────────────────────
        fr = QFormLayout(); fr.setSpacing(6)
        self.uid = QLineEdit(args.user or "")
        self.uid.setPlaceholderText("e.g. dr_smith")
        self.uid.textChanged.connect(self._refresh_status)
        self.wm  = QComboBox()
        self.wm.addItems(["tiny","base","small","medium"])
        self.wm.setCurrentText(getattr(args,"whisper_model","small"))
        enr      = getattr(args,"enrollment_dir","enrollment")
        enr_row  = QHBoxLayout(); enr_row.setSpacing(6)
        self.enr = QLineEdit(enr)
        btn_br   = QPushButton("Browse"); btn_br.setFixedWidth(100)
        btn_br.clicked.connect(self._browse)
        enr_row.addWidget(self.enr); enr_row.addWidget(btn_br)
        self.mic_cb = QComboBox()
        self._populate_mics()
        fr.addRow("Clinician name:", self.uid)
        fr.addRow("Whisper model:",  self.wm)
        fr.addRow("Enrollment folder:", enr_row)
        fr.addRow("Microphone:", self.mic_cb)
        lay.addLayout(fr)

        # ── Status ──────────────────────────────────────────────────────────
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("font-size:19px;color:" + MUTED + ";")
        self.status_lbl.setWordWrap(True)
        lay.addWidget(self.status_lbl)

        # ── TABS: Enrollment | Fine-Tune ────────────────────────────────────
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        # ── TAB 1: Enrollment ────────────────────────────────────────────────
        enr_tab = QWidget(); el = QVBoxLayout(enr_tab); el.setSpacing(8)

        # Overall progress
        prog_row = QHBoxLayout()
        self.enr_prog_lbl = QLabel("0 / 0 recorded")
        self.enr_prog_lbl.setStyleSheet("font-size:20px;color:" + MUTED + ";")
        self.enr_prog = QProgressBar(); self.enr_prog.setRange(0,1)
        prog_row.addWidget(self.enr_prog); prog_row.addWidget(self.enr_prog_lbl)
        el.addLayout(prog_row)

        # Phrase list (scrollable) on the left, controls on the right
        main_row = QHBoxLayout(); main_row.setSpacing(10)

        # Phrase list
        self.phrase_list = QListWidget()
        self.phrase_list.setFixedWidth(320)
        self.phrase_list.setStyleSheet(
            "font-size:20px;background:#161b22;color:#e6edf3;border-radius:4px;")
        self.phrase_list.currentRowChanged.connect(self._on_phrase_sel)
        main_row.addWidget(self.phrase_list)

        # Right panel
        right = QVBoxLayout(); right.setSpacing(8)

        # Current phrase display
        self.phrase_lbl = QLabel("—")
        self.phrase_lbl.setAlignment(Qt.AlignCenter if hasattr(Qt,'AlignCenter') else Qt.AlignmentFlag.AlignCenter)
        self.phrase_lbl.setWordWrap(True)
        self.phrase_lbl.setStyleSheet(
            "font-size:34px;font-weight:700;color:" + TEAL + ";"
            "border:2px solid " + TEAL + ";border-radius:8px;padding:12px;")
        right.addWidget(self.phrase_lbl)

        # Recording status / countdown
        self.rec_status = QLabel("Click RECORD to start")
        self.rec_status.setAlignment(Qt.AlignCenter if hasattr(Qt,'AlignCenter') else Qt.AlignmentFlag.AlignCenter)
        self.rec_status.setStyleSheet("font-size:24px;color:" + MUTED + ";")
        right.addWidget(self.rec_status)

        # VU / level bar
        self.vu_bar = QProgressBar()
        self.vu_bar.setRange(0, 100); self.vu_bar.setValue(0)
        self.vu_bar.setTextVisible(False)
        self.vu_bar.setStyleSheet(
            "QProgressBar{background:#0d1117;border-radius:4px;height:14px;}"
            "QProgressBar::chunk{background:" + TEAL + ";border-radius:4px;}")
        right.addWidget(self.vu_bar)

        # Record / navigation buttons
        rec_row = QHBoxLayout(); rec_row.setSpacing(6)
        self.b_rec    = QPushButton("🎤  RECORD"); self.b_rec.setStyleSheet(BTN_PRIMARY)
        self.b_rerec  = QPushButton("↺  Re-record"); self.b_rerec.setStyleSheet(BTN_AMBER)
        self.b_skip   = QPushButton("Skip →")
        self.b_rec  .clicked.connect(self._start_record)
        self.b_rerec.clicked.connect(self._rereceord)
        self.b_skip .clicked.connect(self._skip_phrase)
        rec_row.addWidget(self.b_rec)
        rec_row.addWidget(self.b_rerec)
        rec_row.addWidget(self.b_skip)
        right.addLayout(rec_row)

        nav_row = QHBoxLayout(); nav_row.setSpacing(6)
        self.b_prev = QPushButton("◀  Prev")
        self.b_next = QPushButton("Next  ▶"); self.b_next.setStyleSheet(BTN_TEAL)
        self.b_prev.clicked.connect(self._go_prev)
        self.b_next.clicked.connect(self._go_next)
        nav_row.addWidget(self.b_prev); nav_row.addWidget(self.b_next)
        right.addLayout(nav_row)

        # Start enrollment session button
        self.b_start_enr = QPushButton("Start Enrollment Session")
        self.b_start_enr.setStyleSheet(BTN_SUCCESS)
        self.b_start_enr.clicked.connect(self._start_enrollment_session)
        right.addWidget(self.b_start_enr)

        right.addStretch()
        main_row.addLayout(right)
        el.addLayout(main_row)
        self.tabs.addTab(enr_tab, "① Enrollment")

        # ── TAB 2: Fine-Tune ─────────────────────────────────────────────────
        ft_tab = QWidget(); fl = QVBoxLayout(ft_tab); fl.setSpacing(8)
        ft_hint = QLabel(
            "Fine-tuning adapts Whisper to this clinician's accent using recorded phrases. "
            "Requires a GPU for reasonable speed (~5-15 min). "
            "The adapted model loads automatically on next session start.")
        ft_hint.setWordWrap(True)
        ft_hint.setStyleSheet("font-size:20px;color:" + MUTED + ";")
        fl.addWidget(ft_hint)
        self.b_finetune = QPushButton("② Start Fine-Tuning Whisper (LoRA)")
        self.b_finetune.setStyleSheet(BTN_AMBER)
        self.b_finetune.clicked.connect(self._start_finetune)
        fl.addWidget(self.b_finetune)
        self.ft_prog = QProgressBar(); self.ft_prog.setRange(0,0)
        self.ft_prog.setVisible(False); fl.addWidget(self.ft_prog)
        self.log = QListWidget()
        self.log.setStyleSheet(
            "font-family:Consolas,monospace;font-size:18px;"
            "background:#0d1117;color:#e6edf3;border-radius:4px;")
        fl.addWidget(self.log)
        self.tabs.addTab(ft_tab, "② Fine-Tune")

        # ── Close ────────────────────────────────────────────────────────────
        close_row = QHBoxLayout()
        b_close = QPushButton("Close"); b_close.clicked.connect(self.accept)
        close_row.addStretch(); close_row.addWidget(b_close)
        lay.addLayout(close_row)

        QTimer.singleShot(100, self._refresh_status)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _browse(self):
        try: from PyQt5.QtWidgets import QFileDialog
        except: from PyQt6.QtWidgets import QFileDialog
        p = QFileDialog.getExistingDirectory(self,"Enrollment Folder",self.enr.text() or ".")
        if p: self.enr.setText(p); self._refresh_status()

    def _populate_mics(self):
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels",0) > 0:
                    self.mic_cb.addItem(f"{i}: {info['name'][:45]}", i)
            pa.terminate()
        except: self.mic_cb.addItem("Default mic", None)

    def _enr_dir(self):
        from pathlib import Path
        uid  = self.uid.text().strip().lower().replace(" ","_")
        root = Path(self.enr.text().strip() or "enrollment")
        return root / uid if uid else root

    def _refresh_status(self):
        edir = self._enr_dir()
        recs = list((edir/"recordings").glob("phrase_*.wav"))                if (edir/"recordings").exists() else []
        model = edir / "adapted_model.pt"
        self._load_phrases()
        n = len(self._phrases) if self._phrases else 0
        done = len(recs)
        if model.exists():
            self.status_lbl.setText(f"✓  {done}/{n} recordings  |  adapted_model.pt ready")
            self.status_lbl.setStyleSheet("font-size:19px;color:" + TEAL + ";")
        elif recs:
            self.status_lbl.setText(f"⚠  {done}/{n} recordings — run Fine-Tune to create model")
            self.status_lbl.setStyleSheet("font-size:19px;color:" + AMBER + ";")
        else:
            self.status_lbl.setText(f"No recordings yet — click 'Start Enrollment Session'")
            self.status_lbl.setStyleSheet("font-size:19px;color:" + MUTED + ";")
        self._update_phrase_list()

    def _load_phrases(self):
        if self._phrases is None:
            try:
                import finetune_whisper as FT
                self._phrases = FT.STANDARD_PHRASES
            except: self._phrases = ["go","stop","end","on","off","buccal","lingual"]

    def _start_enrollment_session(self):
        """Populate phrase list and jump to first unrecorded phrase."""
        self._load_phrases()
        self._update_phrase_list()
        # Jump to first missing phrase
        edir = self._enr_dir()
        rec_dir = edir / "recordings"
        for i, _ in enumerate(self._phrases):
            wav = rec_dir / f"phrase_{i:03d}.wav"
            if not wav.exists():
                self.phrase_list.setCurrentRow(i)
                break
        else:
            self.phrase_list.setCurrentRow(0)
        self.tabs.setCurrentIndex(0)

    def _update_phrase_list(self):
        self._load_phrases()
        if not self._phrases: return
        edir = self._enr_dir(); rec_dir = edir / "recordings"
        self.phrase_list.blockSignals(True)
        self.phrase_list.clear()
        done = 0
        for i, phrase in enumerate(self._phrases):
            wav = rec_dir / f"phrase_{i:03d}.wav"
            exists = wav.exists()
            if exists: done += 1
            icon = "✓" if exists else "○"
            col  = TEAL if exists else MUTED
            item = QListWidgetItem(f"{icon}  {phrase}")
            item.setForeground(QColor(col))
            self.phrase_list.addItem(item)
        if self._cur_idx < self.phrase_list.count():
            self.phrase_list.setCurrentRow(self._cur_idx)
        self.phrase_list.blockSignals(False)
        n = len(self._phrases)
        self.enr_prog.setRange(0, n); self.enr_prog.setValue(done)
        self.enr_prog_lbl.setText(f"{done} / {n} recorded")

    def _on_phrase_sel(self, row):
        if row < 0 or not self._phrases or row >= len(self._phrases): return
        self._cur_idx = row
        phrase = self._phrases[row]
        self.phrase_lbl.setText(f'"{phrase}"')
        edir = self._enr_dir()
        wav  = edir / "recordings" / f"phrase_{row:03d}.wav"
        if wav.exists():
            self.rec_status.setText("Already recorded — click Re-record to redo")
            self.rec_status.setStyleSheet("font-size:22px;color:" + TEAL + ";")
            self.b_rec.setEnabled(False)
            self.b_rerec.setEnabled(True)
        else:
            self.rec_status.setText("Click RECORD then speak clearly")
            self.rec_status.setStyleSheet("font-size:22px;color:" + MUTED + ";")
            self.b_rec.setEnabled(True)
            self.b_rerec.setEnabled(False)

    def _start_record(self):
        if self._recording: return
        import pyaudio, wave, io
        phrase = self._phrases[self._cur_idx]
        n_words = len(phrase.split())
        self._rec_duration = 2.0 if n_words <= 2 else 3.5
        self._rec_elapsed  = 0.0
        self._rec_frames   = []
        CHUNK = 1024; RATE = 16000
        mic_idx = self.mic_cb.currentData()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16, channels=1, rate=RATE,
            input=True, input_device_index=mic_idx,
            frames_per_buffer=CHUNK)
        self._CHUNK = CHUNK; self._RATE = RATE
        self._recording = True
        self.b_rec.setEnabled(False); self.b_rerec.setEnabled(False)
        self.b_prev.setEnabled(False); self.b_next.setEnabled(False)
        self.rec_status.setText(f"🎤  Recording…  {self._rec_duration:.0f}s")
        self.rec_status.setStyleSheet("font-size:22px;font-weight:700;color:" + RED + ";")
        self._rec_timer.start(50)   # 50 ms ticks

    def _on_rec_tick(self):
        """Called every 50 ms during recording — reads audio chunk and updates VU."""
        if not self._recording: return
        import struct, math
        try:
            data = self._stream.read(self._CHUNK, exception_on_overflow=False)
            self._rec_frames.append(data)
            # VU meter from RMS
            shorts = struct.unpack(f"{len(data)//2}h", data)
            rms    = math.sqrt(sum(s*s for s in shorts) / max(len(shorts),1))
            vu     = min(100, int(rms / 327))  # 32767 max → 100
            self.vu_bar.setValue(vu)
        except: pass
        self._rec_elapsed += 0.05
        remain = max(0, self._rec_duration - self._rec_elapsed)
        self.rec_status.setText(f"🎤  Recording…  {remain:.1f}s")
        if self._rec_elapsed >= self._rec_duration:
            self._stop_record()

    def _stop_record(self):
        self._rec_timer.stop()
        self._recording = False
        import wave, io, pyaudio
        self._stream.stop_stream(); self._stream.close()
        self.vu_bar.setValue(0)
        # Save WAV
        edir    = self._enr_dir()
        rec_dir = edir / "recordings"; rec_dir.mkdir(parents=True, exist_ok=True)
        wav_path= rec_dir / f"phrase_{self._cur_idx:03d}.wav"
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self._RATE)
            wf.writeframes(b''.join(self._rec_frames))
        wav_path.write_bytes(buf.getvalue())
        self._pa.terminate(); self._pa = None
        self.rec_status.setText(f"✓  Saved!  Moving to next…")
        self.rec_status.setStyleSheet("font-size:22px;color:" + TEAL + ";")
        self.b_prev.setEnabled(True); self.b_next.setEnabled(True)
        self._update_phrase_list()
        # Auto-advance to next unrecorded phrase
        QTimer.singleShot(600, self._auto_advance)

    def _auto_advance(self):
        edir = self._enr_dir(); rec_dir = edir / "recordings"
        # Find next unrecorded phrase after current
        for i in range(self._cur_idx+1, len(self._phrases)):
            if not (rec_dir / f"phrase_{i:03d}.wav").exists():
                self.phrase_list.setCurrentRow(i)
                return
        # All done from here — check from start
        for i in range(len(self._phrases)):
            if not (rec_dir / f"phrase_{i:03d}.wav").exists():
                self.phrase_list.setCurrentRow(i)
                return
        # All recorded!
        self.rec_status.setText("🎉  All phrases recorded! Go to Fine-Tune tab.")
        self.rec_status.setStyleSheet("font-size:22px;font-weight:700;color:" + TEAL + ";")

    def _rereceord(self):
        """Delete existing recording and start fresh."""
        edir    = self._enr_dir()
        wav     = edir / "recordings" / f"phrase_{self._cur_idx:03d}.wav"
        if wav.exists(): wav.unlink()
        self.b_rec.setEnabled(True); self.b_rerec.setEnabled(False)
        self.rec_status.setText("Re-recording — click RECORD")
        self._update_phrase_list()

    def _skip_phrase(self):
        self._go_next()

    def _go_prev(self):
        r = max(0, self._cur_idx - 1)
        self.phrase_list.setCurrentRow(r)

    def _go_next(self):
        r = min(len(self._phrases)-1, self._cur_idx + 1) if self._phrases else 0
        self.phrase_list.setCurrentRow(r)

    # ── Fine-tune ────────────────────────────────────────────────────────────

    def _start_finetune(self):
        if self._thread and self._thread.isRunning():
            self._ft_log("Already running…", color=AMBER); return
        self.log.clear(); self.ft_prog.setVisible(True)
        self.b_finetune.setEnabled(False)
        uid = self.uid.text().strip() or None
        self._thread = VoiceTuneThread(
            mode           = "finetune",
            enrollment_dir = self.enr.text().strip() or "enrollment",
            user           = uid,
            whisper_model  = self.wm.currentText(),
            mic_device     = self.mic_cb.currentData(),
        )
        self._thread.log_line   .connect(self._ft_log)
        self._thread.finished_ok.connect(self._ft_done)
        self._thread.error_sig  .connect(self._ft_err)
        self._thread.start()

    def _ft_log(self, text, color=None):
        item = QListWidgetItem(text)
        if color: item.setForeground(QColor(color))
        self.log.addItem(item); self.log.scrollToBottom()

    def _ft_done(self, msg):
        self.ft_prog.setVisible(False); self.b_finetune.setEnabled(True)
        self._ft_log("✓  " + msg, color=TEAL); self._refresh_status()

    def _ft_err(self, msg):
        self.ft_prog.setVisible(False); self.b_finetune.setEnabled(True)
        self._ft_log("✗  " + msg, color=RED)

    def _refresh_status(self):
        edir = self._enr_dir()
        recs = list((edir/"recordings").glob("phrase_*.wav"))                if (edir/"recordings").exists() else []
        model = edir / "adapted_model.pt"
        self._load_phrases()
        n = len(self._phrases) if self._phrases else 0
        done = len(recs)
        if model.exists():
            self.status_lbl.setText(f"✓  {done}/{n} recordings  |  adapted_model.pt ready")
            self.status_lbl.setStyleSheet("font-size:19px;color:" + TEAL + ";")
        elif recs:
            self.status_lbl.setText(f"⚠  {done}/{n} recordings — run Fine-Tune to create model")
            self.status_lbl.setStyleSheet("font-size:19px;color:" + AMBER + ";")
        else:
            self.status_lbl.setText("No recordings yet — click 'Start Enrollment Session'")
            self.status_lbl.setStyleSheet("font-size:19px;color:" + MUTED + ";")
        self._update_phrase_list()


# =============================================================================
#  SettingsDialog
# =============================================================================

class SettingsDialog(QDialog):
    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings"); self.setMinimumWidth(460); self.setStyleSheet(STYLE)
        tabs = QTabWidget()

        # Stitching tab
        sw = QWidget(); sf = QFormLayout(sw); sf.setSpacing(12)
        self.qs     = QDoubleSpinBox(); self.qs.setRange(0,1);   self.qs.setSingleStep(0.01); self.qs.setValue(args.qs_thresh)
        self.tau    = QDoubleSpinBox(); self.tau.setRange(.01,2); self.tau.setSingleStep(0.01); self.tau.setValue(args.tau)
        self.sign   = QComboBox();      self.sign.addItems(["-1","+1"]); self.sign.setCurrentText(str(int(args.imu_sign)))
        self.iou    = QDoubleSpinBox(); self.iou.setRange(0,1);  self.iou.setSingleStep(0.01); self.iou.setValue(args.iou_thresh)
        self.jump   = QDoubleSpinBox(); self.jump.setRange(1,100);self.jump.setSingleStep(.5); self.jump.setValue(args.jump_thresh)
        self.bidir  = QCheckBox("Bidirectional sweep"); self.bidir.setChecked(args.bidirectional)
        for lbl, w in [("Quality threshold (--qs_thresh)", self.qs),
                        ("Filter tau (--tau)", self.tau),
                        ("IMU sign (--imu_sign)", self.sign),
                        ("IOU threshold", self.iou),
                        ("Jump threshold px", self.jump),
                        ("", self.bidir)]:
            sf.addRow(lbl, w)
        tabs.addTab(sw, "Stitching")

        # Charting tab
        cw = QWidget(); cf = QFormLayout(cw); cf.setSpacing(12)
        self.jit = QDoubleSpinBox(); self.jit.setRange(50, 2000)
        self.jit.setSuffix(" µm"); self.jit.setSingleStep(50)
        self.jit.setValue(getattr(args,'jitter_um', 100.0))
        self.mfr = QSpinBox();       self.mfr.setRange(3,50);                               self.mfr.setValue(args.min_frames)
        cf.addRow("Jitter threshold (--jitter_um)", self.jit)
        cf.addRow("Min stable frames (--min_frames)", self.mfr)
        tabs.addTab(cw, "Charting")

        # Voice tab
        vw = QWidget(); vf = QFormLayout(vw); vf.setSpacing(12)
        self.wm   = QComboBox(); self.wm.addItems(["tiny","base","small","medium"]); self.wm.setCurrentText(args.whisper_model)
        self.vdev = QSpinBox();  self.vdev.setRange(0,20); self.vdev.setValue(getattr(args,'voice_device',None) or 0)
        self.fuzz = QDoubleSpinBox(); self.fuzz.setRange(.3,.9); self.fuzz.setSingleStep(.05); self.fuzz.setValue(0.60)
        self.uid  = QLineEdit(args.user or "Reza"); self.uid.setPlaceholderText("Operator name")
        self.nov  = QCheckBox("Disable voice"); self.nov.setChecked(args.no_voice)
        for lbl, w in [("Whisper model (--whisper_model)", self.wm),
                        ("Mic device (--voice_device)", self.vdev),
                        ("Fuzzy threshold", self.fuzz),
                        ("User / adapted model (--user)", self.uid),
                        ("", self.nov)]:
            vf.addRow(lbl, w)
        tabs.addTab(vw, "Voice")

        # Output tab
        ow = QWidget(); of = QFormLayout(ow); of.setSpacing(12)
        out_row = QHBoxLayout(); out_row.setSpacing(6)
        self.out_edit = QLineEdit(args.out)
        btn_out = QPushButton("Browse"); btn_out.setFixedWidth(120)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit); out_row.addWidget(btn_out)
        of.addRow("Output folder:", out_row)
        note = QLabel("Stitching: <out>/stitching/\nCharting:  <out>/charting/")
        note.setStyleSheet("font-size:24px;color:" + MUTED + ";")
        of.addRow("", note)
        tabs.addTab(ow, "Output")
        btns = QHBoxLayout()
        ok = QPushButton("Apply"); ok.setStyleSheet(BTN_TEAL); ok.clicked.connect(self.accept)
        cl = QPushButton("Close"); cl.clicked.connect(self.reject)
        btns.addStretch(); btns.addWidget(ok); btns.addWidget(cl)
        lay = QVBoxLayout(self); lay.addWidget(tabs); lay.addLayout(btns)

    def _browse_out(self):
        from PyQt5.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, "Select Output Folder",
                                                 self.out_edit.text() or ".")
        if path: self.out_edit.setText(path)

    def values(self):
        return dict(qs_thresh=self.qs.value(), tau=self.tau.value(),
                    imu_sign=float(self.sign.currentText()),
                    iou_thresh=self.iou.value(), jump_thresh=self.jump.value(),
                    bidirectional=self.bidir.isChecked(),
                    jitter_um=self.jit.value(), min_frames=self.mfr.value(),
                    whisper_model=self.wm.currentText(), voice_device=self.vdev.value(),
                    user=self.uid.text().strip() or None, no_voice=self.nov.isChecked(),
                    out=self.out_edit.text().strip())


# =============================================================================
#  MainWindow
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self, args, pano_system, chart_annotator, chart_system,
                 capture, imu, voice, stitch_out, chart_out):
        super().__init__()
        self.args            = args
        self.pano_system     = pano_system
        self.chart_annotator = chart_annotator
        self.chart_system    = chart_system
        self.capture         = capture
        self.imu             = imu
        self.voice           = voice
        self.stitch_out      = stitch_out
        self.chart_out       = chart_out
        self._mode           = "stitching"
        self._cstate         = "inactive"   # chart state machine
        self._surf           = "B"
        self._saved          = {}  # {tooth_num: set of surfaces done}
        self._stitch_tooth   = None  # tooth being imaged during stitching
        self._rec_secs       = 0
        self._rec_tmr        = QTimer(); self._rec_tmr.timeout.connect(self._tick)

        self.setWindowTitle("DentWise")
        self.resize(1600, 1000)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(STYLE)
        self._build_ui()
        self._build_keys()
        self._start_threads()
        # Persistent design credit in status bar
        _credit_lbl = QLabel("  DentWise  |  Designed by Reza Amjadian  ")
        _credit_lbl.setStyleSheet(
            "color:" + TEAL + ";font-size:19px;font-weight:700;letter-spacing:0.05em;")
        self.statusBar().addPermanentWidget(_credit_lbl)
        self.statusBar().showMessage("Ready -- press 1 or say 'stitching' to begin")

    # -------------------------------------------------------------------------

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QHBoxLayout(cw); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # QSplitter makes all panels user-resizable by dragging the dividers
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            "QSplitter::handle{background:" + BORDER + ";}"
            "QSplitter::handle:hover{background:" + TEAL + ";}"
        )

        # LEFT column (controls + stats) -- initial width 260, resizable
        left = QFrame()
        left.setStyleSheet("background:" + PANEL + ";")
        left.setMinimumWidth(180)
        lv = QVBoxLayout(left); lv.setContentsMargins(10,10,10,10); lv.setSpacing(8)

        # Mode toggle
        mr = QHBoxLayout(); mr.setSpacing(4)
        self.bms = QPushButton("STITCHING"); self.bms.setStyleSheet(BTN_PRIMARY)
        self.bmc = QPushButton("CHARTING")
        self.bms.clicked.connect(lambda: self._mode_set("stitching"))
        self.bmc.clicked.connect(lambda: self._mode_set("charting"))
        mr.addWidget(self.bms); mr.addWidget(self.bmc); lv.addLayout(mr)

        # Stitching group
        self.sg = QGroupBox("Stitching"); sv = QVBoxLayout(self.sg); sv.setSpacing(5)
        self.b_start = QPushButton("START"); self.b_start.setStyleSheet(BTN_PRIMARY)
        self.b_stop  = QPushButton("STOP + SAVE"); self.b_stop.setStyleSheet(BTN_DANGER)
        self.b_start.clicked.connect(self._stitch_start)
        self.b_stop.clicked.connect(self._stitch_stop)
        sv.addWidget(self.b_start); sv.addWidget(self.b_stop)
        # Surface selector — used in both stitching (labelling) and charting (scanning)
        sr0 = QHBoxLayout(); sr0.setSpacing(4)
        self.b_sb = QPushButton("Buccal (B)"); self.b_sb.setStyleSheet(BTN_PRIMARY)
        self.b_sl = QPushButton("Lingual (L)")
        self.b_sb.clicked.connect(lambda: self._surf_set("B"))
        self.b_sl.clicked.connect(lambda: self._surf_set("L"))
        sr0.addWidget(self.b_sb); sr0.addWidget(self.b_sl)
        sv.addLayout(sr0)
        # Removed: HOLD, GO, MARK, BAD, DROP, DISCARD buttons
        r1 = QHBoxLayout(); r1.setSpacing(4)
        self.b_pause = QPushButton("PAUSE")
        self.b_pause.setCheckable(True)
        self.b_pause.setStyleSheet(BTN_AMBER)
        self.b_pause.clicked.connect(self._cmd_pause)
        r1.addWidget(self.b_pause)
        self.b_imu = QPushButton("IMU VIEW")
        self.b_imu.clicked.connect(self._cmd_view); r1.addWidget(self.b_imu)
        sv.addLayout(r1)
        # Calibrate IMU button — press C twice on keyboard or click this
        self.b_cal = QPushButton("Calibrate IMU  (C)")
        self.b_cal.setStyleSheet(BTN_TEAL)
        self.b_cal.setToolTip(
            "Step 1: Click to mark start.\n"
            "Step 2: Sweep probe 10-15 mm along arch.\n"
            "Step 3: Click again — scale is set automatically.")
        self.b_cal.clicked.connect(self._cmd_calibrate)
        sv.addWidget(self.b_cal)
        self.cal_lbl = QLabel("IMU not calibrated — click Calibrate IMU before scanning")
        self.cal_lbl.setStyleSheet("font-size:18px;color:" + AMBER + ";")
        self.cal_lbl.setWordWrap(True)
        sv.addWidget(self.cal_lbl)
        # Voice toggle button — disables/enables voice without stopping the thread
        self.b_voice_toggle = QPushButton("Voice ON")
        self.b_voice_toggle.setStyleSheet(BTN_TEAL)
        self.b_voice_toggle.setCheckable(True)
        self.b_voice_toggle.clicked.connect(self._toggle_voice_commands)
        self.b_voice_toggle.setToolTip("Toggle voice command recognition on/off")
        sv.addWidget(self.b_voice_toggle)
        sv.addWidget(QLabel("Voice: ON  OFF  SAVE  PAUSE  RESUME  VIEW | say charting to switch"))
        self._style_help(sv.itemAt(sv.count()-1).widget())
        lv.addWidget(self.sg)

        # Charting group
        self.cg = QGroupBox("Charting"); cv2v = QVBoxLayout(self.cg); cv2v.setSpacing(5)
        self.b_go = QPushButton("ACTIVATE"); self.b_go.setStyleSheet(BTN_SUCCESS)
        self.b_go.clicked.connect(self._chart_go); cv2v.addWidget(self.b_go)

        # Surface selector for charting — Buccal / Lingual
        # Shows which surface will be scanned when a tooth button is clicked
        csurf_lbl = QLabel("Surface:"); csurf_lbl.setStyleSheet("font-size:22px;color:" + MUTED + ";")
        cv2v.addWidget(csurf_lbl)
        csurf_row = QHBoxLayout(); csurf_row.setSpacing(4)
        self.b_chart_buccal  = QPushButton("Buccal (B)")
        self.b_chart_lingual = QPushButton("Lingual (L)")
        self.b_chart_buccal .setStyleSheet(BTN_PRIMARY)   # default selected
        self.b_chart_lingual.setStyleSheet("")
        self.b_chart_buccal .clicked.connect(lambda: self._chart_surf_set("B"))
        self.b_chart_lingual.clicked.connect(lambda: self._chart_surf_set("L"))
        csurf_row.addWidget(self.b_chart_buccal)
        csurf_row.addWidget(self.b_chart_lingual)
        cv2v.addLayout(csurf_row)
        self._chart_surf_btns = {
            "B": self.b_chart_buccal,
            "L": self.b_chart_lingual,
        }
        self._chart_surf = "B"   # default surface for charting

        # Current tooth display
        self.t_lbl = QLabel("--"); self.t_lbl.setAlignment(Qt.AlignCenter)
        self.t_lbl.setStyleSheet("font-size:44px;font-weight:600;color:" + TEAL + ";padding:4px 0;")
        cv2v.addWidget(self.t_lbl)

        cr = QHBoxLayout(); cr.setSpacing(4)
        self.b_cs = QPushButton("STOP"); self.b_cs.setStyleSheet(BTN_AMBER)
        self.b_ce = QPushButton("END");  self.b_ce.setStyleSheet(BTN_DANGER)
        self.b_cs.clicked.connect(self._chart_stop); self.b_ce.clicked.connect(self._chart_end)
        cr.addWidget(self.b_cs); cr.addWidget(self.b_ce); cv2v.addLayout(cr)
        cv2v.addWidget(QLabel("Voice: GO  [14B / 3L]  STOP  END | say stitching to switch"))
        self._style_help(cv2v.itemAt(cv2v.count()-1).widget())
        self.cg.hide(); lv.addWidget(self.cg)

        # Tooth grid
        self.tg = QGroupBox("Tooth Selector")
        tgl = QGridLayout(self.tg); tgl.setSpacing(6); tgl.setContentsMargins(8,8,8,8)
        self._tbtns = {}
        tbase = "QPushButton{background:" + SURFACE + ";border:1px solid " + BORDER + ";border-radius:6px;font-size:22px;font-weight:500;padding:4px;}"
        for i in range(1, 33):
            b = QPushButton(str(i)); b.setFixedSize(48, 48); b.setStyleSheet(tbase)
            b.clicked.connect(lambda _, n=i: self._tooth_sel(n))
            tgl.addWidget(b, (i-1)//8, (i-1)%8); self._tbtns[i] = b
        lv.addWidget(self.tg)  # visible in both modes

        lv.addStretch()

        # Session stats
        stg = QGroupBox("Session"); stv = QVBoxLayout(stg); stv.setSpacing(2)
        self._slbls = {}
        for k in ["Mode","State","Session","Frames","Keyframes","Memory","Jitter X","Gate","IMU","Voice"]:
            rw = QHBoxLayout(); rw.setSpacing(0)
            kl = QLabel(k); kl.setStyleSheet("color:" + MUTED + ";font-size:20px;")
            vl = QLabel("--"); vl.setAlignment(Qt.AlignRight)
            vl.setStyleSheet("font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:500;")
            rw.addWidget(kl); rw.addWidget(vl); stv.addLayout(rw)
            self._slbls[k] = vl
        self._slbls["Mode"].setText("Stitching"); self._slbls["State"].setText("Idle")
        self._slbls["Voice"].setText("ON"); lv.addWidget(stg)

        # Voice log
        lg = QGroupBox("Voice Log"); lgv = QVBoxLayout(lg)
        self.vlog = QListWidget(); self.vlog.setMaximumHeight(100); lgv.addWidget(self.vlog)
        lv.addWidget(lg)

        btn_cfg  = QPushButton("Settings"); btn_cfg.clicked.connect(self._open_settings)
        btn_voice= QPushButton("Voice Adapt"); btn_voice.setStyleSheet(BTN_PRIMARY)
        btn_voice.clicked.connect(self._open_voice_tune)
        lv.addWidget(btn_cfg)

        # RIGHT column -- must be declared before splitter.addWidget(right)
        right = QWidget(); right.setStyleSheet("background:" + DARK_BG + ";")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([280, 1200])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

        rv = QVBoxLayout(right); rv.setContentsMargins(0,0,0,0); rv.setSpacing(0)

        # Top bar
        tb_frame = QFrame(); tb_frame.setFixedHeight(32)
        tb_frame.setStyleSheet("background:" + PANEL + ";border-bottom:1px solid " + BORDER + ";")
        tb = QHBoxLayout(tb_frame); tb.setContentsMargins(10,0,10,0)
        self.lbdg = QLabel("IDLE")
        self.lbdg.setStyleSheet("font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:600;color:" + MUTED + ";background:" + SURFACE + ";border-radius:10px;padding:2px 8px;")
        self.linfo = QLabel("HD60X — DentScan ready"); self.linfo.setStyleSheet("font-size:20px;color:" + MUTED + ";font-family:Consolas,SF Mono,Cascadia Code,monospace;")
        self.l_fps = QLabel("-- fps");            self.l_fps.setStyleSheet("font-size:20px;color:" + MUTED + ";font-family:Consolas,SF Mono,Cascadia Code,monospace;")
        self.l_tmr = QLabel("00:00");             self.l_tmr.setStyleSheet("font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:22px;color:" + MUTED + ";")
        tb.addWidget(self.lbdg); tb.addWidget(self.linfo)
        tb.addStretch(); tb.addWidget(self.l_tmr); tb.addWidget(self.l_fps)
        rv.addWidget(tb_frame)

        # Square live frame label -- enforced square in resizeEvent
        self.flabel = QLabel()
        self.flabel.setAlignment(Qt.AlignCenter)
        self.flabel.setStyleSheet("background:#060d0a;")
        self.flabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.flabel.setMinimumSize(800, 800)
        rv.addWidget(self.flabel, stretch=1)

        # Quality bars
        qf = QFrame(); qf.setFixedHeight(28)
        qf.setStyleSheet("background:" + PANEL + ";border-top:1px solid " + BORDER + ";")
        qbl = QHBoxLayout(qf); qbl.setContentsMargins(10,0,10,0); qbl.setSpacing(8)
        bar_ss_t = "QProgressBar{background:" + SURFACE + ";border:none;border-radius:2px;} QProgressBar::chunk{background:" + TEAL + ";border-radius:2px;}"
        bar_ss_b = "QProgressBar{background:" + SURFACE + ";border:none;border-radius:2px;} QProgressBar::chunk{background:" + BLUE + ";border-radius:2px;}"
        for lbl, bss, attr in [("B1", bar_ss_t, "qbar"), ("Gate", bar_ss_b, "gbar")]:
            ql = QLabel(lbl); ql.setStyleSheet("font-size:20px;color:" + MUTED + ";min-width:24px;")
            bar = QProgressBar(); bar.setRange(0,100); bar.setFixedHeight(4)
            bar.setStyleSheet(bss); bar.setTextVisible(False)
            vl = QLabel("--"); vl.setStyleSheet("font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:24px;font-weight:600;color:" + TEXT + ";min-width:44px;")
            setattr(self, attr, bar); setattr(self, attr + "_lbl", vl)
            qbl.addWidget(ql); qbl.addWidget(bar); qbl.addWidget(vl)
        rv.addWidget(qf)

        # IMU strip
        imu_f = QFrame(); imu_f.setFixedHeight(26)
        imu_f.setStyleSheet("background:" + PANEL + ";border-top:1px solid " + BORDER + ";")
        il = QHBoxLayout(imu_f); il.setContentsMargins(10,0,10,0); il.setSpacing(14)
        self._idot = QLabel("o"); self._idot.setStyleSheet("color:" + HINT + ";font-size:16px;")
        il.addWidget(self._idot)
        self._ifields = {}
        for k in ["IMU","Hz","alpha","Corr","Yaw","Jitter"]:
            lb = QLabel(k + " --"); lb.setStyleSheet("font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:24px;color:" + MUTED + ";")
            il.addWidget(lb); self._ifields[k] = lb
        il.addStretch(); rv.addWidget(imu_f)

        # right widget now owned by splitter above

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Inference window fills the right panel freely -- no square constraint.
        # The frame is upscaled to fit whatever size the user drags it to.
        self.flabel.setMaximumWidth(16777215)
        self.flabel.setMaximumHeight(16777215)

    def _build_keys(self):
        QShortcut(QKeySequence("1"), self, self._key1)
        QShortcut(QKeySequence("2"), self, self._key2)
        QShortcut(QKeySequence("R"), self, self._cmd_drop)
        QShortcut(QKeySequence("S"), self, self._screenshot)

    def _start_threads(self):
        self.inf = InferenceThread(self.capture, self.pano_system,
                                   self.chart_annotator, self.chart_system, self.voice)
        self.inf.frame_ready.connect(self._on_frame)
        self.inf.metrics_ready.connect(self._on_metrics)
        self.inf.error.connect(lambda e: self.statusBar().showMessage("Error: " + e))
        self.inf.start()
        if self.voice:
            self.vt = VoiceThread(self.voice)
            self.vt.command_received.connect(self._on_vcmd)
            self.vt.mode_changed.connect(self._mode_set)
            self.vt.log_entry.connect(self._vlog)
            self.vt.start()

    # ---- slots ---------------------------------------------------------------

    def _on_frame(self, img):
        # 2x nearest-neighbor upscale (each pixel drawn 2x2, no interpolation blur)
        w2 = img.width()  * 2
        h2 = img.height() * 2
        fast = (Qt.TransformationMode.FastTransformation
                if hasattr(Qt, "TransformationMode") else Qt.FastTransformation)
        img2 = img.scaled(w2, h2, Qt.KeepAspectRatio, fast)
        # Fill the full label size while keeping aspect ratio
        lw = max(self.flabel.width(),  100)
        lh = max(self.flabel.height(), 100)
        px = QPixmap.fromImage(img2).scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.flabel.setPixmap(px)

    def _on_metrics(self, m):
        if "error" in m: return
        q = m.get("quality", 0); g = m.get("gate", q)
        self.qbar.setValue(int(q * 100)); self.qbar_lbl.setText(f"{q:.3f}")
        self.gbar.setValue(int(g * 100)); self.gbar_lbl.setText(f"{g:.2f}")
        self.l_fps.setText(f"{m.get('fps',0):.0f} fps")
        if m.get("mode") == "stitching":
            self._slbls["Frames"].setText(str(m.get("frames", 0)))
            self._slbls["Keyframes"].setText(str(m.get("kf", 0)))
            self._slbls["Memory"].setText(f"{m.get('mem_mb',0):.1f} MB")
            j = m.get("jitter_x", 0)
            self._slbls["Jitter X"].setText(f"{j:.1f} um" if j >= 0 else "-- um")
            iv = m.get("imu_valid", False)
            self._idot.setStyleSheet("color:" + (TEAL if iv else HINT) + ";font-size:16px;")
            for k, v in [("IMU", "OK" if iv else "--"),
                         ("Hz",  f"{m.get('imu_hz',0):.1f}" if iv else "--"),
                         ("alpha", f"{m.get('imu_alpha',0):.3f}"),
                         ("Yaw", f"{m.get('imu_yaw',0):.1f}d")]:
                self._ifields[k].setText(k + " " + v)
        elif m.get("mode") == "charting":
            tooth = m.get("tooth", "--")
            self.t_lbl.setText(tooth)
            self._slbls["Frames"].setText(f"{m.get('fps',0):.0f} fps")
            self._slbls["Keyframes"].setText(str(sum(len(v) for v in self._saved.values())))
            meas = m.get("meas", {})
            if meas:
                parts = [f"{k}={v:.2f}mm" for k, v in meas.items() if isinstance(v, float)]
                if parts: self.statusBar().showMessage("  ".join(parts))

    def _on_vcmd(self, ctype, val):
        print(f"[GUI/_on_vcmd] ctype={ctype!r} val={val!r} cstate={self._cstate!r} "
              f"mode={self._mode!r} current_tooth={getattr(self.chart_system,'current_tooth',None)!r}")
        if ctype == "pano":
            {"on": self._stitch_start, "off": self._stitch_stop,
             "save": self._stitch_stop, "drop": self._cmd_drop,
             "hold": self._cmd_hold, "go": self._cmd_go,
             "mark": self._cmd_mark, "bad": self._cmd_bad,
             "view": self._cmd_view}.get(val, lambda: None)()
        elif ctype == "action":
            print(f"[GUI] chart action: {val!r}  cstate={self._cstate!r}  "
                  f"scanning={getattr(self.chart_system,'scanning_active',None)}  "
                  f"tooth={getattr(self.chart_system,'current_tooth',None)}")
            {"go": self._chart_go, "stop": self._chart_stop,
             "end": self._chart_end}.get(val, lambda: None)()
        elif ctype == "tooth":
            n = int(re.sub(r"[^0-9]", "", val) or "0")
            s = re.sub(r"[^BL]", "", val.upper())
            print(f"[GUI] tooth cmd: n={n} s={s!r} cstate={self._cstate!r}")
            if s: self._surf_set(s)
            if 1 <= n <= 32: self._tooth_sel(n, surf=s if s else None)

    def _vlog(self, raw, cmd):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.vlog.insertItem(0, QListWidgetItem(ts + "  " + raw))
        while self.vlog.count() > 15: self.vlog.takeItem(self.vlog.count() - 1)

    # ---- mode ----------------------------------------------------------------

    def _mode_set(self, m):
        self._mode = m; self.inf.set_mode(m)
        if self.voice: self.voice.mode = m
        s = m == "stitching"
        self.sg.setVisible(s); self.cg.setVisible(not s)
        # Tooth grid stays visible in BOTH modes:
        # stitching = label anatomy; charting = select tooth to scan
        self.tg.setVisible(True)
        self.bms.setStyleSheet(BTN_PRIMARY if s else "")
        self.bmc.setStyleSheet(BTN_TEAL if not s else "")
        self._slbls["Mode"].setText("Stitching" if s else "Charting")
        self.statusBar().showMessage("Mode: " + m.upper())

    # ---- stitching -----------------------------------------------------------

    def _stitch_start(self):
        if self.pano_system.recording: return
        # Reset pause state on new recording
        self._chart_paused = False
        self.b_pause.setText("PAUSE"); self.b_pause.setStyleSheet(BTN_AMBER)
        self.b_pause.setChecked(False)
        if hasattr(self, '_inf_thread'):
            self._inf_thread.set_paused(False)

        self.pano_system._panorama_frozen = False

        # Set tooth label NOW (from current selection) so _do_start()
        # creates the debug/ subfolder with the correct tooth name.
        # If user changes tooth after START, _stitch_stop updates it too.
        if self._stitch_tooth:
            self.pano_system._session_tooth_label = f"{self._stitch_tooth}{self._surf}"
        else:
            self.pano_system._session_tooth_label = None
        self.pano_system._do_start(0., self.imu.get_latest())
        self.b_start.setText("RECORDING"); self.b_start.setStyleSheet(BTN_REC)
        self.lbdg.setText("REC"); self.lbdg.setStyleSheet(
            "font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:600;color:" + RED + ";background:#3a1a1a;border-radius:10px;padding:2px 8px;")
        self._slbls["State"].setText("Recording"); self._rec_secs = 0; self._rec_tmr.start(1000)
        self.statusBar().showMessage("Recording stitch -- press 2 or say OFF to stop")
        self._vlog("key/voice", "ON")

    def _stitch_stop(self):
        if not self.pano_system.recording: return
        self._rec_tmr.stop()
        # Set tooth label NOW using whatever tooth is currently selected
        # This works whether user clicked tooth before OR after pressing START
        if self._stitch_tooth:
            self.pano_system._session_tooth_label = f"{self._stitch_tooth}{self._surf}"
        else:
            self.pano_system._session_tooth_label = None
        sharp = self.pano_system._do_stop_save()
        self.b_start.setText("START"); self.b_start.setStyleSheet(BTN_PRIMARY)
        self.lbdg.setText("SAVED"); self.lbdg.setStyleSheet(
            "font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:600;color:" + TEAL + ";background:" + TEAL_BG + ";border-radius:10px;padding:2px 8px;")
        self._slbls["State"].setText("Saved")
        n = self.pano_system.stats.get("frames_good", 0)
        sid = max(1, self.pano_system._session_id - 1)
        if sharp is None:
            # Too few frames — app is still stable, just show warning
            self.lbdg.setText("NO DATA"); self.lbdg.setStyleSheet(
                "font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;"
                "font-weight:600;color:" + AMBER + ";background:" + TEAL_BG +
                ";border-radius:10px;padding:2px 8px;")
            self.statusBar().showMessage(
                "⚠ Too few frames — hold probe in contact for at least 2 seconds before stopping")
            self._vlog("key/voice", "OFF -> too few frames, not saved")
        else:
            lbl = getattr(self.pano_system, '_session_tooth_label', None)
            folder_name = lbl if lbl else f"stitch_{sid:03d}"
            self._vlog("key/voice", f"OFF -> saved {folder_name}")
            self.statusBar().showMessage(
                f"Saved {folder_name} — {n} frames → {self.stitch_out}")
            self._show_pano_qt(sharp, f"Panorama — {folder_name}")
        QTimer.singleShot(3000, lambda: (
            self.lbdg.setText("IDLE"),
            self.lbdg.setStyleSheet("font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:600;color:" + MUTED + ";background:" + SURFACE + ";border-radius:10px;padding:2px 8px;")))

    def _show_pano_qt(self, pano_rgb, title="Panorama"):
        """Show panorama in a Qt window -- works for every session."""
        import numpy as np
        if pano_rgb is None: return
        # pano_rgb is float32 [0,1] or uint8
        arr = pano_rgb
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        # RGB -> QImage
        if arr is None or arr.size == 0: return
        h, w = arr.shape[:2]
        ch = arr.shape[2] if arr.ndim == 3 else 1
        if ch == 1:
            arr = np.stack([arr]*3, axis=2); ch = 3
        img = QImage(arr.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setStyleSheet("background:#000;")
        lay = QVBoxLayout(dlg); lay.setContentsMargins(4,4,4,4)
        lbl = QLabel(); lbl.setPixmap(QPixmap.fromImage(img).scaled(
            1200, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lbl.setStyleSheet("background:#000;")
        lay.addWidget(lbl)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(dlg.accept)
        close_btn.setStyleSheet(BTN_TEAL); lay.addWidget(close_btn)
        dlg.show()   # non-blocking -- clinician can dismiss when ready

    def _tick(self):
        self._rec_secs += 1; m, s = divmod(self._rec_secs, 60)
        self.l_tmr.setText(f"{m:02d}:{s:02d}")

    def _style_help(self, lbl):
        """Style a voice-command help label."""
        lbl.setStyleSheet(
            "font-size:16px;color:" + HINT + ";padding:4px 2px;")
        lbl.setWordWrap(True)

    def _update_tooth_grid(self, active_n=None):
        """Redraw tooth grid — B/L/both/none per tooth.
        In stitching mode: amber highlight shows which tooth is being imaged."""
        tbase = ("QPushButton{background:" + SURFACE + ";border:1px solid " + BORDER +
                 ";border-radius:6px;font-size:22px;font-weight:500;padding:4px;}")
        for i, b in self._tbtns.items():
            done = self._saved.get(i, set())
            is_stitch_active = (self._mode == "stitching" and i == self._stitch_tooth)
            if is_stitch_active:
                b.setStyleSheet("QPushButton{background:#3a2a10;border:2px solid " + AMBER +
                                ";border-radius:6px;font-size:22px;font-weight:700;"
                                "padding:4px;color:" + AMBER + ";}")
                b.setToolTip("Currently stitching this tooth")
            elif i == active_n:
                sty = BTN_TEAL if self._surf == "L" else BTN_PRIMARY
                b.setStyleSheet(sty.replace("padding:8px 14px", "padding:4px"))
            elif "B" in done and "L" in done:
                b.setStyleSheet("QPushButton{background:#1a3a20;border:2px solid " + GREEN +
                                ";border-radius:6px;font-size:22px;font-weight:700;"
                                "padding:4px;color:" + GREEN + ";}")
                b.setToolTip("Both B+L done")
            elif "B" in done:
                b.setStyleSheet("QPushButton{background:#1a2a40;border:2px solid " + BLUE +
                                ";border-radius:6px;font-size:22px;font-weight:600;"
                                "padding:4px;color:" + BLUE + ";}")
                b.setToolTip("Buccal done")
            elif "L" in done:
                b.setStyleSheet("QPushButton{background:#0a2a1e;border:2px solid " + TEAL +
                                ";border-radius:6px;font-size:22px;font-weight:600;"
                                "padding:4px;color:" + TEAL + ";}")
                b.setToolTip("Lingual done")
            else:
                b.setStyleSheet(tbase); b.setToolTip("")

    def _cmd_pause(self):
        """
        Toggle pause — freezes both stitching and charting.
        Stitching: sets pano_system.voice_paused=True → canvas stops compositing.
        Charting:  sets self._chart_paused=True → InferenceThread skips add_frame().
        Camera keeps running so the live view stays visible.
        """
        # Toggle unified pause state
        currently_paused = getattr(self, '_chart_paused', False)
        new_state = not currently_paused
        self._chart_paused = new_state

        # Apply to stitching
        ps = self.pano_system
        ps.voice_paused = new_state  # always set — harmless when not recording

        self.b_pause.setText("RESUME" if new_state else "PAUSE")
        self.b_pause.setStyleSheet(BTN_TEAL if new_state else BTN_AMBER)
        self.b_pause.setChecked(new_state)

        # Update InferenceThread so charting also freezes
        if hasattr(self, '_inf_thread'):
            self._inf_thread.set_paused(new_state)

        mode_str = self._mode.upper()
        if new_state:
            self.statusBar().showMessage(
                f"PAUSED [{mode_str}] — probe lifted / standby. Click PAUSE again to resume.")
        else:
            self.statusBar().showMessage(
                f"Resumed [{mode_str}] — compositing active")
        self._vlog("btn", "PAUSE" if new_state else "RESUME")

    def _cmd_hold(self):
        if self.pano_system.recording: self.pano_system.voice_paused = True; self.statusBar().showMessage("HOLD -- say GO to resume")
    def _cmd_go(self):
        self.pano_system.voice_paused = False; self.statusBar().showMessage("GO -- resumed")
    def _cmd_mark(self):
        self.pano_system.anchor_marks.append(len(self.pano_system._session_coords)); self.statusBar().showMessage("Anchor marked")
    def _cmd_bad(self):
        ps = self.pano_system
        if ps._bad_start is None:
            ps._bad_start = len(ps._session_coords); self.statusBar().showMessage("Bad start")
        else:
            ps.bad_marks.append((ps._bad_start, len(ps._session_coords))); ps._bad_start = None
            self.statusBar().showMessage("Bad end marked")
    def _cmd_calibrate(self):
        """
        IMU calibration — two-step process:
          Click 1: Mark start position (probe stationary)
          Click 2: Mark end position after 10-15mm sweep → scale computed
        """
        if not hasattr(self, '_cal_state'): self._cal_state = 0
        if self._cal_state == 0:
            try:
                self.pano_system._do_calibrate_start()
            except Exception:
                self.pano_system._cal_start_gm  = None
                self.pano_system._cal_start_imu = None
            self._cal_state = 1
            self.b_cal.setText("▶  Mark END  (sweep done → click here)")
            self.b_cal.setStyleSheet(BTN_AMBER)
            msg = "Step 2: Sweep probe 10-15 mm along arch, then click Mark END"
            self.statusBar().showMessage(msg)
            if hasattr(self, 'cal_lbl'):
                self.cal_lbl.setText("● Calibrating… sweep probe 10-15 mm then click button again")
                self.cal_lbl.setStyleSheet("font-size:18px;color:" + TEAL + ";")
        else:
            scale = None
            try:
                scale = self.pano_system._do_calibrate_end()
            except Exception as e:
                self.statusBar().showMessage(f"Calibration error: {e}")
            self._cal_state = 0
            self.b_cal.setText("Calibrate IMU  (C)")
            self.b_cal.setStyleSheet(BTN_TEAL)
            if scale:
                msg = f"✓ IMU calibrated — scale={scale:.4f} mm/° — ready to scan"
                self.statusBar().showMessage(msg)
                if hasattr(self, 'cal_lbl'):
                    self.cal_lbl.setText(f"✓ Calibrated: {scale:.4f} mm/deg")
                    self.cal_lbl.setStyleSheet("font-size:18px;color:" + TEAL + ";")
            else:
                self.statusBar().showMessage("IMU calibrated")
                if hasattr(self, 'cal_lbl'):
                    self.cal_lbl.setText("✓ IMU calibrated")
                    self.cal_lbl.setStyleSheet("font-size:18px;color:" + TEAL + ";")

    def _cmd_view(self):
        self.pano_system.show_imu = not self.pano_system.show_imu
        self.statusBar().showMessage("IMU overlay " + ("ON" if self.pano_system.show_imu else "OFF"))

    def _toggle_voice_commands(self):
        """Toggle voice command processing on/off while app is running."""
        if not self.voice: return
        disabled = getattr(self.voice, '_voice_disabled', False)
        if disabled:
            self.voice.enable()
            if hasattr(self, 'b_voice_toggle'):
                self.b_voice_toggle.setText("Voice ON")
                self.b_voice_toggle.setStyleSheet(BTN_TEAL)
            self.statusBar().showMessage("Voice commands enabled")
        else:
            self.voice.disable()
            if hasattr(self, 'b_voice_toggle'):
                self.b_voice_toggle.setText("Voice OFF")
                self.b_voice_toggle.setStyleSheet(BTN_AMBER)
            self.statusBar().showMessage("Voice commands disabled")
    def _cmd_drop(self):
        ps = self.pano_system
        if ps.recording:
            ps.recording = False; ps.canvas.reset()
            ps._session_frames.clear(); ps._session_coords.clear(); self._rec_tmr.stop()
            self.b_start.setText("START"); self.b_start.setStyleSheet(BTN_PRIMARY)
            self.lbdg.setText("IDLE"); self.lbdg.setStyleSheet(
                "font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:600;color:" + MUTED + ";background:" + SURFACE + ";border-radius:10px;padding:2px 8px;")
            self.statusBar().showMessage("Discarded")

    # ---- charting ------------------------------------------------------------

    def _chart_go(self):
        if self._cstate == "inactive":
            self._cstate = "waiting"; self.b_go.setText("ACTIVATED"); self.b_go.setStyleSheet(BTN_TEAL)
            self._slbls["State"].setText("Waiting for tooth")
            self.lbdg.setText("CHART"); self.lbdg.setStyleSheet(
                "font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:600;color:" + TEAL + ";background:" + TEAL_BG + ";border-radius:10px;padding:2px 8px;")
            self.statusBar().showMessage("Activated -- tap tooth or say '14 B'")

    def _surf_set(self, s):
        """Surface selector for STITCHING mode."""
        self._surf = s
        self.b_sb.setStyleSheet(BTN_PRIMARY if s == "B" else "")
        self.b_sl.setStyleSheet(BTN_TEAL    if s == "L" else "")

    def _chart_surf_set(self, s):
        """Surface selector for CHARTING mode — B or L only."""
        self._chart_surf = s
        colors = {"B": BTN_PRIMARY, "L": BTN_TEAL}
        for surf, btn in self._chart_surf_btns.items():
            btn.setStyleSheet(colors.get(surf, "") if surf == s else "")
        self.statusBar().showMessage(
            f"Charting surface: {'Buccal' if s=='B' else 'Lingual'}  — tap tooth number to start scan")

    def _tooth_sel(self, n, surf=None):
        """Select tooth n. surf overrides auto-detected surface.
        Stitching mode: uses self._surf (set by stitching surface buttons).
        Charting mode:  uses self._chart_surf (set by charting surface buttons).
        """
        if self._mode == "charting":
            s = surf if surf in ("B","L") else self._chart_surf
        else:
            s = surf if surf in ("B","L") else self._surf
        if self._mode == "stitching":
            tid = str(n) + s
            self.statusBar().showMessage(f"Stitching tooth {tid} region")
            self._vlog("tooth-label", tid)
            self._stitch_tooth = n
            self._update_tooth_grid()
            return
        print(f"[GUI/_tooth_sel] cstate={self._cstate!r} n={n} s={s!r}")
        if self._cstate == "inactive":
            # Voice said tooth directly without GO — auto-activate then scan
            # Clinical workflow: clinician says "14 buccal" not "activate, 14 buccal"
            print(f"[GUI/_tooth_sel] Auto-activating from voice tooth command")
            self._chart_go()   # sets _cstate = 'waiting'
        if self._cstate not in ("waiting", "scanning"):
            print(f"[GUI/_tooth_sel] BLOCKED — unexpected cstate {self._cstate!r}")
            return
        tid = str(n) + s
        print(f"[GUI/_tooth_sel] Starting scan for {tid}")
        # If already scanning a different tooth, stop it first
        prev = getattr(self.chart_system, 'current_tooth', None)
        if prev and prev != tid and self.chart_system.scanning_active:
            print(f"[GUI/_tooth_sel] Auto-stopping previous tooth {prev}")
            self.chart_system.stop_tooth_scan()
        self.chart_system.start_tooth_scan(tid); self._cstate = "scanning"
        self.t_lbl.setText(tid); self._slbls["State"].setText("Scanning " + tid)
        self.statusBar().showMessage("Scanning " + tid + " -- say STOP or click STOP")
        self._update_tooth_grid(active_n=n)
        self._vlog("tooth", tid)

    def _chart_stop(self):
        # Try chart_system first, fall back to chart_annotator.tooth_chart
        cs = self.chart_system
        if not getattr(cs, 'current_tooth', None) and hasattr(self, 'chart_annotator'):
            alt = getattr(self.chart_annotator, 'tooth_chart', None)
            if alt and getattr(alt, 'current_tooth', None):
                cs = alt
                print(f"[GUI/_chart_stop] Using chart_annotator.tooth_chart as fallback")
        tid      = getattr(cs, "current_tooth", None)
        scanning = getattr(cs, "scanning_active", None)
        buf_len  = len(getattr(cs, "tooth_buffer", []))
        print(f"[GUI/_chart_stop] current_tooth={tid!r} scanning={scanning} buffer={buf_len}")
        if not tid:
            print("[Chart] _chart_stop: no active tooth scan")
            return
        cs.stop_tooth_scan()
        n   = int(re.sub(r"[^0-9]", "", tid) or "0")
        srf = re.sub(r"[^BL]", "", tid.upper())
        if n not in self._saved: self._saved[n] = set()
        if srf: self._saved[n].add(srf)
        self._update_tooth_grid()
        total = sum(len(v) for v in self._saved.values())
        self._slbls["Keyframes"].setText(str(total))
        self.statusBar().showMessage("Saved " + tid + " -- say next tooth")
        self.t_lbl.setText("--"); self._cstate = "waiting"
        self._vlog("key/voice", "STOP -> saved " + tid)

    def _chart_end(self):
        self._chart_stop()
        import datetime
        ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use same fallback logic for export
        cs = self.chart_system
        if not cs.best_frames_saved and hasattr(self, 'chart_annotator'):
            alt = getattr(self.chart_annotator, 'tooth_chart', None)
            if alt and alt.best_frames_saved:
                cs = alt
                print(f"[GUI/_chart_end] Using chart_annotator.tooth_chart for export")
        cs.output_dir = self.chart_out
        xl  = str(self.chart_out / f"chart_{ts}.xlsx")
        print(f"[EXCEL DEBUG] chart_out = {self.chart_out}  is_abs={self.chart_out.is_absolute()}")
        print(f"[EXCEL DEBUG] xl = {xl!r}  teeth={len(cs.best_frames_saved)}")
        ok  = cs.export_to_excel(xl)
        self._cstate = "inactive"; self.b_go.setText("ACTIVATE"); self.b_go.setStyleSheet(BTN_SUCCESS)
        self.lbdg.setText("IDLE"); self.lbdg.setStyleSheet(
            "font-family:Consolas,SF Mono,Cascadia Code,monospace;font-size:20px;font-weight:600;color:" + MUTED + ";background:" + SURFACE + ";border-radius:10px;padding:2px 8px;")
        self.statusBar().showMessage(f"Session ended -- {len(self._saved)} teeth ({sum(len(v) for v in self._saved.values())} scans)" + (" -> " + xl if ok else ""))
        self._vlog("key/voice", f"END -- {len(self._saved)} teeth")

    # ---- keys ----------------------------------------------------------------

    def _key1(self):
        if self._mode == "stitching":
            if not self.pano_system.recording: self._stitch_start()
        else: self._mode_set("stitching")

    def _key2(self):
        if self._mode == "stitching":
            if self.pano_system.recording: self._stitch_stop()
            else: self._mode_set("charting")
        else: self._mode_set("stitching")

    def _screenshot(self):
        if self.flabel.pixmap():
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            d  = self.chart_out if self._mode == "charting" else self.stitch_out
            self.flabel.pixmap().save(str(d / f"screenshot_{ts}.png"))
            self.statusBar().showMessage("Screenshot saved")

    # ---- settings ------------------------------------------------------------

    def _open_voice_tune(self):
        """Open the in-app Whisper adaptation dialog."""
        dlg = VoiceTuneDialog(self.args, self)
        dlg.exec() if hasattr(dlg,"exec") else dlg.exec_()

    def _open_settings(self):
        dlg = SettingsDialog(self.args, self)
        if dlg.exec() if hasattr(dlg, "exec") else dlg.exec_():
            v = dlg.values()
            # Apply ALL settings to live objects immediately
            ps = self.pano_system
            ps.cf.tau                         = v["tau"]
            ps.cf.imu_sign                    = v["imu_sign"]
            ps.proc.iou_gate.iou_thresh       = v["iou_thresh"]
            ps.proc.iou_gate.jump_thresh      = v["jump_thresh"]
            ps.proc.gater.threshold           = v["qs_thresh"]   # was missing
            ps._sweep_tracker.forward_only    = not v["bidirectional"]
            # Apply jitter to BOTH charting and stitching
            self.chart_system.jitter_threshold_um = v["jitter_um"]
            self.chart_system.min_stable_frames   = v["min_frames"]
            # Stitching jitter — pano_system stores it directly
            if hasattr(ps, 'jitter_threshold_um'):
                ps.jitter_threshold_um = v["jitter_um"]
            # Also update in the sharp canvas if it has its own threshold
            if hasattr(ps, 'sharp_canvas') and hasattr(ps.sharp_canvas, 'jitter_threshold_um'):
                ps.sharp_canvas.jitter_threshold_um = v["jitter_um"]
            # Update args so dialog re-opens with current values
            for k, val in v.items():
                if hasattr(self.args, k):
                    setattr(self.args, k, val)
            self.statusBar().showMessage(
                f"Settings applied — qs={v['qs_thresh']:.2f}  "
                f"tau={v['tau']:.2f}  jitter={v['jitter_um']:.0f}µm")

    # ---- cleanup -------------------------------------------------------------

    def closeEvent(self, ev):
        if self.pano_system.recording:
            r = QMessageBox.question(self, "Recording active",
                "Recording in progress. Save before exit?",
                QMessageBox.Yes | QMessageBox.No)
            if r == QMessageBox.Yes: self.pano_system._do_stop_save()

        # Stop voice first (daemon thread — exits quickly)
        if self.voice:
            try: self.voice.stop()
            except Exception: pass

        # Signal inference + voice threads to stop
        self.inf.stop()
        if hasattr(self, "vt"):
            self.vt.stop()

        # Stop hardware (releases camera + BLE)
        try: self.capture.stop_capture()
        except Exception: pass
        try: self.imu.stop()
        except Exception: pass

        # Wait briefly for threads — terminate if they don't stop in time
        if not self.inf.wait(1500):
            self.inf.terminate()
        if hasattr(self, "vt") and not self.vt.wait(800):
            self.vt.terminate()

        ev.accept()


# =============================================================================
#  DevicePreviewDialog -- shows a live frame from the selected HD60X device
# =============================================================================

class DevicePreviewDialog(QDialog):
    """
    Opens the selected device index with cv2.VideoCapture and shows a live
    preview so the clinician can confirm they picked the right camera.
    Scans devices 0-5 and shows thumbnails of all that respond.
    """
    def __init__(self, initial_device: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HD60X Device Preview")
        self.setMinimumWidth(800)
        self.setMinimumHeight(520)
        self.setStyleSheet(STYLE)
        self._caps = {}        # device_index -> cv2.VideoCapture
        self._selected = initial_device
        self._timer = QTimer(); self._timer.timeout.connect(self._refresh)

        lay = QVBoxLayout(self); lay.setSpacing(12)

        title = QLabel("Select the correct HD60X device")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:22px;font-weight:600;color:" + TEXT + ";padding:8px 0;")
        lay.addWidget(title)

        hint = QLabel("Click a preview to select it. Green border = currently selected.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("font-size:16px;color:" + MUTED + ";")
        lay.addWidget(hint)

        # Grid of device previews
        self._preview_grid = QHBoxLayout()
        self._preview_grid.setSpacing(12)
        self._preview_labels = {}    # device_index -> QLabel (shows frame)
        self._preview_btns   = {}    # device_index -> QPushButton (select)

        self._scan_devices()

        lay.addLayout(self._preview_grid)

        # Status
        self.status_lbl = QLabel(f"Selected: device {self._selected}")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        self.status_lbl.setStyleSheet("font-size:18px;color:" + TEAL + ";font-weight:600;")
        lay.addWidget(self.status_lbl)

        # Buttons
        br = QHBoxLayout()
        ok_btn   = QPushButton("Use selected device"); ok_btn.setStyleSheet(BTN_TEAL)
        skip_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        skip_btn.clicked.connect(self.reject)
        br.addStretch(); br.addWidget(ok_btn); br.addWidget(skip_btn)
        lay.addLayout(br)

        self._timer.start(200)   # refresh at 5 fps

    def _scan_devices(self):
        """Try opening devices 0-5, create preview panel for each that works."""
        for idx in range(6):
            import os, platform
            # Suppress OpenCV backend warnings during device scan
            os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
            cap = cv2.VideoCapture(idx)  # default backend — most compatible
            if not cap.isOpened():
                cap.release(); continue
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release(); continue

            # Device works -- add preview panel
            self._caps[idx] = cap
            panel = QFrame()
            panel.setStyleSheet(
                "QFrame{border:3px solid " + (TEAL if idx == self._selected else BORDER) + ";border-radius:8px;background:" + PANEL + ";}")
            pv = QVBoxLayout(panel); pv.setSpacing(4); pv.setContentsMargins(6,6,6,6)

            lbl = QLabel(); lbl.setFixedSize(240, 180)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#000;border-radius:4px;")
            pv.addWidget(lbl)

            dev_lbl = QLabel(f"Device {idx}")
            dev_lbl.setAlignment(Qt.AlignCenter)
            dev_lbl.setStyleSheet("font-size:16px;font-weight:600;color:" + TEXT + ";")
            pv.addWidget(dev_lbl)

            btn = QPushButton("Select"); btn.setStyleSheet(BTN_PRIMARY if idx != self._selected else BTN_TEAL)
            btn.clicked.connect(lambda _, i=idx: self._select(i))
            pv.addWidget(btn)

            self._preview_labels[idx] = lbl
            self._preview_btns[idx]   = btn
            self._preview_grid.addWidget(panel)
            self._panels = getattr(self, '_panels', {})
            self._panels[idx] = panel

        if not self._caps:
            no_dev = QLabel("No cameras found. Check HD60X USB connection.")
            no_dev.setAlignment(Qt.AlignCenter)
            no_dev.setStyleSheet("font-size:16px;color:" + RED + ";")
            self._preview_grid.addWidget(no_dev)

    def _select(self, idx: int):
        self._selected = idx
        self.status_lbl.setText(f"Selected: device {idx}")
        for i, panel in getattr(self, '_panels', {}).items():
            border_col = TEAL if i == idx else BORDER
            panel.setStyleSheet(f"QFrame{{border:3px solid {border_col};border-radius:8px;background:{PANEL};}}")
            self._preview_btns[i].setStyleSheet(BTN_TEAL if i == idx else BTN_PRIMARY)

    def _refresh(self):
        """Grab one frame from each open capture and update the label."""
        for idx, cap in self._caps.items():
            ret, frame = cap.read()
            if not ret or frame is None: continue
            lbl = self._preview_labels.get(idx)
            if lbl is None: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(bytes(rgb.data), w, h, ch * w, QImage.Format_RGB888)
            px = QPixmap.fromImage(img).scaled(
                240, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(px)

    def get_selected(self) -> int:
        return self._selected

    def closeEvent(self, ev):
        self._timer.stop()
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()
        ev.accept()

    def accept(self):
        self._timer.stop()
        for cap in self._caps.values(): cap.release()
        self._caps.clear()
        super().accept()

    def reject(self):
        self._timer.stop()
        for cap in self._caps.values(): cap.release()
        self._caps.clear()
        super().reject()


# =============================================================================
#  StartupDialog -- shown when no --model given, or always for easy config
# =============================================================================

class StartupDialog(QDialog):
    """
    Shown at launch so the clinician can pick model, output dir, and user
    without needing to remember command-line arguments.
    All fields pre-filled from args if provided.
    """
    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DentWise — Startup")
        self.setMinimumWidth(780)
        self.setMinimumHeight(600)
        self.resize(780, 620)
        self.setStyleSheet(STYLE)

        lay = QVBoxLayout(self); lay.setSpacing(14)

        # Title
        title = QLabel("DentWise")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:40px;font-weight:600;color:" + TEAL + ";padding:12px 0 4px;letter-spacing:0.08em;")
        lay.addWidget(title)

        subtitle = QLabel("Configure session before starting")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size:26px;color:" + MUTED + ";padding-bottom:4px;")
        lay.addWidget(subtitle)

        credit = QLabel("DentWise  |  Designed by Reza Amjadian")
        credit.setAlignment(Qt.AlignCenter)
        credit.setStyleSheet(
            "font-size:22px;font-weight:700;color:" + TEAL + ";"
            "padding-bottom:10px;letter-spacing:0.06em;")
        lay.addWidget(credit)

        form = QFormLayout(); form.setSpacing(12); form.setLabelAlignment(Qt.AlignRight)

        # Model path
        model_row = QHBoxLayout(); model_row.setSpacing(6)
        self.model_edit = QLineEdit(args.model or "")
        self.model_edit.setPlaceholderText("Browse to model file (.h5)...")
        btn_model = QPushButton("Browse")
        btn_model.setFixedWidth(120)
        btn_model.clicked.connect(self._browse_model)
        model_row.addWidget(self.model_edit); model_row.addWidget(btn_model)
        form.addRow("DentScan Model (.h5):", model_row)

        # Output directory
        out_row = QHBoxLayout(); out_row.setSpacing(6)
        self.out_edit = QLineEdit(args.out or "results")
        self.out_edit.setPlaceholderText("Select output folder...")
        btn_out = QPushButton("Browse")
        btn_out.setFixedWidth(120)
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit); out_row.addWidget(btn_out)
        form.addRow("Output folder:", out_row)

        # User / adapted model
        self.user_edit = QLineEdit(args.user or "")
        self.user_edit.setPlaceholderText("Reza")
        form.addRow("Clinician ID:", self.user_edit)

        # HD60X device row with preview button
        dev_row = QHBoxLayout(); dev_row.setSpacing(6)
        self.dev_spin = QSpinBox(); self.dev_spin.setRange(0, 10)
        self.dev_spin.setValue(args.device)
        btn_preview = QPushButton("Preview")
        btn_preview.setFixedWidth(120)
        btn_preview.setStyleSheet(BTN_TEAL)
        btn_preview.clicked.connect(self._preview_device)
        dev_row.addWidget(self.dev_spin); dev_row.addWidget(btn_preview)
        form.addRow("HD60X device:", dev_row)

        # Voice model
        self.wm = QComboBox(); self.wm.addItems(["tiny","base","small","medium"])
        self.wm.setCurrentText(args.whisper_model)
        form.addRow("Whisper model:", self.wm)

        # Microphone selector
        self.mic_cb = QComboBox()
        self.mic_cb.setMinimumWidth(320)
        self._startup_populate_mics(getattr(args, 'voice_device', None))
        form.addRow("Microphone:", self.mic_cb)

        # No voice
        self.nov = QCheckBox("Disable voice control")
        self.nov.setChecked(args.no_voice)
        form.addRow("", self.nov)

        # IMU connection type
        imu_row = QHBoxLayout(); imu_row.setSpacing(6)
        self.imu_mode = QComboBox()
        self.imu_mode.addItems(["BLE (wireless)", "Serial / USB (wired)"])
        # Default to wired (index 1) — change to 0 for BLE default
        self.imu_mode.setCurrentIndex(1 if getattr(args,'serial',None) is not None else 1)
        self.imu_mode.currentIndexChanged.connect(self._on_imu_mode)
        imu_row.addWidget(self.imu_mode)
        form.addRow("IMU connection:", imu_row)

        # Serial port (shown only when Serial mode selected)
        serial_row = QHBoxLayout(); serial_row.setSpacing(6)
        self.serial_edit = QLineEdit(getattr(args,'serial',None) or "COM8")
        self.serial_edit.setPlaceholderText("e.g. COM8")
        self.serial_edit.setFixedWidth(100)
        self.baud_spin = QSpinBox()
        self.baud_spin.setRange(1200, 921600)
        self.baud_spin.setValue(getattr(args,'baud',9600))
        self.baud_spin.setSingleStep(100)
        serial_row.addWidget(self.serial_edit)
        serial_row.addWidget(QLabel("Baud:"))
        serial_row.addWidget(self.baud_spin)
        serial_row.addStretch()
        self._serial_row_widget = QWidget()
        self._serial_row_widget.setLayout(serial_row)
        form.addRow("Serial port:", self._serial_row_widget)

        # BLE MAC address (shown only when BLE mode selected)
        self.mac_edit = QLineEdit(getattr(args,'mac',None) or "")
        self.mac_edit.setPlaceholderText("e.g. D3:F9:B0:EF:13:A0  (leave blank to auto-scan)")
        self._mac_row_lbl = QLabel("BLE MAC address:")
        form.addRow(self._mac_row_lbl, self.mac_edit)

        # Set initial visibility
        self._on_imu_mode(1)   # default wired

        lay.addLayout(form)

        # ── Voice personalisation section ─────────────────────────────────────
        sep = QFrame(); sep.setFrameShape(QFrame.HLine if hasattr(QFrame,'HLine') else QFrame.Shape.HLine)
        sep.setStyleSheet("color:" + BORDER + ";")
        lay.addWidget(sep)

        vp_lbl = QLabel("Voice Personalisation")
        vp_lbl.setStyleSheet("font-size:22px;font-weight:600;color:" + TEAL + ";")
        lay.addWidget(vp_lbl)

        vp_hint = QLabel(
            "Record 30–90 s of dental phrases once per clinician, then fine-tune Whisper "
            "to their accent. The adapted model loads automatically on next launch.")
        vp_hint.setWordWrap(True)
        vp_hint.setStyleSheet("font-size:20px;color:" + MUTED + ";")
        lay.addWidget(vp_hint)

        # Enrollment dir
        enr_row = QHBoxLayout(); enr_row.setSpacing(6)
        self.enr_edit = QLineEdit(getattr(args, 'enrollment_dir', 'enrollment'))
        self.enr_edit.setPlaceholderText("enrollment/")
        btn_enr = QPushButton("Browse"); btn_enr.setFixedWidth(120)
        btn_enr.clicked.connect(self._browse_enr)
        enr_row.addWidget(self.enr_edit); enr_row.addWidget(btn_enr)
        form_vp = QFormLayout(); form_vp.setSpacing(8)
        form_vp.addRow("Enrollment folder:", enr_row)
        lay.addLayout(form_vp)

        # Status label — shows enrolled users and model status
        self.enr_status = QLabel("")
        self.enr_status.setStyleSheet("font-size:20px;color:" + MUTED + ";")
        self.enr_status.setWordWrap(True)
        lay.addWidget(self.enr_status)
        QTimer.singleShot(100, self._refresh_enr_status)

        # Enroll + Fine-tune buttons
        vp_row = QHBoxLayout(); vp_row.setSpacing(8)
        self.b_enroll  = QPushButton("Record Voice (--enroll)")
        self.b_finetune= QPushButton("Fine-Tune Whisper")
        self.b_enroll  .setStyleSheet(BTN_PRIMARY)
        self.b_finetune.setStyleSheet(BTN_AMBER)
        self.b_enroll  .clicked.connect(self._do_enroll)
        self.b_finetune.clicked.connect(self._do_finetune)
        vp_row.addWidget(self.b_enroll); vp_row.addWidget(self.b_finetune)
        lay.addLayout(vp_row)

        self.vp_log = QLabel("")
        self.vp_log.setStyleSheet("font-size:20px;color:" + AMBER + ";")
        self.vp_log.setWordWrap(True)
        lay.addWidget(self.vp_log)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine if hasattr(QFrame,'HLine') else QFrame.Shape.HLine)
        sep2.setStyleSheet("color:" + BORDER + ";")
        lay.addWidget(sep2)

        # Validation message
        self.msg = QLabel("")
        self.msg.setStyleSheet("color:" + RED + ";font-size:24px;")
        self.msg.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.msg)

        # Buttons
        br = QHBoxLayout(); br.setSpacing(8)
        self.ok_btn = QPushButton("Start Session"); self.ok_btn.setStyleSheet(BTN_TEAL)
        quit_btn    = QPushButton("Quit")
        self.ok_btn.clicked.connect(self._on_ok)
        quit_btn.clicked.connect(self.reject)
        br.addStretch(); br.addWidget(self.ok_btn); br.addWidget(quit_btn)
        lay.addLayout(br)

    def _startup_populate_mics(self, current_device=None):
        """Populate microphone dropdown with all available input devices."""
        self.mic_cb.clear()
        self.mic_cb.addItem("System default", None)
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            selected_idx = 0
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    name = info["name"][:55]
                    item_text = f"{i}: {name}"
                    # Highlight XREAL or known voice mics
                    if any(k in name.upper() for k in
                           ["XREAL","HEADSET","HEADPHONE","EARPHONE","USB MIC",
                            "MICROPHONE","WEBCAM","LOGITECH"]):
                        item_text = f"{i}: ★ {name}"
                    self.mic_cb.addItem(item_text, i)
                    if current_device is not None and i == current_device:
                        selected_idx = self.mic_cb.count() - 1
            pa.terminate()
            if selected_idx > 0:
                self.mic_cb.setCurrentIndex(selected_idx)
        except ImportError:
            self.mic_cb.addItem("pyaudio not installed — cannot list mics", None)
        except Exception as e:
            self.mic_cb.addItem(f"Error listing mics: {e}", None)

    def _on_imu_mode(self, idx):
        """Show serial fields for wired mode, MAC field for BLE mode."""
        is_serial = (idx == 1)
        self._serial_row_widget.setVisible(is_serial)
        self._mac_row_lbl.setVisible(not is_serial)
        self.mac_edit.setVisible(not is_serial)

    def _preview_device(self):
        dev = self.dev_spin.value()
        dlg = DevicePreviewDialog(dev, self)
        if dlg.exec() if hasattr(dlg, "exec") else dlg.exec_():
            self.dev_spin.setValue(dlg.get_selected())

    def _browse_enr(self):
        try:
            from PyQt5.QtWidgets import QFileDialog
        except ImportError:
            from PyQt6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(
            self, "Select Enrollment Folder", self.enr_edit.text() or ".")
        if path:
            self.enr_edit.setText(path)
            self._refresh_enr_status()

    def _refresh_enr_status(self):
        """Show which users are enrolled and whether their model is ready."""
        enr_root = Path(self.enr_edit.text().strip() or "enrollment")
        user     = self.user_edit.text().strip()
        if user:
            enr_dir = enr_root / user.lower().replace(" ", "_")
        else:
            enr_dir = enr_root
        recs  = list((enr_dir / "recordings").glob("phrase_*.wav")) if                 (enr_dir / "recordings").exists() else []
        model = enr_dir / "adapted_model.pt"
        if model.exists():
            self.enr_status.setText(
                f"✓  {len(recs)} recordings  |  adapted_model.pt ready — "
                f"Whisper will load personalised model at launch.")
            self.enr_status.setStyleSheet("font-size:20px;color:" + TEAL + ";")
        elif recs:
            self.enr_status.setText(
                f"⚠  {len(recs)} recordings found — click Fine-Tune Whisper to create adapted model.")
            self.enr_status.setStyleSheet("font-size:20px;color:" + AMBER + ";")
        else:
            self.enr_status.setText(
                "No enrollment found for this user — click Record Voice to start.")
            self.enr_status.setStyleSheet("font-size:20px;color:" + MUTED + ";")

    def _do_enroll(self):
        """Open in-app VoiceTuneDialog and start enrollment."""
        self._open_tune_dialog("enroll")

    def _do_finetune(self):
        """Open in-app VoiceTuneDialog and start fine-tuning."""
        self._open_tune_dialog("finetune")

    def _open_tune_dialog(self, mode=None):
        """Open the VoiceTuneDialog — all output shown inside the app."""
        dlg = VoiceTuneDialog(self._make_args(), self)
        if mode == "enroll":
            # Switch to enrollment tab and start session
            QTimer.singleShot(200, dlg._start_enrollment_session)
        elif mode == "finetune":
            # Switch to fine-tune tab and start training
            QTimer.singleShot(200, lambda: (
                dlg.tabs.setCurrentIndex(1),
                dlg._start_finetune()
            ))
        dlg.exec() if hasattr(dlg,"exec") else dlg.exec_()
        self._refresh_enr_status()   # update status after dialog closes

    def _make_args(self):
        """Build a minimal args namespace from current dialog values."""
        import argparse
        a = argparse.Namespace()
        a.user            = self.user_edit.text().strip() or None
        a.enrollment_dir  = self.enr_edit.text().strip() or "enrollment"
        a.whisper_model   = self.wm.currentText()
        a.voice_device    = None
        return a

    def _browse_model(self):
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DentScan Model", self.model_edit.text() or ".",
            "H5 model (*.h5);;All files (*.*)")
        if path:
            self.model_edit.setText(path)

    def _browse_out(self):
        from PyQt5.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.out_edit.text() or ".")
        if path:
            self.out_edit.setText(path)

    def _on_ok(self):
        if not self.model_edit.text().strip():
            self.msg.setText("Please select a model file (.h5)")
            return
        if not Path(self.model_edit.text().strip()).exists():
            self.msg.setText("Model file not found -- please check the path")
            return
        if not self.out_edit.text().strip():
            self.msg.setText("Please select an output folder")
            return
        self.accept()

    def values(self):
        is_serial = (self.imu_mode.currentIndex() == 1)
        return dict(
            model         = self.model_edit.text().strip(),
            out           = self.out_edit.text().strip(),
            user          = self.user_edit.text().strip() or None,
            device        = self.dev_spin.value(),
            whisper_model = self.wm.currentText(),
            no_voice      = self.nov.isChecked(),
            enrollment_dir= self.enr_edit.text().strip() or "enrollment",
            serial        = self.serial_edit.text().strip() if is_serial else None,
            baud          = self.baud_spin.value()          if is_serial else 9600,
            mac           = self.mac_edit.text().strip()    if not is_serial else None,
            voice_device  = self.mic_cb.currentData(),
        )


def _suppress_terminal_windows():
    """
    Patch subprocess.Popen to always use CREATE_NO_WINDOW on Windows.
    Silences terminal flashes from Whisper (ffmpeg), wmic, and any other
    library spawning subprocesses. Called as first line of main().
    """
    import sys
    if sys.platform != "win32":
        return
    import subprocess
    _orig_init = subprocess.Popen.__init__
    _CREATE_NO_WINDOW = 0x08000000
    def _patched_init(self, *args, **kwargs):
        # Only suppress terminal for specific tools — not for audio/whisper processes
        cmd = args[0] if args else kwargs.get('args','')
        cmd_str = str(cmd[0] if isinstance(cmd, list) else cmd).lower()
        # Do NOT suppress for ffmpeg (whisper needs it) or portaudio
        skip = any(x in cmd_str for x in ['ffmpeg','portaudio','pactl','arecord'])
        if not skip:
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | _CREATE_NO_WINDOW
        _orig_init(self, *args, **kwargs)
    subprocess.Popen.__init__ = _patched_init


def main():
    _suppress_terminal_windows()
    p = argparse.ArgumentParser(description="DentWise (PyQt5/6)")
    p.add_argument("--model","-m",     type=str, default=None,
                   help="Path to MKUNet .h5 model. If omitted, a startup dialog lets you browse.")
    p.add_argument("--device","-d",    type=int, default=0)
    p.add_argument("--user",           type=str, default=None)
    p.add_argument("--enrollment_dir", type=str, default="enrollment")
    p.add_argument("--out",            type=str, default="results")
    p.add_argument("--no_voice",       action="store_true")
    p.add_argument("--voice_device",   type=int, default=None)
    p.add_argument("--whisper_model",  type=str, default="small",
                   choices=["tiny","base","small","medium"])
    p.add_argument("--serial",         type=str, default=None)
    p.add_argument("--baud",           type=int, default=9600)
    p.add_argument("--mac",            type=str, default=None)
    p.add_argument("--no_imu",         action="store_true")
    p.add_argument("--tau",            type=float, default=0.2)
    p.add_argument("--imu_sign",       type=float, default=-1.0)
    p.add_argument("--qs_thresh",      type=float, default=0.28)
    p.add_argument("--qs_sharp",       type=float, default=400.)
    p.add_argument("--qs_contact",     type=float, default=1800.)
    p.add_argument("--qs_snr",         type=float, default=18.)
    p.add_argument("--bidirectional",  action="store_true")
    p.add_argument("--iou_thresh",     type=float, default=0.25)
    p.add_argument("--jump_thresh",    type=float, default=15.0)
    p.add_argument("--jitter_um",      type=float, default=100.0)

    p.add_argument("--min_frames",     type=int,   default=10)
    p.add_argument("--scale",          type=float, default=0.12)
    args = p.parse_args()

    # Show startup dialog -- always shown so clinician can browse for model/output
    # (also allows running with no command-line args at all)
    app = QApplication(sys.argv)
    app.setApplicationName("DentWise")
    app.setStyleSheet(STYLE)

    dlg = StartupDialog(args)
    if not (dlg.exec() if hasattr(dlg, "exec") else dlg.exec_()):
        sys.exit(0)
    cfg = dlg.values()
    args.model        = cfg["model"]
    args.out          = cfg["out"]
    args.user         = cfg["user"]
    args.device       = cfg["device"]
    args.whisper_model   = cfg["whisper_model"]
    args.no_voice        = cfg["no_voice"]
    args.enrollment_dir  = cfg.get("enrollment_dir", getattr(args,'enrollment_dir','enrollment'))
    args.serial          = cfg.get("serial",       getattr(args,"serial",       None))
    args.baud            = cfg.get("baud",         getattr(args,"baud",         9600))
    args.mac             = cfg.get("mac",           getattr(args,"mac",          None))
    args.voice_device    = cfg.get("voice_device",  getattr(args,"voice_device", None))

    # >>> Compute paths HERE — after args.out is set from Startup Dialog >>>
    root_out = Path(args.out).resolve()
    print(f"[PATH DEBUG] args.out = {args.out!r}")
    print(f"[PATH DEBUG] root_out = {root_out}")
    print(f"[PATH DEBUG] is_absolute = {root_out.is_absolute()}")
    # Guard double-nesting
    if root_out.name == "charting":
        root_out = root_out.parent
        print(f"[PATH DEBUG] Corrected root_out = {root_out}")
    stitch_out = root_out / "stitching"
    chart_out  = root_out / "charting"
    stitch_out.mkdir(parents=True, exist_ok=True)
    chart_out.mkdir(parents=True,  exist_ok=True)
    print(f"[PATH DEBUG] chart_out = {chart_out}  abs={chart_out.is_absolute()}")

    # paths computed after cfg — see >>> marker below

    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(args.model).parent))

    # ── Loading window — stays open while model loads in a background thread ────
    # Uses QThread so the Qt event loop keeps running and the window stays
    # responsive and visible. The window only closes when loading is done.

    class _LoadThread(QThread):
        step_sig  = pyqtSignal(int, str)   # (step_number, message)
        done_sig  = pyqtSignal(object, object, object, object)  # results
        error_sig = pyqtSignal(str)

        def __init__(self, args):
            super().__init__()
            self._args = args

        def run(self):
            try:
                self.step_sig.emit(1, "Loading AI modules…")
                import live_unified as _U
                from video_inference_pipeline import MKUNetInference as _MKU

                # Import capture class — guarded because live_hybrid imports
                # training scripts (pandas, sklearn) that may not be present
                try:
                    from live_hybrid_fusion_realtime2 import HD60XOptimizedCapture as _HD60X
                except ModuleNotFoundError as _me:
                    # Specific missing package — give a clear message
                    raise RuntimeError(
                        f"Missing package required by capture module: {_me}\n\n"
                        f"Fix: pip install {str(_me).split(chr(39))[1]}") from _me

                self.step_sig.emit(2, "Locating model…")
                _model_path = self._args.model

                self.step_sig.emit(3, "Loading AI model into GPU memory…  (please wait — this takes 1–3 minutes)")
                mkunet = _MKU(_model_path)

                self.step_sig.emit(4, "Initialising annotators…")
                pano_ann  = _U.MKUNetLiveAnnotator(mkunet)
                chart_ann = _U.RealTimeB1ChartAnnotator(
                    model_path=self._args.model,
                    enrollment_dir=self._args.enrollment_dir,
                    whisper_model_size=self._args.whisper_model,
                    voice_device=self._args.voice_device,
                    enable_voice=False,
                    mkunet_inference=mkunet)

                self.step_sig.emit(5, "Starting capture card…")
                cap = _HD60X(device_id=self._args.device)
                cap.start_capture()

                self.step_sig.emit(6, "Ready!")
                self.done_sig.emit(_U, mkunet, pano_ann, chart_ann)
                # Store cap as attribute so it survives thread exit
                self._cap = cap

            except Exception as e:
                import traceback
                self.error_sig.emit(str(e) + chr(10) + traceback.format_exc())

    # Build the loading dialog
    load_dlg = QDialog()
    load_dlg.setWindowTitle("DentWise — Starting")
    load_dlg.setFixedSize(560, 220)
    load_dlg.setStyleSheet(STYLE)
    # Remove close button — user cannot cancel mid-load
    flags = load_dlg.windowFlags()
    if hasattr(Qt, 'WindowCloseButtonHint'):
        flags &= ~Qt.WindowCloseButtonHint
        flags |= Qt.WindowStaysOnTopHint
    else:
        flags &= ~Qt.WindowType.WindowCloseButtonHint
        flags |= Qt.WindowType.WindowStaysOnTopHint
    load_dlg.setWindowFlags(flags)

    _llay = QVBoxLayout(load_dlg)
    _llay.setContentsMargins(30, 24, 30, 24); _llay.setSpacing(16)

    _ltitle = QLabel("DentWise  —  Starting up")
    _ltitle.setStyleSheet(f"font-size:28px;font-weight:700;color:{TEAL};")
    _ltitle.setAlignment(Qt.AlignCenter if hasattr(Qt,'AlignCenter')
                         else Qt.AlignmentFlag.AlignCenter)
    _llay.addWidget(_ltitle)

    _lstep = QLabel("Initialising…")
    _lstep.setStyleSheet(f"font-size:21px;color:{MUTED};")
    _lstep.setAlignment(Qt.AlignCenter if hasattr(Qt,'AlignCenter')
                        else Qt.AlignmentFlag.AlignCenter)
    _lstep.setWordWrap(True)
    _llay.addWidget(_lstep)

    _lbar = QProgressBar()
    _lbar.setRange(0, 6); _lbar.setValue(0); _lbar.setTextVisible(False)
    _lbar.setStyleSheet(
        f"QProgressBar{{background:#1a1f2e;border-radius:4px;height:12px;}}"
        f"QProgressBar::chunk{{background:{TEAL};border-radius:4px;}}")
    _llay.addWidget(_lbar)

    # Mutable container to pass results out of slots
    _boot_result = [None, None, None, None, None]  # [U, mkunet, pano, chart, cap]
    _boot_error  = [None]

    def _on_step(n, msg):
        _lstep.setText(f"Step {n}/6  —  {msg}"); _lbar.setValue(n)

    def _on_done(U_mod, mkunet_obj, pano_obj, chart_obj):
        _boot_result[0] = U_mod
        _boot_result[1] = mkunet_obj
        _boot_result[2] = pano_obj
        _boot_result[3] = chart_obj
        _boot_result[4] = _loader_thread._cap
        load_dlg.accept()   # closes dialog, unblocks exec()

    def _on_error(msg):
        _boot_error[0] = msg
        load_dlg.reject()

    _loader_thread = _LoadThread(args)
    _loader_thread.step_sig .connect(_on_step)
    _loader_thread.done_sig .connect(_on_done)
    _loader_thread.error_sig.connect(_on_error)

    load_dlg.show()
    _loader_thread.start()
    load_dlg.exec() if hasattr(load_dlg,'exec') else load_dlg.exec_()

    # Check for error
    if _boot_error[0]:
        QMessageBox = __import__('PyQt5.QtWidgets', fromlist=['QMessageBox']).QMessageBox
        QMessageBox.critical(None, "DentWise — Startup Error",
                             f"Failed to load model or hardware:\n\n{_boot_error[0]}")
        sys.exit(1)

    # Unpack results
    U          = _boot_result[0]
    mkunet     = _boot_result[1]
    pano_annotator  = _boot_result[2]
    chart_annotator = _boot_result[3]
    capture    = _boot_result[4]

    if args.no_imu:
        imu = U.IMUReader(mac=args.mac or U.IMU_MAC)
    elif args.serial:
        imu = U.SerialIMUReader(port=args.serial, baud=args.baud); imu.start()
        for _ in range(8):
            time.sleep(1)
            if imu.connected: break
    else:
        imu = U.IMUReader(mac=args.mac or U.IMU_MAC); imu.start()
        for _ in range(8):
            time.sleep(0.5)
            if imu.connected: break

    pano_system = U.LivePanoramaSystem(
        annotator=pano_annotator, capture=capture,
        imu_reader=imu, output_dir=str(stitch_out), voice=None)
    pano_system.cf.tau      = args.tau
    pano_system.cf.scale    = args.scale
    pano_system.cf.imu_sign = args.imu_sign
    pano_system.proc.iou_gate.iou_thresh  = args.iou_thresh
    pano_system.proc.iou_gate.jump_thresh = args.jump_thresh
    pano_system._sweep_tracker.forward_only = not args.bidirectional
    existing = sorted([d for d in stitch_out.glob("stitch_*")
                       if d.is_dir() and d.name.split("_")[-1].isdigit()])
    if existing:
        pano_system._session_id = int(existing[-1].name.split("_")[-1]) + 1
        print(f"[Boot] Resuming from stitch_{pano_system._session_id:03d}")

    # Use chart_annotator's internal tooth_chart as chart_system.
    # This ensures voice commands (which go through chart_annotator)
    # and GUI buttons (which call chart_system) operate on the SAME object.
    # A separate ToothChartManager would always have current_tooth=None
    # because voice fills the annotator's internal one, not a separate instance.
    chart_system = chart_annotator.tooth_chart
    chart_system.output_dir       = chart_out
    chart_system.jitter_threshold_um = args.jitter_um
    chart_system.min_stable_frames   = args.min_frames
    print(f"[Boot] chart_system = chart_annotator.tooth_chart  (unified — voice + GUI share same object)")

    voice = None
    if not args.no_voice:
        print(f"[VOICE DEBUG] Creating voice controller...")
        print(f"[VOICE DEBUG] whisper_model={args.whisper_model}")
        print(f"[VOICE DEBUG] voice_device={args.voice_device}")
        print(f"[VOICE DEBUG] enrollment_dir={args.enrollment_dir}")
        print(f"[VOICE DEBUG] user={args.user}")
        try:
            import pyaudio as _pa_test
            _pa_obj = _pa_test.PyAudio()
            n = _pa_obj.get_device_count()
            print(f"[VOICE DEBUG] PyAudio OK — {n} devices found")
            for i in range(n):
                try:
                    info = _pa_obj.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        print(f"[VOICE DEBUG]   mic {i}: {info['name'][:50]}")
                except: pass
            _pa_obj.terminate()
        except Exception as _pae:
            print(f"[VOICE DEBUG] PyAudio FAILED: {_pae}")
        voice = U.UnifiedVoiceController(
            whisper_model_size=args.whisper_model, voice_device=args.voice_device,
            enrollment_dir=args.enrollment_dir, user=args.user)
        voice.start()
        print(f"[VOICE DEBUG] voice.start() called — thread alive: {voice._thread.is_alive() if voice._thread else False}")

    print("[Boot] Launching GUI...")
    win = MainWindow(args, pano_system, chart_annotator, chart_system,
                     capture, imu, voice, stitch_out, chart_out)
    win.show()
    code = app.exec() if hasattr(app, "exec") else app.exec_()
    sys.exit(code)


if __name__ == "__main__":
    main()
