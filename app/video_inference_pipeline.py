"""
video_inference_pipeline.py
═══════════════════════════════════════════════════════════════════════════════
Step 1 of the Periodontal Ultrasound Panorama Pipeline
───────────────────────────────────────────────────────────────────────────────
PURPOSE
    Extract frames from a Visualsonics Vevo F2 MP4 clip, run MKUNet_LGAG_SASE
    inference on every frame, detect valid (on-tooth) frames, detect the sweep
    window, and save per-frame results for the stitching stage.

PHYSICAL COORDINATE SYSTEM
    Original frame  : ~600–620 × 600–620 px  →  7 × 7 mm field of view
    Pixel spacing   : 7000 µm / 620 px  ≈ 11.29 µm/px  (original resolution)
    Model input     : 256 × 256 px  →  scale factor = 620/256 ≈ 2.422
    Model px spacing: 11.29 × 2.422 ≈ 27.35 µm/px  (in 256×256 space)

    All landmark coordinates stored in BOTH:
        • pixel space  (row, col) at 256×256
        • physical mm  (x_mm, y_mm) using the pixel spacing above

MODEL OUTPUT FORMAT
    Each of the 6 output heads returns a (256, 256, NUM_CLASSES) softmax map.
    CLASS 0 = background, CLASS 1 = foreground.
    Point landmarks (GM, CEJ, ABC): soft-argmax on class-1 probability map.
    Region labels  (TOOTH, GINGIVA, BONE): class-1 probability > 0.5 threshold.

VALID FRAME DETECTION (two-stage)
    Stage 1 — On-tooth vs off-tooth
        A frame is ON-TOOTH if:
            • TOOTH region mean confidence  ≥ TOOTH_CONF_THRESH  (default 0.35)
            • GM landmark peak confidence   ≥ GM_CONF_THRESH     (default 0.40)
        Both thresholds must be satisfied simultaneously.

    Stage 2 — Sweeping vs static
        Inter-frame GM displacement (Euclidean in mm) is computed between
        consecutive on-tooth frames.
        • displacement < STATIC_THRESH_MM  →  static hold (collapse to 1 frame)
        • displacement ≥ STATIC_THRESH_MM  →  sweeping (keep for stitching)
        The longest contiguous sweeping window is selected as the panorama source.

OUTPUTS
    Per-video JSON  : <output_dir>/<video_stem>_inference.json
    ─────────────────────────────────────────────────────────
    {
      "video_path"        : str,
      "total_frames"      : int,
      "fps"               : float,
      "pixel_spacing_um"  : float,        # µm/px at 256×256
      "original_hw"       : [H, W],
      "valid_frames"      : [...],        # on-tooth frame indices
      "sweep_window"      : [start, end], # inclusive frame indices
      "sweep_frames"      : [...],        # frame indices in sweep window
      "static_representative": int,       # best static hold frame index
      "frames"            : {
          "<frame_idx>": {
              "on_tooth"      : bool,
              "is_sweeping"   : bool,
              "quality_score" : float,    # composite [0–1]
              "gm_displacement_mm": float,# vs previous on-tooth frame
              "landmarks": {
                  "GM":  {"row":float,"col":float,"x_mm":float,"y_mm":float,
                          "confidence":float, "visible":bool},
                  "CEJ": {...},
                  "ABC": {...}
              },
              "regions": {
                  "TOOTH":   {"mean_conf":float, "pixel_area":int},
                  "GINGIVA": {"mean_conf":float, "pixel_area":int},
                  "BONE":    {"mean_conf":float, "pixel_area":int}
              }
          }
      }
    }

USAGE
    # Single video
    python video_inference_pipeline.py \
        --model_path /path/to/MKUNet_LGAG_SASE.h5 \
        --video_path /path/to/clip.mp4 \
        --output_dir /path/to/results

    # Batch — all MP4 files in a directory
    python video_inference_pipeline.py \
        --model_path /path/to/MKUNet_LGAG_SASE.h5 \
        --video_dir  /path/to/videos \
        --output_dir /path/to/results

    # Tune thresholds
    python video_inference_pipeline.py \
        --model_path ... --video_path ... \
        --tooth_conf_thresh 0.30 \
        --gm_conf_thresh    0.35 \
        --static_thresh_mm  0.025

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

# ── TensorFlow import (suppress verbose logs) ─────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_INPUT_SIZE   = 256          # MKUNet expects 256×256
ORIGINAL_FOV_MM    = 7.0          # Vevo F2 field of view (width = height = 7 mm)
NOMINAL_ORIG_PX    = 620          # nominal original frame size (px)

# Pixel spacing at model resolution (256×256)
# = (7000 µm / 620 px) × (620 / 256) = 7000 / 256
PIXEL_SPACING_UM   = (ORIGINAL_FOV_MM * 1000) / MODEL_INPUT_SIZE  # ≈ 27.34 µm/px
PIXEL_SPACING_MM   = PIXEL_SPACING_UM / 1000                       # ≈ 0.02734 mm/px

# Output class indices
BG_CLASS = 0
FG_CLASS = 1

# Feature names — must match model output order from training script
POINT_FEATURES  = ['GM', 'CEJ', 'ABC']
REGION_FEATURES = ['TOOTH', 'GINGIVA', 'BONE']
ALL_FEATURES    = ['GM', 'CEJ', 'ABC', 'TOOTH', 'BONE', 'GINGIVA']

# ── Threshold defaults ────────────────────────────────────────────────────────
# SOLE detector: quality_score >= DEFAULT_QS_THRESH
# Empirical distribution (7 clips):
#   off-tooth cluster : QS ≈ 0.10–0.15
#   on-tooth cluster  : QS ≈ 0.20–0.51
# Threshold 0.20 is safely above the off-tooth peak and catches ALL valid frames.
#
# WHY NO SECONDARY CHECKS:
#   TOOTH mean_conf check blocked 6–79 valid frames per clip because mean_conf
#   drops below 0.010 even when GM is perfectly detected (partial tooth echo).
#   GM conf check blocked 0 frames — all QS>=0.20 frames already have GM>=0.10.
DEFAULT_QS_THRESH          = 0.20   # quality_score — SOLE validity criterion
DEFAULT_STATIC_THRESH_MM   = 0.025  # mm — below this = static (~1 px at 27µm)
DEFAULT_BATCH_SIZE         = 16     # frames per inference batch
MIN_SWEEP_FRAMES           = 5      # warn if fewer valid frames than this


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

# ── Import architecture builders (same pattern as evaluate_mkunet_ablation.py) ─
# mkunet_ablation_train.py must be in the same directory as this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mkunet_ablation_train import (
        ALL_FEATURES as _TRAIN_ALL_FEATURES,
        NUM_CLASSES  as _TRAIN_NUM_CLASSES,
        IMAGE_HEIGHT as _TRAIN_H,
        IMAGE_WIDTH  as _TRAIN_W,
        build_mkunet_lgag_sase,
    )
    _BUILDERS_AVAILABLE = True
    # Use constants from training script (authoritative)
    ALL_FEATURES  = _TRAIN_ALL_FEATURES
    NUM_CLASSES   = _TRAIN_NUM_CLASSES
    MODEL_INPUT_SIZE = _TRAIN_H          # should be 256
    print("✓ Imported builders from mkunet_ablation_train.py")
except ImportError as _e:
    _BUILDERS_AVAILABLE = False
    NUM_CLASSES = 2
    print(f"WARNING: mkunet_ablation_train.py not found ({_e}).\n"
          f"  → Place mkunet_ablation_train.py in the same folder as this script.")


class MKUNetInference:
    """
    Loads MKUNet_LGAG_SASE by REBUILDING the architecture then loading weights.

    This mirrors evaluate_mkunet_ablation.py exactly:
        model, output_order = build_mkunet_lgag_sase(shape, features, num_classes)
        model.load_weights(weights_path)

    This avoids the 'Unknown layer: MKIRBlock' error that occurs when using
    tf.keras.models.load_model() on a .h5 file with custom layers.
    """

    def __init__(self, model_path: str,
                 architecture: str = 'MKUNet_LGAG_SASE'):
        print(f"[ModelLoader] Architecture : {architecture}")
        print(f"[ModelLoader] Weights path : {model_path}")

        if not _BUILDERS_AVAILABLE:
            raise RuntimeError(
                "Cannot load model: mkunet_ablation_train.py is missing.\n"
                "Copy it into the same directory as video_inference_pipeline.py")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights file not found: {model_path}")

        t0 = time.time()
        shape = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)

        # ── Step 1: rebuild the graph (no weights yet) ────────────────────────
        print(f"[ModelLoader] Rebuilding {architecture} graph …")
        self.model, self.output_names = build_mkunet_lgag_sase(
            shape, ALL_FEATURES, NUM_CLASSES)

        # ── Step 2: load weights into the rebuilt graph ───────────────────────
        print(f"[ModelLoader] Loading weights …")
        try:
            self.model.load_weights(model_path)
            print("[ModelLoader] ✓ Weights loaded (full match)")
        except Exception as e1:
            print(f"[ModelLoader] ⚠  Full load failed: {e1}")
            try:
                self.model.load_weights(model_path,
                                        by_name=True, skip_mismatch=True)
                print("[ModelLoader] ✓ Weights loaded (by_name, skipped mismatches)")
            except Exception as e2:
                raise RuntimeError(
                    f"Weight loading failed: {e2}\n"
                    f"Ensure the .h5 was saved from MKUNet_LGAG_SASE training.") from e2

        # ── Step 3: parse output feature order from model output names ────────
        # build_mkunet_lgag_sase returns output_names list directly (2nd return val)
        # but also double-check from model.output_names
        parsed = []
        for name in self.model.output_names:
            feat = name.replace('_output', '').split('/')[-1].replace('_output','')
            parsed.append(feat if feat in ALL_FEATURES else name)
        if all(f in ALL_FEATURES for f in parsed):
            self.output_names = parsed
        # else keep the list returned by the builder

        print(f"[ModelLoader] Output order : {self.output_names}")
        print(f"[ModelLoader] Parameters   : {self.model.count_params():,}")
        print(f"[ModelLoader] Ready in {time.time()-t0:.1f}s")

    def predict_batch(self, frames_256: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on a batch of preprocessed frames.

        Parameters
        ----------
        frames_256 : np.ndarray, shape (B, 256, 256, 3), float32, [0,1]

        Returns
        -------
        dict  {feature_name: np.ndarray (B, 256, 256, NUM_CLASSES)}
        """
        raw = self.model(frames_256, training=False)

        # raw is either a list of tensors or a single tensor
        if isinstance(raw, (list, tuple)):
            outputs = [r.numpy() for r in raw]
        else:
            outputs = [raw.numpy()]

        return {name: arr for name, arr in zip(self.output_names, outputs)}


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class FrameExtractor:
    """
    Decodes an MP4 clip and yields preprocessed frames for model inference.

    Preprocessing (exactly matching training pipeline):
        1. Resize to 256×256 using bilinear interpolation.
        2. Convert to float32.
        3. Min-max normalize to [0, 1] per frame.

    The original frame dimensions are stored for coordinate back-projection.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps          = cap.get(cv2.CAP_PROP_FPS)
        self.orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"[FrameExtractor] {Path(video_path).name}")
        print(f"  Frames: {self.total_frames}  |  FPS: {self.fps:.1f}  |  "
              f"Size: {self.orig_width}×{self.orig_height}")

    def iter_batches(self, batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Generator that yields (frame_indices, raw_frames, preprocessed_batch).

        Yields
        ------
        frame_indices : List[int]        — 0-based frame indices in this batch
        raw_frames    : np.ndarray (B, H, W, 3) uint8 — original frames (BGR)
        batch         : np.ndarray (B, 256, 256, 3) float32 — model-ready
        """
        cap = cv2.VideoCapture(self.video_path)
        batch_idx  = []
        batch_raw  = []
        batch_prep = []
        frame_idx  = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert BGR → RGB (model was trained on RGB)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Resize to 256×256
            frame_256 = cv2.resize(frame_rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                                   interpolation=cv2.INTER_LINEAR)

            # Normalize [0, 1] per frame (min-max)
            f32 = frame_256.astype(np.float32)
            mn, mx = f32.min(), f32.max()
            if mx - mn > 1e-6:
                f32 = (f32 - mn) / (mx - mn)
            # else: uniform frame — stays as-is (will get low confidence)

            batch_idx.append(frame_idx)
            batch_raw.append(frame_rgb)
            batch_prep.append(f32)

            if len(batch_idx) == batch_size:
                yield (batch_idx,
                       np.stack(batch_raw,  axis=0),
                       np.stack(batch_prep, axis=0))
                batch_idx, batch_raw, batch_prep = [], [], []

            frame_idx += 1

        # Yield remaining frames
        if batch_idx:
            yield (batch_idx,
                   np.stack(batch_raw,  axis=0),
                   np.stack(batch_prep, axis=0))

        cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class LandmarkExtractor:
    """
    Extracts sub-pixel landmark coordinates from softmax heatmaps using
    soft-argmax (differentiable argmax), which is more accurate than
    hard argmax — especially for jittery landmarks like CEJ and ABC.

    The foreground class-1 probability map is used for all operations.
    """

    @staticmethod
    def soft_argmax(heatmap: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute soft-argmax on a 2D probability map.

        Parameters
        ----------
        heatmap : np.ndarray (H, W)  — class-1 foreground probability

        Returns
        -------
        row : float   — soft row coordinate (0-indexed)
        col : float   — soft column coordinate (0-indexed)
        peak_conf : float — maximum probability in the heatmap
        """
        h, w = heatmap.shape
        total = heatmap.sum()
        peak_conf = float(heatmap.max())

        if total < 1e-8:
            # Degenerate heatmap — landmark not visible
            return h / 2.0, w / 2.0, 0.0

        # Normalise to sum=1 (probability distribution over spatial positions)
        prob = heatmap / total

        # Coordinate grids
        rows = np.arange(h, dtype=np.float32)
        cols = np.arange(w, dtype=np.float32)

        # Expected coordinates
        row = float(np.sum(prob.sum(axis=1) * rows))
        col = float(np.sum(prob.sum(axis=0) * cols))

        return row, col, peak_conf

    @staticmethod
    def to_mm(row: float, col: float) -> Tuple[float, float]:
        """
        Convert (row, col) in 256×256 pixel space to physical (x_mm, y_mm).

        Convention:
            x_mm = horizontal (col direction, left → right)
            y_mm = vertical   (row direction, top → bottom, i.e. crown → bone)
        """
        x_mm = col * PIXEL_SPACING_MM
        y_mm = row * PIXEL_SPACING_MM
        return x_mm, y_mm

    @classmethod
    def extract(cls, outputs: Dict[str, np.ndarray],
                frame_i: int) -> Dict[str, dict]:
        """
        Extract landmarks and region stats for a single frame.

        Parameters
        ----------
        outputs : dict {feature: (B, 256, 256, NC)} — full batch inference
        frame_i : int — which element of the batch to process

        Returns
        -------
        landmarks : dict  {GM/CEJ/ABC: {row,col,x_mm,y_mm,confidence,visible}}
        regions   : dict  {TOOTH/GINGIVA/BONE: {mean_conf, pixel_area}}
        """
        landmarks = {}
        regions   = {}

        for feat in POINT_FEATURES:
            if feat not in outputs:
                continue
            heatmap   = outputs[feat][frame_i, :, :, FG_CLASS]  # (256, 256)
            row, col, peak = cls.soft_argmax(heatmap)
            x_mm, y_mm    = cls.to_mm(row, col)
            landmarks[feat] = {
                'row'       : round(row,  2),
                'col'       : round(col,  2),
                'x_mm'      : round(x_mm, 4),
                'y_mm'      : round(y_mm, 4),
                'confidence': round(peak, 4),
                'visible'   : bool(peak >= 0.10),  # soft visibility threshold
            }

        for feat in REGION_FEATURES:
            if feat not in outputs:
                continue
            prob_map  = outputs[feat][frame_i, :, :, FG_CLASS]  # (256, 256)
            binary    = (prob_map >= 0.5).astype(np.uint8)
            mean_conf = float(prob_map.mean())
            pixel_area = int(binary.sum())
            regions[feat] = {
                'mean_conf' : round(mean_conf,  4),
                'pixel_area': pixel_area,
            }

        return landmarks, regions


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORER
# ═══════════════════════════════════════════════════════════════════════════════

class QualityScorer:
    """
    Computes a composite [0,1] quality score per frame combining:
        - TOOTH region mean confidence     (weight 0.40)
        - GM landmark peak confidence      (weight 0.35)
        - ABC landmark peak confidence     (weight 0.15)  — noisy, lower weight
        - GINGIVA region mean confidence   (weight 0.10)

    This score is used for:
        1. On-tooth detection thresholding
        2. Blending weights in the panorama compositor
        3. Selecting the best static hold representative frame
    """

    WEIGHTS = {
        'TOOTH_conf'   : 0.40,
        'GM_conf'      : 0.35,
        'ABC_conf'     : 0.15,
        'GINGIVA_conf' : 0.10,
    }

    @classmethod
    def compute(cls, landmarks: Dict[str, dict],
                regions: Dict[str, dict]) -> float:
        """
        Compute composite quality score from extracted landmarks and regions.
        """
        tooth_c   = regions.get('TOOTH',   {}).get('mean_conf',   0.0)
        gm_c      = landmarks.get('GM',    {}).get('confidence',   0.0)
        abc_c     = landmarks.get('ABC',   {}).get('confidence',   0.0)
        gingiva_c = regions.get('GINGIVA', {}).get('mean_conf',   0.0)

        score = (cls.WEIGHTS['TOOTH_conf']   * tooth_c   +
                 cls.WEIGHTS['GM_conf']      * gm_c      +
                 cls.WEIGHTS['ABC_conf']     * abc_c     +
                 cls.WEIGHTS['GINGIVA_conf'] * gingiva_c)
        return round(float(score), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# VALID FRAME DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ValidFrameDetector:
    """
    Two-stage valid frame detection.

    Stage 1 — On-tooth detection  (PRIMARY: quality_score threshold)
    ─────────────────────────────────────────────────────────────────
    A frame is ON-TOOTH when quality_score >= qs_thresh (default 0.30).

    Why quality_score and not raw TOOTH/GM confidences?
    • TOOTH mean_conf is the mean foreground probability over ALL 256×256 pixels.
      The tooth occupies ~2% of the image, so maximum observed values are
      0.02–0.03 — far below any intuitive threshold like 0.35.
    • GM peak confidence saturates at 1.0 even in off-tooth frames because
      soft-argmax always finds the brightest pixel regardless of content.
    • quality_score (weighted combination) produces a clean bimodal distribution:
        off-tooth ≈ 0.15,  on-tooth ≈ 0.51
      A threshold of 0.30 separates the two clusters in all tested clips.

    SECONDARY checks (tooth_conf_thresh, gm_conf_thresh) are applied on top
    as sanity filters only — tuned to their actual value ranges (≤ 0.05).

    Stage 2 — Sweeping vs static
    ─────────────────────────────
    Inter-frame GM displacement (mm) computed between consecutive on-tooth frames.
    • disp < static_thresh_mm  →  static hold (collapse to 1 representative)
    • disp ≥ static_thresh_mm  →  sweeping (source frames for panorama)
    Longest contiguous sweep window (allowing ≤2 static-hold gaps) is selected.
    """

    def __init__(self,
                 qs_thresh        : float = DEFAULT_QS_THRESH,
                 static_thresh_mm : float = DEFAULT_STATIC_THRESH_MM):
        self.qs_thresh        = qs_thresh
        self.static_thresh_mm = static_thresh_mm

    def detect(self, frame_results: Dict[int, dict]) \
            -> Tuple[List[int], List[int], List[int], Optional[int]]:
        """
        Returns:
            valid_frames  : on-tooth frame indices (sorted)
            sweep_frames  : frames classified as sweeping
            static_frames : on-tooth but stationary
            static_rep    : best static hold frame index (highest quality_score)
        """
        sorted_idx = sorted(frame_results.keys())
        n_total    = len(sorted_idx)

        # Print distribution summary to aid threshold tuning
        qs_vals    = [frame_results[i]['quality_score'] for i in sorted_idx]
        tooth_vals = [frame_results[i]['regions'].get('TOOTH',{})
                      .get('mean_conf', 0.0) for i in sorted_idx]
        print(f"[ValidFrameDetector] quality_score  "
              f"min={min(qs_vals):.3f}  "
              f"med={float(np.median(qs_vals)):.3f}  "
              f"max={max(qs_vals):.3f}")
        print(f"[ValidFrameDetector] TOOTH mean_conf "
              f"min={min(tooth_vals):.4f}  "
              f"med={float(np.median(tooth_vals)):.4f}  "
              f"max={max(tooth_vals):.4f}")

        # ── Stage 1: on-tooth — quality_score ONLY ──────────────────────────
        # No secondary checks: TOOTH mean_conf and GM conf add no signal and
        # silently block valid frames (root-cause confirmed across 7 test clips).
        valid_frames = []
        for idx in sorted_idx:
            r        = frame_results[idx]
            on_tooth = r['quality_score'] >= self.qs_thresh
            r['on_tooth'] = on_tooth
            if on_tooth:
                valid_frames.append(idx)

        pct = 100 * len(valid_frames) / max(n_total, 1)
        print(f"[ValidFrameDetector] Stage 1 — on-tooth: "
              f"{len(valid_frames)} / {n_total} ({pct:.1f}%)  "
              f"[QS≥{self.qs_thresh}]")

        if not valid_frames:
            print("[ValidFrameDetector] ⚠  No on-tooth frames detected.\n"
                  "  Suggestions:\n"
                  f"  • Lower --qs_thresh (current={self.qs_thresh}, try 0.20)\n"
                  f"  • Lower --tooth_conf_thresh (current={self.tooth_conf_thresh})\n"
                  f"  • Check quality_score distribution above — if max < 0.30,\n"
                  f"    the model may not be detecting anatomy in this clip.")
            return [], [], [], None

        if len(valid_frames) < MIN_SWEEP_FRAMES:
            print(f"[ValidFrameDetector] ⚠  Only {len(valid_frames)} on-tooth frames "
                  f"— clip may be too short or poorly captured.")

        # ── Stage 2: inter-frame GM displacement ─────────────────────────────
        displacements = {}
        prev_idx = None
        for idx in valid_frames:
            if prev_idx is None:
                displacements[idx] = 0.0
            else:
                gm_c = frame_results[idx]['landmarks'].get('GM', {})
                gm_p = frame_results[prev_idx]['landmarks'].get('GM', {})
                if gm_c.get('visible', False) and gm_p.get('visible', False):
                    dx = gm_c['x_mm'] - gm_p['x_mm']
                    dy = gm_c['y_mm'] - gm_p['y_mm']
                    displacements[idx] = float(np.sqrt(dx**2 + dy**2))
                else:
                    displacements[idx] = 0.0
            prev_idx = idx

        for idx in valid_frames:
            frame_results[idx]['gm_displacement_mm'] = round(displacements[idx], 5)

        # Classify sweeping vs static
        sweep_frames  = []
        static_frames = []
        for idx in valid_frames:
            is_sweep = displacements[idx] >= self.static_thresh_mm
            frame_results[idx]['is_sweeping'] = is_sweep
            if is_sweep:
                sweep_frames.append(idx)
            else:
                static_frames.append(idx)

        disp_vals = [displacements[i] for i in valid_frames if displacements[i] > 0]
        if disp_vals:
            print(f"[ValidFrameDetector] Stage 2 — GM displacement "
                  f"med={float(np.median(disp_vals)):.4f}mm  "
                  f"max={max(disp_vals):.4f}mm  "
                  f"| sweep={len(sweep_frames)}  static={len(static_frames)}")

        # ── Find longest contiguous sweep window ──────────────────────────────
        sweep_window = self._find_sweep_window(valid_frames, frame_results)

        # ── Best static representative ────────────────────────────────────────
        static_rep = None
        if static_frames:
            static_rep = max(static_frames,
                             key=lambda i: frame_results[i]['quality_score'])

        if sweep_window:
            span = sweep_window[1] - sweep_window[0] + 1
            print(f"[ValidFrameDetector] Sweep window: "
                  f"frames {sweep_window[0]}–{sweep_window[1]} "
                  f"(span={span}, sweep_frames={len(sweep_frames)})")
        if static_rep is not None:
            qs_rep = frame_results[static_rep]['quality_score']
            print(f"[ValidFrameDetector] Static rep: "
                  f"frame #{static_rep}  quality={qs_rep:.3f}")

        return valid_frames, sweep_frames, static_frames, static_rep

    def _find_sweep_window(self, valid_frames: List[int],
                           frame_results: Dict[int, dict]
                           ) -> Optional[Tuple[int, int]]:
        """
        Longest contiguous sweep window, allowing ≤2 static-hold gaps.
        """
        if not valid_frames:
            return None
        best_start = best_end = None
        best_len   = 0
        run_start  = None
        gap_count  = 0
        for idx in valid_frames:
            if frame_results[idx].get('is_sweeping', False):
                if run_start is None:
                    run_start = idx
                gap_count = 0
                run_end   = idx
                if run_end - run_start + 1 > best_len:
                    best_len   = run_end - run_start + 1
                    best_start = run_start
                    best_end   = run_end
            else:
                if run_start is not None:
                    gap_count += 1
                    if gap_count > 2:
                        run_start = None
                        gap_count = 0
        return (best_start, best_end) if best_start is not None else None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class VideoPanoramaPipeline:
    """
    Orchestrates frame extraction → inference → landmark extraction →
    quality scoring → valid frame detection for a single video.
    """

    def __init__(self, model: MKUNetInference,
                 qs_thresh        : float = DEFAULT_QS_THRESH,
                 static_thresh_mm : float = DEFAULT_STATIC_THRESH_MM,
                 batch_size       : int   = DEFAULT_BATCH_SIZE):
        self.model          = model
        self.batch_size     = batch_size
        self.valid_detector = ValidFrameDetector(qs_thresh, static_thresh_mm)

    def process_video(self, video_path: str,
                      output_dir: str) -> dict:
        """
        Full pipeline for one video. Saves JSON result and returns result dict.
        """
        extractor = FrameExtractor(video_path)
        stem      = Path(video_path).stem

        # ── Inference pass ────────────────────────────────────────────────────
        frame_results: Dict[int, dict] = {}
        t0 = time.time()
        total_batches = int(np.ceil(extractor.total_frames / self.batch_size))

        print(f"\n[Pipeline] Running inference on {extractor.total_frames} frames "
              f"({total_batches} batches, batch_size={self.batch_size})")

        for batch_idx_list, _, batch_prep in extractor.iter_batches(self.batch_size):
            # Model forward pass
            outputs = self.model.predict_batch(batch_prep)

            # Per-frame post-processing
            for i, frame_idx in enumerate(batch_idx_list):
                # Extract single-frame slice from batch outputs
                single = {k: v[i:i+1] for k, v in outputs.items()}

                landmarks, regions = LandmarkExtractor.extract(single, 0)
                quality_score      = QualityScorer.compute(landmarks, regions)

                frame_results[frame_idx] = {
                    'landmarks'         : landmarks,
                    'regions'           : regions,
                    'quality_score'     : quality_score,
                    'on_tooth'          : False,   # filled by ValidFrameDetector
                    'is_sweeping'       : False,   # filled by ValidFrameDetector
                    'gm_displacement_mm': 0.0,     # filled by ValidFrameDetector
                }

            # Progress
            processed = batch_idx_list[-1] + 1
            pct = 100 * processed / extractor.total_frames
            elapsed = time.time() - t0
            fps_inf = processed / max(elapsed, 1e-3)
            print(f"  [{processed:5d}/{extractor.total_frames}]  "
                  f"{pct:5.1f}%  |  {fps_inf:5.1f} frames/s", end='\r')

        print(f"\n[Pipeline] Inference done in {time.time()-t0:.1f}s")

        # ── Valid frame detection ─────────────────────────────────────────────
        valid_frames, sweep_frames, static_frames, static_rep = \
            self.valid_detector.detect(frame_results)

        # Determine sweep window bounds
        sweep_window = None
        if sweep_frames:
            sweep_window = [min(sweep_frames), max(sweep_frames)]

        # ── Assemble result dict ──────────────────────────────────────────────
        result = {
            'video_path'             : str(video_path),
            'total_frames'           : extractor.total_frames,
            'fps'                    : round(extractor.fps, 3),
            'pixel_spacing_um'       : round(PIXEL_SPACING_UM, 4),
            'pixel_spacing_mm'       : round(PIXEL_SPACING_MM, 6),
            'original_hw'            : [extractor.orig_height, extractor.orig_width],
            'model_input_size'       : MODEL_INPUT_SIZE,
            'valid_frames'           : valid_frames,
            'sweep_window'           : sweep_window,
            'sweep_frames'           : sweep_frames,
            'static_frames'          : static_frames,
            'static_representative'  : static_rep,
            'n_valid'                : len(valid_frames),
            'n_sweep'                : len(sweep_frames),
            'n_static'               : len(static_frames),
            'frames'                 : {str(k): v for k, v in frame_results.items()},
        }

        # ── Save JSON ─────────────────────────────────────────────────────────
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'{stem}_inference.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[Pipeline] Result saved → {out_path}")

        # ── Print summary ─────────────────────────────────────────────────────
        self._print_summary(result)

        return result

    @staticmethod
    def _print_summary(result: dict):
        print("\n" + "─" * 60)
        print(f"  VIDEO SUMMARY: {Path(result['video_path']).name}")
        print("─" * 60)
        print(f"  Total frames     : {result['total_frames']}")
        print(f"  FPS              : {result['fps']}")
        print(f"  On-tooth frames  : {result['n_valid']} "
              f"({100*result['n_valid']/max(result['total_frames'],1):.1f}%)")
        print(f"  Sweep frames     : {result['n_sweep']}")
        print(f"  Static frames    : {result['n_static']}")
        if result['sweep_window']:
            sw = result['sweep_window']
            fps = result['fps']
            dur = (sw[1] - sw[0]) / max(fps, 1)
            print(f"  Sweep window     : frames {sw[0]}–{sw[1]} "
                  f"({dur:.2f}s at {fps:.0f} FPS)")
        if result['static_representative'] is not None:
            sr = result['static_representative']
            qs = result['frames'][str(sr)]['quality_score']
            print(f"  Static rep frame : #{sr}  (quality={qs:.3f})")
        print("─" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='Periodontal Ultrasound Panorama — Step 1: Video Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Model
    p.add_argument('--model_path', type=str, required=True,
                   help='Path to saved MKUNet weights (.h5)')
    p.add_argument('--architecture', type=str, default='MKUNet_LGAG_SASE',
                   help='Architecture name matching mkunet_ablation_train.py '
                        '(default: MKUNet_LGAG_SASE)')

    # Input — mutually exclusive: single video or directory
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument('--video_path', type=str,
                     help='Path to a single MP4 file')
    grp.add_argument('--video_dir',  type=str,
                     help='Directory containing MP4 files (processed recursively)')

    # Output
    p.add_argument('--output_dir', type=str, default='./inference_results',
                   help='Directory to save JSON results (default: ./inference_results)')

    # Thresholds
    p.add_argument('--qs_thresh', type=float,
                   default=DEFAULT_QS_THRESH,
                   help=f'Quality-score threshold for on-tooth detection. '
                        f'Bimodal: off-tooth≈0.15, on-tooth≈0.51. '
                        f'Default {DEFAULT_QS_THRESH} catches all valid frames.')
    p.add_argument('--static_thresh_mm', type=float,
                   default=DEFAULT_STATIC_THRESH_MM,
                   help=f'Min GM inter-frame displacement (mm) to classify as '
                        f'sweeping vs static hold (default: {DEFAULT_STATIC_THRESH_MM})')

    # Performance
    p.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                   help=f'Frames per inference batch (default: {DEFAULT_BATCH_SIZE})')

    # GPU
    p.add_argument('--gpu', type=str, default='0',
                   help='GPU index to use (default: 0). Set -1 to force CPU.')

    return p.parse_args()


def setup_gpu(gpu_idx: str):
    if gpu_idx == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("[GPU] Running on CPU")
        return
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("[GPU] No GPUs found — running on CPU")
        return
    try:
        tf.config.set_visible_devices(gpus[int(gpu_idx)], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[int(gpu_idx)], True)
        print(f"[GPU] Using GPU {gpu_idx}: {gpus[int(gpu_idx)].name}")
    except (IndexError, RuntimeError) as e:
        print(f"[GPU] WARNING: {e} — falling back to CPU")


def collect_videos(video_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(video_dir):
        for f in sorted(files):
            if f.lower().endswith('.mp4'):
                paths.append(os.path.join(root, f))
    return paths


def main():
    args = parse_args()
    setup_gpu(args.gpu)

    # Load model once — rebuild graph then load weights (no custom_objects needed)
    model = MKUNetInference(args.model_path, architecture=args.architecture)

    # Build pipeline
    pipeline = VideoPanoramaPipeline(
        model            = model,
        qs_thresh        = args.qs_thresh,
        static_thresh_mm = args.static_thresh_mm,
        batch_size       = args.batch_size,
    )

    # Collect videos
    if args.video_path:
        videos = [args.video_path]
    else:
        videos = collect_videos(args.video_dir)
        print(f"[Main] Found {len(videos)} MP4 files in {args.video_dir}")

    if not videos:
        print("[Main] ERROR: No videos found.")
        sys.exit(1)

    # Process
    t_total = time.time()
    successes, failures = [], []

    for i, vp in enumerate(videos, 1):
        print(f"\n{'═'*60}")
        print(f"VIDEO {i}/{len(videos)}: {Path(vp).name}")
        print(f"{'═'*60}")
        try:
            pipeline.process_video(vp, args.output_dir)
            successes.append(vp)
        except Exception as e:
            import traceback
            print(f"[Main] ERROR processing {vp}: {e}")
            traceback.print_exc()
            failures.append(vp)

    # Final summary
    elapsed = time.time() - t_total
    print(f"\n{'═'*60}")
    print(f"DONE — {len(successes)} succeeded, {len(failures)} failed")
    print(f"Total time: {elapsed/60:.1f} min")
    if failures:
        print("Failed videos:")
        for f in failures:
            print(f"  {f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'═'*60}")


if __name__ == '__main__':
    main()
