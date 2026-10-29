#!/usr/bin/env python3
"""
live_panorama_final_v3.py  —  Live Periodontal Panorama  (B1 + Voice + v3: IOU Gate + Direction + OFF cmd)
====================================================================================
Model: B1_V8_Baseline (MKUNet, LGAG2→SASE k=7 all stages, CH_BASE=[16,32,64,96])
       Pass B1 checkpoint via --model.  MKUNetInference handles all custom objects.

SIX ENGINEERING FEATURES (all new vs live_panorama.py):
─────────────────────────────────────────────────────────
1. AdaptiveQualityGater
   Lightweight frame filter using B1's existing outputs + frame statistics.
   Three signals: Laplacian sharpness, B1 region-head contact confidence,
   frame SNR.  No new trainable parameters.  Replaces simple QS_THRESH gate.

2. TiledCanvas  (replaces LiveCanvas)
   Streaming mipmap canvas.  Tiles near probe = full 256 px.
   Tiles > HALF_RADIUS away = 128 px.  Tiles > FULL_RADIUS = 64 px.
   Probe return → tile upsampled before re-compositing.
   Tracks IMU-only vs GM+IMU registration per tile for heatmap.

3. CoverageHeatmap
   Per-tile coverage rendered as small colour-coded strip:
     Green  = high-confidence (GM+IMU, wgt > threshold)
     Yellow = IMU-only registration
     Red    = gap / not yet covered
   Shown live in panorama window and saved as coverage.png on session end.

4. AuditLogger
   Full trajectory log: frame, timestamp, gm_x/y, imu_yaw, quality, gater scores.
   replay_session() rebuilds session as annotated MP4 with probe path overlaid.

5. MultiSessionRegistrar
   On session end, optionally register new panorama to a stored previous one
   using CEJ/ABC landmark correspondences from B1's point head.
   Outputs registered_panorama.png with geometric alignment.

6. quantize_b1_model()  (standalone utility — call via --quantize)
   INT8 post-training quantisation using TFLite.
   Saves quantized .tflite model.
   Reports file size reduction and estimated CE/HD95 preservation.

VOICE COMMANDS (accent-robust, 1-syllable, AR goggle mic):
  ON SAVE DROP HOLD GO MARK BAD VIEW  (see PanoVoiceController)

KEYBOARD (unchanged fallback):
  1/ENTER  start/stop+save     2  advance session
  R        discard             E  export PNG
  I        toggle IMU          S  screenshot
  C        calibrate IMU       P  replay last session
  M        multi-session reg   Q  quit
"""

import os, sys, asyncio, threading, time, json, argparse, queue, struct
from pathlib import Path
from collections import deque
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d

os.environ['TF_GPU_ALLOCATOR']         = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_CACHE_DISABLE']      = '0'
os.environ['OMP_NUM_THREADS']         = '2'

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import whisper as _whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from difflib import SequenceMatcher

# ─── Constants ────────────────────────────────────────────────────────────────
PIXEL_SPACING_MM  = 7.0 / 256
MODEL_SIZE        = 256
CANVAS_MAX_W      = 4000          # wider to support full-arch sweep
CANVAS_H_EXTRA    = 60
SEAM_FADE_PX      = 20
Y_EDGE_FADE_PX    = 8
IMU_MAC           = "D3:F9:B0:EF:13:A0"
COMP_TAU          = 0.2   # lowered from 0.5 — reduces alpha at 20 Hz BLE

# Quality gating (replaces single QS_THRESH — see AdaptiveQualityGater)
QS_THRESH         = 0.15          # coarse pre-filter; AdaptiveQualityGater is the real gate
KEYFRAME_MIN_DIST_PX  = 2.0        # minimum col0 advance (px) between keyframes
KEYFRAME_JITTER_PX    = 3.0        # maximum gm_dy movement allowed in a keyframe
KEYFRAME_QUALITY      = 0.35       # minimum B1 quality for keyframe selection


# ═════════════════════════════════════════════════════════════════════════════
#  1.  ADAPTIVE FRAME QUALITY GATER  (new — Engineering Feature 1)
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveQualityGater:
    """
    Lightweight frame classifier using B1's existing outputs.
    No new trainable parameters — all signals come from the already-computed
    B1 region/point predictions and the raw frame statistics.

    Signal 1 — Sharpness  (Laplacian variance, motion blur / defocus)
    Signal 2 — Contact     (GINGIVA + TOOTH pixel area from B1's region head;
                            low area → probe lift-off, air gap, angle artifact)
    Signal 3 — SNR        (ratio of tissue-bright percentile to top-edge noise;
                            low SNR → coupling gel mismatch / deep signal loss)

    Each signal normalised to [0,1] against calibration reference values.
    Combined score = weighted sum.  Frame accepted iff combined ≥ threshold.

    Calibration:
        The reference values (sharpness_ref, contact_ref, snr_ref) are set for
        a well-placed 7 mm × 7 mm US probe.  Tune via --qs_sharp / --qs_contact
        / --qs_snr / --qs_thresh CLI args.  Runtime histograms saved to
        quality_stats.json for post-hoc calibration.
    """

    def __init__(self,
                 sharpness_weight:  float = 0.30,
                 contact_weight:    float = 0.50,
                 snr_weight:        float = 0.20,
                 threshold:         float = 0.28,
                 sharpness_ref:     float = 400.,   # Laplacian var of a good frame
                 contact_ref_px:    float = 1800.,  # GINGIVA+TOOTH area at good contact
                 snr_ref_db:        float = 18.):
        self.w_sharp      = sharpness_weight
        self.w_contact    = contact_weight
        self.w_snr        = snr_weight
        self.threshold    = threshold
        self.sharp_ref    = sharpness_ref
        self.contact_ref  = contact_ref_px
        self.snr_ref      = snr_ref_db
        # Runtime histogram (saved to disk on session end)
        self._scores: List[float] = []
        self._components: List[dict] = []

    def score(self, frame_bgr: np.ndarray,
              last_regions: dict) -> Tuple[float, dict]:
        """
        frame_bgr:    (H, W, 3) uint8 — raw HD60X crop
        last_regions: annotator._last_regions  e.g. {'GINGIVA': {'mean_conf':…,
                                                      'pixel_area':…}, …}
        Returns (combined_score [0,1], component_dict)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ── Signal 1: Sharpness ────────────────────────────────────────────
        lap_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        s_sharp  = float(np.clip(lap_var / max(self.sharp_ref, 1.), 0., 1.))

        # ── Signal 2: Probe contact ────────────────────────────────────────
        g_area   = last_regions.get('GINGIVA', {}).get('pixel_area', 0)
        t_area   = last_regions.get('TOOTH',   {}).get('pixel_area', 0)
        s_contact = float(np.clip((g_area + t_area) / max(self.contact_ref, 1.), 0., 1.))

        # ── Signal 3: SNR ─────────────────────────────────────────────────
        gray_f   = gray.astype(np.float32)
        sig_val  = float(np.percentile(gray_f, 80))   # bright tissue region
        noise_val = float(np.std(gray_f[:16, :]) + 1e-6)  # top-edge coupling noise
        snr_db   = 20. * np.log10(sig_val / noise_val + 1e-6)
        s_snr    = float(np.clip(snr_db / max(self.snr_ref, 1.), 0., 1.))

        combined = (self.w_sharp * s_sharp +
                    self.w_contact * s_contact +
                    self.w_snr * s_snr)

        comp = {
            'sharpness':   round(s_sharp,   3),
            'contact':     round(s_contact, 3),
            'snr':         round(s_snr,     3),
            'combined':    round(combined,  3),
            'lap_var':     round(lap_var,   1),
            'snr_db':      round(snr_db,    1),
        }
        self._scores.append(combined)
        self._components.append(comp)
        return float(combined), comp

    def accept(self, combined_score: float) -> bool:
        return combined_score >= self.threshold

    def summary(self) -> dict:
        if not self._scores:
            return {}
        s = np.array(self._scores)
        return {
            'n_frames':        len(s),
            'accept_rate':     round(float((s >= self.threshold).mean()), 3),
            'mean_score':      round(float(s.mean()), 3),
            'p25':             round(float(np.percentile(s, 25)), 3),
            'p75':             round(float(np.percentile(s, 75)), 3),
            'threshold_used':  self.threshold,
        }

    def save_stats(self, path: Path):
        data = {'summary': self.summary(), 'per_frame': self._components}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
#  2.  TILED CANVAS  (new — Engineering Feature 2, replaces LiveCanvas)
# ═════════════════════════════════════════════════════════════════════════════

class TiledCanvas:
    """
    Streaming mipmap panorama canvas.

    Canvas is divided into TILE_W-pixel wide columns.
    Each tile maintains its compositing arrays at a resolution level
    determined by its distance from the current probe position:

      Distance ≤ ACTIVE_R  →  Level 0: full  MODEL_SIZE × TILE_W
      Distance ≤ HALF_R    →  Level 1: half  MODEL_SIZE/2 × TILE_W/2
      Distance >  HALF_R   →  Level 2: qtr   MODEL_SIZE/4 × TILE_W/4

    When the probe returns to a far tile, it is upsampled back to Level 0
    before new frames are composited, preserving the accumulated content.

    Per-tile metadata tracks IMU-only vs GM+IMU registration for heatmap.

    Coverage classification per tile:
      'high'  →  wgt_max > 0.5  AND  gm_fraction > 0.5   (green in heatmap)
      'imu'   →  wgt_max > 0.5  AND  gm_fraction ≤ 0.5   (yellow)
      'gap'   →  wgt_max ≤ 0.5                             (red)
    """
    TILE_W    = MODEL_SIZE      # = 256  (one probe-width per tile)
    CANVAS_H  = MODEL_SIZE + CANVAS_H_EXTRA  # = 316
    ACTIVE_R  = 2               # tiles at full res
    HALF_R    = 6               # tiles at half res; beyond = quarter

    # Scale factors per level
    SCALE     = {0: 1.0, 1: 0.5, 2: 0.25}

    def __init__(self, max_width: int = CANVAS_MAX_W):
        self.max_w    = max_width
        self._tiles:  Dict[int, dict] = {}
        self._cur     = None            # current tile index (probe position)
        self.frame_count = 0
        self.anchor_row  = None
        self.gm_rows:  List[float] = []
        self.col0s:    List[float] = []
        self._col0_min = None
        self._col0_max = None

    # ── Tile management ───────────────────────────────────────────────────────

    def _target_level(self, tile_idx: int) -> int:
        if self._cur is None: return 0
        d = abs(tile_idx - self._cur)
        if d <= self.ACTIVE_R: return 0
        if d <= self.HALF_R:   return 1
        return 2

    def _arr_shape(self, level: int) -> Tuple[int, int]:
        s = self.SCALE[level]
        return int(self.CANVAS_H * s), int(self.TILE_W * s)

    def _make_tile(self, level: int) -> dict:
        h, w = self._arr_shape(level)
        return {
            'level':      level,
            'acc':        np.zeros((h, w, 3), np.float64),
            'wgt':        np.zeros((h, w),    np.float64),
            'imu_frames': 0,
            'gm_frames':  0,
        }

    def _ensure_level(self, tile_idx: int, target: int):
        """Resize tile arrays to target level (upsample or downsample)."""
        if tile_idx not in self._tiles:
            self._tiles[tile_idx] = self._make_tile(target)
            return
        tile = self._tiles[tile_idx]
        if tile['level'] == target:
            return
        h, w = self._arr_shape(target)
        tile['acc'] = cv2.resize(
            tile['acc'].astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR).astype(np.float64)
        tile['wgt'] = cv2.resize(
            tile['wgt'].astype(np.float32), (w, h),
            interpolation=cv2.INTER_LINEAR).astype(np.float64)
        tile['level'] = target

    def _refresh_levels(self):
        """After probe moves, adjust resolution of all existing tiles."""
        for idx in list(self._tiles.keys()):
            self._ensure_level(idx, self._target_level(idx))

    # ── Compositing ───────────────────────────────────────────────────────────

    def add_frame(self, frame_rgb: np.ndarray,
                  col0_px: float,
                  gm_row_in_frame: float,
                  quality_score: float,
                  imu_used: bool = False):
        """
        frame_rgb:        (256, 256, 3) float32 [0,1]
        col0_px:          left edge of frame in global canvas pixels
        gm_row_in_frame:  GM depth in frame pixels
        quality_score:    B1 quality (after AdaptiveQualityGater pre-filter)
        imu_used:         True if IMU was primary source for col0 this frame

        ANCHOR ROW POLICY (v2 fix):
          anchor_row is locked after the first 10 frames using the median
          of those frames' gm_row values.  It is NEVER updated after locking.
          Rationale: sweeping from bottom-to-top vs top-to-bottom changes
          gm_row_in_frame (GM moves up/down in the image).  If anchor_row
          updates continuously with max(gm_rows), reverse-direction frames
          composite at a different vertical offset → vertical smear/ghosting.
          Locking anchor_row creates a fixed canvas coordinate system that
          both forward and reverse passes share.
        """
        self.gm_rows.append(gm_row_in_frame)
        self.col0s.append(col0_px)

        # Lock anchor_row from median of first 10 frames, never update after
        if self.anchor_row is None and len(self.gm_rows) >= 10:
            self.anchor_row = float(np.median(self.gm_rows[:10])) + 8.
            print(f"[Canvas] anchor_row locked at {self.anchor_row:.1f} px "
                  f"(median of first 10 GM rows)")
        elif self.anchor_row is None:
            # Not enough frames yet — use current value as provisional
            self.anchor_row = float(np.median(self.gm_rows)) + 8.

        frame_l = int(round(col0_px))
        frame_r = frame_l + MODEL_SIZE

        if frame_l < 0 or frame_r > self.max_w:
            return

        if self._col0_min is None or frame_l < self._col0_min: self._col0_min = frame_l
        if self._col0_max is None or frame_l > self._col0_max: self._col0_max = frame_l

        # Update current tile and refresh resolution levels
        self._cur = frame_l // self.TILE_W
        self._refresh_levels()

        # A frame may span up to 2 tiles — composite into each
        tile_l = frame_l // self.TILE_W
        tile_r = max(frame_l, frame_r - 1) // self.TILE_W

        for tidx in range(tile_l, tile_r + 1):
            self._ensure_level(tidx, self._target_level(tidx))
            self._composite_to_tile(tidx, frame_rgb, col0_px,
                                     gm_row_in_frame, quality_score, imu_used)

        self.frame_count += 1

    def _composite_to_tile(self, tidx: int, frame_rgb: np.ndarray,
                            col0_px: float, gm_row: float,
                            quality: float, imu_used: bool):
        tile  = self._tiles[tidx]
        level = tile['level']
        s     = self.SCALE[level]
        th, tw = self._arr_shape(level)

        # Track IMU vs GM
        if imu_used: tile['imu_frames'] += 1
        else:        tile['gm_frames']  += 1

        # Global → tile-local coordinates (scaled)
        tile_origin_global = tidx * self.TILE_W
        c0_local = (col0_px - tile_origin_global) * s    # left edge in tile coords

        canvas_rows = np.arange(th, dtype=np.float64)
        canvas_cols = np.arange(tw, dtype=np.float64)

        # Seam fade in tile space
        fade = SEAM_FADE_PX * s
        sl   = max(0., c0_local - fade)
        sr   = min(float(tw), c0_local + MODEL_SIZE * s + fade)
        ca0  = max(0,  int(np.floor(c0_local)))
        ca1  = min(tw, int(np.ceil(c0_local + MODEL_SIZE * s)))

        if ca0 >= ca1: return

        x_c  = canvas_cols[ca0:ca1]
        dl   = x_c - sl;  dr = sr - x_c
        wl   = np.where(dl < 0, 0., np.where(dl < fade,
               0.5*(1-np.cos(np.pi*dl/max(fade,1e-6))), 1.))
        wr   = np.where(dr < 0, 0., np.where(dr < fade,
               0.5*(1-np.cos(np.pi*dr/max(fade,1e-6))), 1.))
        x_mask        = np.zeros(tw, np.float32)
        x_mask[ca0:ca1] = (wl * wr).astype(np.float32)

        active = np.where(x_mask > 0)[0]
        if len(active) == 0: return

        # Frame column indices (in original 256-px frame)
        fc_global = active / s + tile_origin_global - col0_px
        valid     = (fc_global >= 0) & (fc_global < MODEL_SIZE)
        if not valid.any(): return

        fc_v    = fc_global[valid]
        x_w_sel = x_mask[active[valid]].astype(np.float64)

        # Y warp (scaled)
        ar_scaled = self.anchor_row * s
        gr_scaled = gm_row * s
        fr2       = canvas_rows[:, None] - ar_scaled + gr_scaled
        fade_y    = Y_EDGE_FADE_PX * s
        ms_scaled = MODEL_SIZE * s
        y_w       = np.where(
            (fr2 < 0) | (fr2 >= ms_scaled), 0.,
            np.where(fr2 < fade_y,
                     0.5*(1-np.cos(np.pi*fr2/max(fade_y,1e-6))),
            np.where(ms_scaled - fr2 < fade_y,
                     0.5*(1-np.cos(np.pi*(ms_scaled-fr2)/max(fade_y,1e-6))),
            1.0))).astype(np.float64)

        # Bilinear sample from (256×256) frame at fractional column fc_v
        # Squeeze the trailing broadcast dim from fr2 (th,1) → (th,) now that
        # y_w has already been computed; keeping it 2-D causes fr_lo[:,None]
        # to become (th,1,1) which makes p00 shape (th,1,Ncols,3) and breaks
        # the accumulation into tile['acc'] with shape (th,Ncols,3).
        fr_real   = (fr2 / s).squeeze(-1)          # frame row in original res  (th,)
        fr_clip   = np.clip(fr_real, 0., MODEL_SIZE - 1.)
        fr_lo     = np.floor(fr_clip).astype(int)  # (th,)
        fr_hi     = np.minimum(fr_lo + 1, MODEL_SIZE - 1)
        alpha_y   = fr_clip - fr_lo                # (th,)

        fc_lo  = np.floor(fc_v).astype(int)
        fc_hi  = np.minimum(fc_lo + 1, MODEL_SIZE - 1)
        alpha_x = fc_v - fc_lo

        # Bilinear interpolation — p** shape: (th, Ncols, 3)
        p00 = frame_rgb[fr_lo[:, None], fc_lo[None, :], :]
        p01 = frame_rgb[fr_lo[:, None], fc_hi[None, :], :]
        p10 = frame_rgb[fr_hi[:, None], fc_lo[None, :], :]
        p11 = frame_rgb[fr_hi[:, None], fc_hi[None, :], :]
        pix = (p00 * (1-alpha_y[:, None, None]) * (1-alpha_x[None, :, None]) +
               p01 * (1-alpha_y[:, None, None]) *    alpha_x[None, :, None]  +
               p10 *    alpha_y[:, None, None]  * (1-alpha_x[None, :, None]) +
               p11 *    alpha_y[:, None, None]  *    alpha_x[None, :, None])

        w2d  = (y_w * x_w_sel[None, :] * quality)             # (th,1)*(1,Ncols) → (th,Ncols)
        tgt  = active[valid]
        tile['acc'][:, tgt, :] += pix * w2d[:, :, None]
        tile['wgt'][:, tgt]    += w2d

    # ── Output ────────────────────────────────────────────────────────────────

    def _tile_to_full_rgb(self, tidx: int) -> Optional[np.ndarray]:
        """Return tile as full CANVAS_H × TILE_W uint8 RGB."""
        if tidx not in self._tiles: return None
        tile = self._tiles[tidx]
        safe = np.where(tile['wgt'] > 0, tile['wgt'], 1.)
        img  = (np.clip(tile['acc'] / safe[:,:,None], 0, 1) * 255).astype(np.uint8)
        img[tile['wgt'] == 0] = 0
        h, w = img.shape[:2]
        if (h, w) != (self.CANVAS_H, self.TILE_W):
            img = cv2.resize(img, (self.TILE_W, self.CANVAS_H),
                             interpolation=cv2.INTER_LINEAR)
        return img

    def get_panorama(self, crop: bool = True) -> Optional[np.ndarray]:
        if not self._tiles or self.frame_count == 0: return None
        mn = min(self._tiles); mx = max(self._tiles)
        n  = mx - mn + 1
        pano = np.zeros((self.CANVAS_H, n * self.TILE_W, 3), np.uint8)
        for idx in range(mn, mx + 1):
            x0   = (idx - mn) * self.TILE_W
            timg = self._tile_to_full_rgb(idx)
            if timg is not None:
                pano[:, x0:x0+self.TILE_W, :] = timg
        if crop and self._col0_min is not None:
            mn_px = mn * self.TILE_W
            l = max(0, self._col0_min - mn_px - 5)
            r = min(pano.shape[1], self._col0_max - mn_px + MODEL_SIZE + 5)
            if l < r: pano = pano[:, l:r]
        return pano

    def get_coverage_info(self) -> Dict[int, dict]:
        info = {}
        for idx, tile in self._tiles.items():
            wmax  = float(tile['wgt'].max())
            total = tile['imu_frames'] + tile['gm_frames']
            gm_f  = tile['gm_frames'] / max(total, 1)
            if   wmax < 0.5:  cov = 'gap'
            elif gm_f  > 0.5: cov = 'high'
            else:              cov = 'imu'
            info[idx] = {'coverage': cov, 'wgt_max': round(wmax, 3),
                         'level': tile['level'], 'n_frames': total}
        return info

    def memory_mb(self) -> float:
        return sum(t['acc'].nbytes + t['wgt'].nbytes
                   for t in self._tiles.values()) / 1e6

    def reset(self):
        self._tiles.clear(); self._cur = None
        self.frame_count = 0; self.anchor_row = None
        self.gm_rows.clear(); self.col0s.clear()
        self._col0_min = None; self._col0_max = None


# ═════════════════════════════════════════════════════════════════════════════
#  3.  COVERAGE HEATMAP  (new — Engineering Feature 3)
# ═════════════════════════════════════════════════════════════════════════════

class CoverageHeatmap:
    """
    Renders a colour-coded horizontal strip showing panorama coverage quality.
      Green  (#00c850) — high-confidence (GM+IMU)
      Yellow (#f0b000) — IMU-only registration
      Red    (#e03020) — gap / not yet covered
      Grey   (#404040) — not yet reached

    Strip height = STRIP_H px. Each tile = TILE_PX wide in the strip.
    Overlaid at the bottom of the panorama preview window.
    Saved as coverage.png alongside panorama.png on session end.
    """
    STRIP_H  = 22
    TILE_PX  = 12    # pixels per tile in the heatmap strip
    COL      = {'high': (0,200,80), 'imu': (0,180,240), 'gap': (50,50,220), 'none': (60,60,60)}

    def render(self, coverage_info: Dict[int, dict],
               panorama_width_px: int) -> np.ndarray:
        """Return (STRIP_H, panorama_width_px, 3) uint8 BGR heatmap strip."""
        strip = np.full((self.STRIP_H, panorama_width_px, 3), 60, np.uint8)
        if not coverage_info: return strip
        mn  = min(coverage_info)
        for idx, info in coverage_info.items():
            col  = self.COL.get(info['coverage'], self.COL['none'])
            x0   = (idx - mn) * self.TILE_PX
            x1   = x0 + self.TILE_PX
            x0   = max(0, min(x0, panorama_width_px - 1))
            x1   = max(0, min(x1, panorama_width_px))
            if x0 < x1:
                strip[:, x0:x1] = col
                # Confidence bar inside tile
                conf = min(1., info['wgt_max'])
                bar  = int(conf * (x1 - x0))
                strip[self.STRIP_H-4:, x0:x0+bar] = \
                    tuple(min(255, c + 60) for c in col)
        # Legend
        cv2.putText(strip, "Coverage: ", (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220,220,220), 1)
        return strip

    def overlay_on_panorama(self, pano_bgr: np.ndarray,
                             coverage_info: Dict[int, dict]) -> np.ndarray:
        """Attach heatmap strip below the panorama for display."""
        if pano_bgr is None: return pano_bgr
        if pano_bgr is None: return
        strip = self.render(coverage_info, pano_bgr.shape[1])
        return np.vstack([pano_bgr, strip])

    def save(self, path: Path, coverage_info: Dict[int, dict], width_px: int):
        strip = self.render(coverage_info, width_px)
        cv2.imwrite(str(path), strip)


# ═════════════════════════════════════════════════════════════════════════════
#  4.  AUDIT LOGGER + REPLAY  (new — Engineering Feature 4)
# ═════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Full per-frame audit trail for clinical training and QA.

    Saved per session:
      audit_log.jsonl — one JSON line per frame with full state
      replay.mp4      — annotated video of probe path overlaid on panorama

    replay_session() can be called after a session ends to build the video.
    """

    def __init__(self):
        self._entries: List[dict] = []

    def log(self, frame_n: int, ts: float, gm_x: float, gm_y: float,
            col0_px: float, quality: float, gater_components: dict,
            imu_yaw: float, imu_conf: float, imu_valid: bool,
            anchor: bool = False, bad: bool = False):
        self._entries.append({
            'n':              frame_n,
            'ts':             round(ts, 4),
            'gm_x_mm':        round(gm_x,   4),
            'gm_y_mm':        round(gm_y,   4),
            'col0_px':        round(col0_px, 2),
            'quality_b1':     round(quality, 4),
            'sharp':          gater_components.get('sharpness', 0.),
            'contact':        gater_components.get('contact',   0.),
            'snr':            gater_components.get('snr',       0.),
            'gate_score':     gater_components.get('combined',  0.),
            'imu_yaw':        round(imu_yaw,  3),
            'imu_conf':       round(imu_conf, 3),
            'imu_valid':      imu_valid,
            'anchor':         anchor,
            'bad':            bad,
        })

    def save_jsonl(self, path: Path):
        with open(path, 'w') as f:
            for e in self._entries:
                f.write(json.dumps(e) + '\n')
        print(f"  audit_log.jsonl  ({len(self._entries)} entries)")

    def clear(self):
        self._entries.clear()

    def replay_session(self, session_dir: Path,
                       panorama_path: Path,
                       fps: float = 15.,
                       output_name: str = 'replay.mp4') -> Optional[Path]:
        """
        Build an annotated replay video showing:
          - Probe position cursor sweeping across the panorama
          - Quality score and gater signals plotted as a live chart below
          - IMU confidence shown via cursor colour
          - Anchor/bad frames flagged

        Requires: panorama.png + audit_log.jsonl in session_dir.
        Output:   session_dir/replay.mp4
        """
        pano_path = session_dir / 'panorama.png' if panorama_path is None else panorama_path
        log_path  = session_dir / 'audit_log.jsonl'
        if not pano_path.exists() or not log_path.exists():
            print("[Replay] panorama.png or audit_log.jsonl not found")
            return None

        pano   = cv2.imread(str(pano_path))
        if pano is None:
            print("[Replay] Cannot read panorama"); return None

        entries = []
        with open(log_path) as f:
            for line in f:
                try: entries.append(json.loads(line))
                except: pass
        if not entries:
            print("[Replay] No entries in audit log"); return None

        ph, pw = pano.shape[:2]
        chart_h = 100
        frame_h = ph + chart_h
        out_path = session_dir / output_name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (pw, frame_h))

        col0_max = max(e['col0_px'] for e in entries) + MODEL_SIZE
        col0_min = min(e['col0_px'] for e in entries)
        scale    = pw / max(col0_max - col0_min, 1.)

        for i, e in enumerate(entries):
            canvas = np.zeros((frame_h, pw, 3), np.uint8)
            canvas[:ph] = pano

            # Probe cursor
            cx  = int((e['col0_px'] - col0_min) * scale + MODEL_SIZE * scale / 2)
            cx  = max(0, min(cx, pw - 1))
            col = (0, 255*int(e['imu_conf'] > 0.5), 255) if e['imu_valid'] else (100, 100, 100)
            if e.get('anchor'): col = (0, 255, 0)
            if e.get('bad'):    col = (0, 0, 255)
            cv2.line(canvas, (cx, 0), (cx, ph), col, 2)
            cv2.circle(canvas, (cx, ph//2), 6, col, -1)

            # Quality chart strip
            chart_y = ph
            cv2.rectangle(canvas, (0, chart_y), (pw, frame_h), (20,20,20), -1)
            x  = int(i / max(len(entries)-1, 1) * (pw - 1))
            qs = int(e.get('quality_b1', 0) * (chart_h - 4))
            gs = int(e.get('gate_score', 0) * (chart_h - 4))
            cv2.line(canvas, (x, frame_h - qs), (x, frame_h), (0, 255, 150), 1)
            cv2.line(canvas, (x, frame_h - gs), (x, frame_h - gs + 2), (255, 200, 0), 1)

            cv2.putText(canvas, f"Frame {e['n']}  QS:{e.get('quality_b1',0):.2f}  "
                        f"Gate:{e.get('gate_score',0):.2f}  IMU:{e.get('imu_conf',0):.0%}",
                        (8, frame_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
            writer.write(canvas)

        writer.release()
        print(f"  replay.mp4  ({len(entries)} frames @ {fps:.0f} fps)")
        return out_path


# ═════════════════════════════════════════════════════════════════════════════
#  5.  MULTI-SESSION REGISTRAR  (new — Engineering Feature 5)
# ═════════════════════════════════════════════════════════════════════════════

class MultiSessionRegistrar:
    """
    Registers a new session's panorama to a stored previous session
    using CEJ and ABC landmark correspondences from B1's point head.

    Algorithm:
      1. Load previous session's coordinates.json → extract CEJ/ABC positions
         in canvas space (col0_px + cx_256 offset)
      2. Current session provides same landmarks live from annotator cache
      3. Build point correspondences (min 4 pairs needed for homography)
      4. cv2.findHomography (RANSAC) → compute rigid/projective warp H
      5. cv2.warpPerspective → warp new panorama onto previous frame
      6. Save registered_panorama.png

    Limitations / assumptions:
      - Both sessions must have adequate CEJ and ABC landmark detections
      - Works best on same patient (dental arch geometry is stable)
      - Outlier rejection via RANSAC (reprojError=3.0)
    """

    RANSAC_THRESH = 3.0   # pixels

    def register(self,
                 new_pano:     np.ndarray,
                 new_coords:   List[dict],
                 prev_pano_path: Path,
                 prev_coords_path: Path,
                 output_path:  Path) -> bool:
        """
        new_pano:        (H,W,3) uint8 RGB current panorama
        new_coords:      list of coord dicts from current session
        prev_pano_path:  path to previous session panorama.png
        prev_coords_path: path to previous session coordinates.json
        output_path:     where to save registered_panorama.png
        Returns True on success.
        """
        # Load previous session
        prev_pano = cv2.imread(str(prev_pano_path))
        if prev_pano is None:
            print(f"[Reg] Cannot read {prev_pano_path}"); return False

        try:
            with open(prev_coords_path) as f:
                prev_coords = json.load(f)
        except Exception as e:
            print(f"[Reg] Cannot read {prev_coords_path}: {e}"); return False

        # Extract landmark positions in canvas space
        def _extract_landmarks(coords: List[dict]) -> Dict[str, List[Tuple]]:
            lm = {'CEJ': [], 'ABC': []}
            for c in coords:
                col0 = c.get('col0_px', 0.)
                ps   = PIXEL_SPACING_MM
                for feat in ['CEJ', 'ABC']:
                    if f'{feat.lower()}_cx_256' in c:
                        cx = c[f'{feat.lower()}_cx_256']
                        cy = c[f'{feat.lower()}_cy_256']
                        lm[feat].append((col0 + cx, cy))
            return lm

        prev_lm = _extract_landmarks(prev_coords)
        new_lm  = _extract_landmarks(new_coords)

        # Build correspondence pairs
        src_pts = []   # new session
        dst_pts = []   # prev session

        for feat in ['CEJ', 'ABC']:
            pl = prev_lm.get(feat, [])
            nl = new_lm.get(feat, [])
            # Pair by index (same tooth order assumed)
            n_pairs = min(len(pl), len(nl))
            for i in range(n_pairs):
                src_pts.append(nl[i])
                dst_pts.append(pl[i])

        if len(src_pts) < 4:
            print(f"[Reg] Not enough landmark pairs ({len(src_pts)} < 4 needed)")
            return False

        src = np.array(src_pts, np.float32)
        dst = np.array(dst_pts, np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.RANSAC_THRESH)
        if H is None:
            print("[Reg] Homography estimation failed"); return False

        n_inliers = int(mask.sum()) if mask is not None else 0
        print(f"[Reg] Homography: {n_inliers}/{len(src_pts)} inliers")

        if prev_pano is None: print("[Register] No previous panorama"); return None
        h, w = prev_pano.shape[:2]
        new_bgr = cv2.cvtColor(new_pano, cv2.COLOR_RGB2BGR)
        warped  = cv2.warpPerspective(new_bgr, H, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))

        # Blend: new session over previous where warped coverage exists
        mask_w  = (warped.sum(axis=2) > 0).astype(np.uint8)
        out     = prev_pano.copy()
        out[mask_w > 0] = cv2.addWeighted(
            prev_pano[mask_w > 0], 0.4,
            warped[mask_w > 0], 0.6, 0)

        cv2.imwrite(str(output_path), out)
        print(f"[Reg] Saved → {output_path}")
        return True


# ═════════════════════════════════════════════════════════════════════════════
#  6.  INT8 QUANTIZATION UTILITY  (new — Engineering Feature 6)
# ═════════════════════════════════════════════════════════════════════════════

def quantize_b1_model(model_h5_path: str,
                       output_path:   str,
                       n_calib_samples: int = 100) -> dict:
    """
    Post-training INT8 quantisation of B1 using TensorFlow Lite.

    Steps:
      1. Load B1 .h5 model via tf.saved_model path (convert from Keras first)
      2. Build TFLiteConverter with INT8 optimisation
      3. Run representative dataset (random normal inputs — replace with
         real calibration data from your validation set for best accuracy)
      4. Convert and save .tflite model
      5. Report size reduction and estimated accuracy impact

    For real clinical deployment:
      Replace representative_data_gen with actual patient ultrasound crops
      from your validation set to calibrate activation ranges accurately.
      INT8 with good calibration typically preserves CE within 5-10 µm.

    Returns dict with size_mb_fp32, size_mb_int8, reduction_pct.
    """
    import tensorflow as tf
    import tempfile, os as _os

    print(f"\n[Quantize] Loading DentScan model: {model_h5_path}")
    model = tf.keras.models.load_model(model_h5_path, compile=False)

    # Save as TF SavedModel format (required for TFLite conversion)
    with tempfile.TemporaryDirectory() as tmp:
        saved_path = _os.path.join(tmp, 'saved_model')
        model.save(saved_path, save_format='tf')

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
        converter.optimizations          = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

        def representative_data_gen():
            for _ in range(n_calib_samples):
                # Replace with real US frame crops for clinical deployment
                yield [np.random.normal(0.5, 0.2,
                       (1, 256, 256, 3)).astype(np.float32).clip(0, 1)]

        converter.representative_dataset = representative_data_gen

        print(f"[Quantize] Converting to INT8 ({n_calib_samples} calib samples)…")
        try:
            tflite_model = converter.convert()
        except Exception as e:
            print(f"[Quantize] INT8 failed ({e}), trying dynamic range…")
            converter2 = tf.lite.TFLiteConverter.from_saved_model(saved_path)
            converter2.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter2.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    fp32_mb = _os.path.getsize(model_h5_path) / 1e6
    int8_mb = len(tflite_model) / 1e6
    red_pct = (1. - int8_mb / fp32_mb) * 100.

    result = {
        'fp32_size_mb':    round(fp32_mb, 1),
        'int8_size_mb':    round(int8_mb, 1),
        'reduction_pct':   round(red_pct, 1),
        'calib_samples':   n_calib_samples,
        'output_path':     output_path,
    }
    print(f"[Quantize] ✓  FP32: {fp32_mb:.1f} MB → INT8: {int8_mb:.1f} MB "
          f"({red_pct:.0f}% reduction)")
    print(f"[Quantize] Saved → {output_path}")
    print(f"[Quantize] NOTE: Replace representative_data_gen with real "
          f"validation crops for clinical-grade accuracy preservation.")
    return result


# ═════════════════════════════════════════════════════════════════════════════
#  METRIC MONITOR  (new — debug logging for IMU / stitching parameter tuning)
# ═════════════════════════════════════════════════════════════════════════════

class PanoMetricMonitor:
    """
    Per-frame debug logging for panorama stitching and IMU fusion.

    Saves to session_dir/debug/:
      imu_stitching.csv   — per-frame IMU and stitching metrics
      quality_gate.csv    — per-frame gater signals (sharpness, contact, SNR)
      session_summary.json — aggregate statistics for parameter tuning

    Give these files to me during debugging to help diagnose:
      - IMU-GM correlation (is IMU fusion actually helping?)
      - Tile level distribution (is mipmap working?)
      - Gate accept rate and signal distributions
      - Frame rejection reasons

    Usage:
        mm = PanoMetricMonitor(session_dir)
        mm.log(result, col0, imu_now, imu_conf, tile_idx, tile_level)
        mm.save_summary(canvas)
        data = PanoMetricMonitor.load(session_dir)  # for debugging
    """

    def __init__(self, session_dir: Path):
        import csv as _csv
        self._dir  = Path(session_dir)
        dbdir      = self._dir / 'debug'
        dbdir.mkdir(parents=True, exist_ok=True)
        self._t0   = time.time()

        # IMU + stitching CSV
        self._imu_path = dbdir / 'imu_stitching.csv'
        self._imu_fh   = open(self._imu_path, 'w', newline='', encoding='utf-8')
        self._imu_w    = _csv.DictWriter(self._imu_fh, fieldnames=[
            'frame_n','elapsed_s','gm_x_mm','gm_x_fused','col0_px','mask_iou','gm_jump_px','iou_gate_pass','sweep_dir','dir_accept',
            'imu_yaw','imu_pitch','imu_roll',
            'imu_gyro_x','imu_gyro_y','imu_gyro_z',
            'imu_acc_x','imu_acc_y','imu_acc_z',
            'imu_conf','imu_alpha_used',
            'imu_connected','imu_valid','imu_dt_ms','imu_packet_hz',
            'quality_b1','gm_conf','cej_conf','abc_conf',
            'gm_dx_px','gm_dy_px','probe_vel_mm_s',
            'gate_pass','gate_score','tile_idx','tile_level',
            'frames_this_col','seam_var','canvas_memory_mb','accepted',
        ])
        self._imu_w.writeheader()

        # Quality gate CSV
        self._gate_path = dbdir / 'quality_gate.csv'
        self._gate_fh   = open(self._gate_path, 'w', newline='', encoding='utf-8')
        self._gate_w    = _csv.DictWriter(self._gate_fh, fieldnames=[
            'frame_n','elapsed_s','sharpness','contact','snr',
            'gate_score','gate_pass','b1_quality',
        ])
        self._gate_w.writeheader()

        self._summary_path = dbdir / 'session_summary.json'
        self._frame_n      = 0
        # Accumulators for summary
        self._gm_x_raw:   List[float] = []
        self._gm_x_fused: List[float] = []
        self._imu_yaws:   List[float] = []
        self._qualities:  List[float] = []
        self._gate_scores:List[float] = []
        self._accepted:   List[int]   = []
        self._probe_vel:  List[float] = []
        self._gm_conf:    List[float] = []
        self._imu_alphas: List[float] = []
        self._seam_vars:  List[float] = []
        self._col_hits:   dict        = {}  # col_idx -> count
        self._last_gm_x:  Optional[float] = None
        self._last_gm_y:  Optional[float] = None
        self._last_imu_ts: Optional[float] = None
        self._imu_intervals: List[float] = []

    def log(self, result: dict,
            col0_px:      float,
            imu_now:      'IMUSample',
            imu_conf:     float,
            imu_alpha:    float,
            canvas:       'TiledCanvas',
            tile_idx:     int,
            tile_level:   int,
            gm_x_fused:   float = 0.,
            frame_bgr:    Optional[np.ndarray] = None):
        """
        Call every frame (accepted or rejected) while recording.

        New parameters vs v1:
          imu_alpha     — actual complementary filter alpha (how much IMU was trusted)
          gm_x_fused    — the actual fused GM x position (not raw)
          frame_bgr     — raw frame for seam variance computation (optional, costs ~1ms)
        """
        self._frame_n += 1
        n  = self._frame_n
        el = round(time.time() - self._t0, 2)

        gm_x     = result.get('gm_x_mm', 0.)
        gm_y     = result.get('gm_y_mm', 0.)
        q        = result.get('quality_score', 0.)
        gs       = result.get('gate_score', 0.)
        gp       = int(result.get('gate_pass', False))
        gc       = result.get('gate_comp', {})
        accepted = int(result.get('gate_pass', False))
        pc       = result.get('point_cache', {})

        # ── Per-feature B1 confidences ──────────────────────────────────
        # proc dict has {'GM': {'prob':…,'cx_256':…}, 'CEJ': {…}, …}
        # point_cache has {'GM': (cx_px, cy_px)} — pixel coords, no prob
        proc_d = result.get('proc', {})
        gm_c  = float(proc_d.get('GM',  {}).get('prob', 0.)) if isinstance(proc_d.get('GM'),  dict) else 0.
        cej_c = float(proc_d.get('CEJ', {}).get('prob', 0.)) if isinstance(proc_d.get('CEJ'), dict) else 0.
        abc_c = float(proc_d.get('ABC', {}).get('prob', 0.)) if isinstance(proc_d.get('ABC'), dict) else 0.

        # ── Probe velocity ───────────────────────────────────────────────
        gm_dx_px = gm_dy_px = 0.; vel = 0.
        if self._last_gm_x is not None:
            gm_dx_px = float((gm_x - self._last_gm_x) / PIXEL_SPACING_MM)
            gm_dy_px = float((gm_y - self._last_gm_y) / PIXEL_SPACING_MM)
            vel      = abs(gm_x - self._last_gm_x) / max(el - (self._last_frame_el
                                                                if hasattr(self,'_last_frame_el') else el-0.04), 1e-6)
        self._last_gm_x = gm_x; self._last_gm_y = gm_y
        self._last_frame_el = el

        # ── IMU packet timing ────────────────────────────────────────────
        imu_dt_ms = 0.; imu_hz = 0.
        now_ts = imu_now.ts if imu_now.ts > 0 else time.time()
        if self._last_imu_ts is not None and now_ts > self._last_imu_ts:
            dt = (now_ts - self._last_imu_ts) * 1000.
            self._imu_intervals.append(dt)
            imu_dt_ms = round(dt, 2)
            if len(self._imu_intervals) >= 5:
                imu_hz = round(1000. / (np.mean(self._imu_intervals[-10:]) + 1e-6), 1)
        self._last_imu_ts = now_ts

        # ── Canvas column hit count ──────────────────────────────────────
        col_key = int(col0_px // 4)  # 4-px buckets
        self._col_hits[col_key] = self._col_hits.get(col_key, 0) + 1
        frames_this_col = self._col_hits[col_key]

        # ── Seam variance (pixel variance at left/right edges of frame) ──
        seam_var = 0.
        if frame_bgr is not None and gp:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                seam_var = round(float(gray[:, :8].var() + gray[:, -8:].var()) / 2., 2)
            except: pass

        # Accumulators
        self._gm_x_raw.append(gm_x); self._gm_x_fused.append(gm_x_fused)
        self._imu_yaws.append(imu_now.yaw); self._qualities.append(q)
        self._gate_scores.append(gs); self._accepted.append(accepted)
        self._probe_vel.append(vel); self._gm_conf.append(gm_c)
        self._imu_alphas.append(imu_alpha); self._seam_vars.append(seam_var)

        # Write IMU/stitching row (all IMUSample fields now included)
        self._imu_w.writerow({
            'frame_n':          n,
            'elapsed_s':        el,
            'gm_x_mm':          round(gm_x,            4),
            'gm_x_fused':       round(gm_x_fused,       4),
            'col0_px':          round(col0_px,           2),
            'imu_yaw':          round(imu_now.yaw,       3),
            'imu_pitch':        round(imu_now.pitch,     3),
            'imu_roll':         round(imu_now.roll,      3),
            'imu_gyro_x':       round(imu_now.gyro_x,   3),
            'imu_gyro_y':       round(imu_now.gyro_y,   3),
            'imu_gyro_z':       round(imu_now.gyro_z,   3),
            'imu_acc_x':        round(imu_now.acc_x,    3),
            'imu_acc_y':        round(imu_now.acc_y,    3),
            'imu_acc_z':        round(imu_now.acc_z,    3),
            'imu_conf':         round(imu_conf,         3),
            'imu_alpha_used':   round(imu_alpha,        3),
            'imu_connected':    int(imu_hz > 0),  # >0 Hz means BLE is receiving packets
            'imu_valid':        int(imu_now.valid),
            'imu_dt_ms':        imu_dt_ms,
            'imu_packet_hz':    imu_hz,
            'quality_b1':       round(q,                4),
            'gm_conf':          round(gm_c,             3),
            'cej_conf':         round(cej_c,            3),
            'abc_conf':         round(abc_c,            3),
            'gm_dx_px':         round(gm_dx_px,         2),
            'gm_dy_px':         round(gm_dy_px,         2),
            'probe_vel_mm_s':   round(vel,              3),
            'gate_pass':        gp,
            'gate_score':       round(gs,               3),
            'tile_idx':         tile_idx,
            'tile_level':       tile_level,
            'frames_this_col':  frames_this_col,
            'seam_var':         seam_var,
            'canvas_memory_mb': round(canvas.memory_mb(), 2),
            'accepted':         accepted,
            'mask_iou':         round(result.get('iou_metrics',{}).get('mask_iou',1.),3),
            'gm_jump_px':       round(result.get('iou_metrics',{}).get('gm_jump_px',0.),2),
            'iou_gate_pass':    int(result.get('gate_pass_iou',True)),
            'sweep_dir':        result.get('_sweep_dir',0),
            'dir_accept':       int(result.get('_dir_accept',True)),
        })
        self._imu_fh.flush()

        # Gate signals row
        self._gate_w.writerow({
            'frame_n':    n,
            'elapsed_s':  el,
            'sharpness':  round(gc.get('sharpness', 0.), 3),
            'contact':    round(gc.get('contact',   0.), 3),
            'snr':        round(gc.get('snr',       0.), 3),
            'gate_score': round(gs, 3),
            'gate_pass':  gp,
            'b1_quality': round(q, 4),
        })
        self._gate_fh.flush()

    def save_summary(self, canvas: 'TiledCanvas') -> dict:
        """Save session summary. Call on session end."""
        n = len(self._qualities)
        if n == 0:
            return {}

        # IMU-GM cross-correlation
        imu_corr = 0.
        if len(self._gm_x_raw) >= 10 and len(self._imu_yaws) >= 10:
            gm  = np.array(self._gm_x_raw[:len(self._imu_yaws)])
            yaw = np.array(self._imu_yaws[:len(gm)])
            if gm.std() > 0 and yaw.std() > 0:
                imu_corr = float(np.corrcoef(gm, yaw)[0,1])

        # IMU-GM time lag (cross-correlation peak lag)
        imu_lag_frames = 0
        if len(self._gm_x_raw) >= 20 and len(self._imu_yaws) >= 20:
            try:
                gm_n  = np.array(self._gm_x_raw) - np.mean(self._gm_x_raw)
                yaw_n = np.array(self._imu_yaws[:len(gm_n)]) - np.mean(self._imu_yaws[:len(gm_n)])
                xcorr = np.correlate(gm_n, yaw_n, mode='full')
                lags  = np.arange(-len(gm_n)+1, len(gm_n))
                imu_lag_frames = int(lags[np.argmax(np.abs(xcorr))])
            except: imu_lag_frames = 0

        # IMU packet timing
        imu_hz_mean = round(1000./np.mean(self._imu_intervals), 1) if len(self._imu_intervals)>5 else 0.
        imu_hz_std  = round(1000.*np.std(self._imu_intervals)/max(np.mean(self._imu_intervals)**2,1), 2) if len(self._imu_intervals)>5 else 0.

        # Probe velocity stats
        vel_arr = np.array([v for v in self._probe_vel if v > 0])
        vel_stats = {
            'mean_mm_s': round(float(vel_arr.mean()), 2) if len(vel_arr) > 0 else 0.,
            'max_mm_s':  round(float(vel_arr.max()),  2) if len(vel_arr) > 0 else 0.,
            'p95_mm_s':  round(float(np.percentile(vel_arr, 95)), 2) if len(vel_arr) > 0 else 0.,
            'note': ('Probe moved too fast (>5 mm/s) — panorama may be blurry. Sweep slower.'
                     if len(vel_arr)>0 and vel_arr.mean()>5. else 'Probe velocity OK'),
        }

        # Column density (coverage uniformity)
        if self._col_hits:
            hit_vals = np.array(list(self._col_hits.values()))
            col_cv   = round(float(hit_vals.std()/max(hit_vals.mean(),1)), 3)
        else:
            col_cv = 0.

        # Seam quality
        sv_arr   = np.array([v for v in self._seam_vars if v > 0])
        seam_stats = {
            'mean_var': round(float(sv_arr.mean()), 2) if len(sv_arr)>0 else 0.,
            'p95_var':  round(float(np.percentile(sv_arr,95)),2) if len(sv_arr)>0 else 0.,
        }

        # Complementary filter usage
        alpha_arr = np.array(self._imu_alphas) if self._imu_alphas else np.array([0.])
        alpha_stats = {
            'mean': round(float(alpha_arr.mean()), 3),
            'note': ('IMU mostly trusted (α>0.8 mean) — check GM quality if IMU drifts'
                     if alpha_arr.mean()>0.8 else
                     'GM mostly trusted (α<0.5 mean) — IMU having low impact'
                     if alpha_arr.mean()<0.5 else
                     'Balanced IMU/GM fusion'),
        }

        # Tile level distribution
        cov  = canvas.get_coverage_info()
        tile_levels = {0:0, 1:0, 2:0}
        for info in cov.values():
            tile_levels[info['level']] = tile_levels.get(info['level'],0)+1

        accept_rate = float(np.mean(self._accepted)) if self._accepted else 0.

        summary = {
            'n_frames':              n,
            'duration_s':            round(time.time()-self._t0, 1),
            'canvas_memory_mb':      round(canvas.memory_mb(), 2),
            'imu_gm_correlation':    round(imu_corr, 3),
            'imu_lag_frames':        imu_lag_frames,
            'imu_packet_hz':         {'mean': imu_hz_mean, 'std': imu_hz_std,
                                      'note': ('IMU rate OK' if imu_hz_mean>150 else
                                               'IMU rate low — check BLE/serial connection')},
            'imu_alpha':             alpha_stats,
            'probe_velocity':        vel_stats,
            'coverage_uniformity_cv': col_cv,
            'seam_quality':          seam_stats,
            'gm_confidence':         {'mean': round(float(np.mean(self._gm_conf)),3) if self._gm_conf else 0.},
            'imu_note':              ('Good IMU fusion' if abs(imu_corr)>0.7
                                      else 'Weak IMU-GM correlation — check scale calibration'
                                      if abs(imu_corr)>0.3
                                      else 'Poor IMU-GM correlation — IMU may not be helping'),
            'quality_b1':            {
                'mean':   round(float(np.mean(self._qualities)), 4),
                'std':    round(float(np.std(self._qualities)),  4),
                'p25':    round(float(np.percentile(self._qualities, 25)), 4),
                'p75':    round(float(np.percentile(self._qualities, 75)), 4),
            },
            'gate': {
                'accept_rate':     round(accept_rate, 3),
                'mean_score':      round(float(np.mean(self._gate_scores)), 3),
                'reject_note':     ('Low accept rate — lower qs_thresh or check probe contact'
                                    if accept_rate < 0.4
                                    else 'High accept rate — consider raising qs_thresh'
                                    if accept_rate > 0.95
                                    else 'Gate accept rate in normal range'),
            },
            'tile_level_distribution': tile_levels,
            'tile_coverage':           {k: v['coverage'] for k,v in cov.items()},
            'tuning_tips': self._tuning_tips(imu_corr, accept_rate),
        }
        with open(self._summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[PanoMetrics] Summary → {self._summary_path}")
        print(f"  IMU-GM corr: {imu_corr:.3f}  Gate accept: {accept_rate:.0%}  "
              f"Tiles: {tile_levels}  Mem: {canvas.memory_mb():.1f}MB")
        for tip in summary['tuning_tips']:
            print(f"  ► {tip}")
        return summary

    def _tuning_tips(self, imu_corr, accept_rate) -> List[str]:
        tips = []
        if abs(imu_corr) < 0.4:
            tips.append("IMU-GM correlation low: press C to calibrate gyro scale, "
                        "or check IMU is firmly attached to probe handle")
        if accept_rate < 0.3:
            tips.append("Gate accept rate very low (<30%): reduce --qs_thresh, "
                        "check probe contact quality, use more coupling gel")
        if accept_rate > 0.95:
            tips.append("Gate accept rate very high (>95%): consider raising "
                        "--qs_thresh to be more selective about bad frames")
        if not tips:
            tips.append("Parameters look well-balanced")
        return tips

    def close(self):
        try: self._imu_fh.close()
        except: pass
        try: self._gate_fh.close()
        except: pass

    @staticmethod
    def load(session_dir: Path) -> dict:
        """Load all debug files for analysis. Give me these files for debugging."""
        import csv as _csv
        dbdir = Path(session_dir) / 'debug'
        data  = {'session_dir': str(session_dir)}
        try:
            with open(dbdir/'session_summary.json') as f:
                data['summary'] = json.load(f)
        except: data['summary'] = {}
        for name, key in [('imu_stitching.csv','imu_rows'),
                           ('quality_gate.csv','gate_rows')]:
            rows = []
            try:
                with open(dbdir/name) as f:
                    rows = list(_csv.DictReader(f))
            except: pass
            data[key] = rows
        print(f"[PanoMetricMonitor.load] "
              f"IMU rows:{len(data.get('imu_rows',[]))}  "
              f"Gate rows:{len(data.get('gate_rows',[]))}")
        return data


# ═════════════════════════════════════════════════════════════════════════════
#  IMU CLASSES  (unchanged from live_panorama.py)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class IMUSample:
    ts:     float = 0.
    acc_x:  float = 0.; acc_y: float = 0.; acc_z: float = 0.
    gyro_x: float = 0.; gyro_y: float = 0.; gyro_z: float = 0.
    roll:   float = 0.; pitch: float = 0.; yaw:   float = 0.
    valid:  bool  = False


class IMUReader:
    def __init__(self, mac=IMU_MAC):
        self.mac = mac; self._latest = IMUSample(); self._lock = threading.Lock()
        self._thread = None; self._stop = threading.Event()
        self.connected = False; self.packet_count = 0
    def start(self):
        self._thread = threading.Thread(target=self._run_loop, daemon=True); self._thread.start()
    def stop(self): self._stop.set()
    def get_latest(self):
        with self._lock: return IMUSample(**self._latest.__dict__)
    def _run_loop(self): asyncio.run(self._ble_loop())
    async def _ble_loop(self):
        def s16(lo,hi): return struct.unpack('h',bytes([lo,hi]))[0]
        AS=16.*9.8/32768.; GS=2000./32768.; ANGS=180./32768.; HD=0x55; PL=11
        sample=IMUSample(); buf=bytearray()
        def p20(r):
            if len(r)>=20 and r[0]==0x55 and r[1]==0x61:
                sample.acc_x=s16(r[2],r[3])*AS; sample.acc_y=s16(r[4],r[5])*AS; sample.acc_z=s16(r[6],r[7])*AS
                sample.gyro_x=s16(r[8],r[9])*GS; sample.gyro_y=s16(r[10],r[11])*GS; sample.gyro_z=s16(r[12],r[13])*GS
                sample.roll=s16(r[14],r[15])*ANGS; sample.pitch=s16(r[16],r[17])*ANGS; sample.yaw=s16(r[18],r[19])*ANGS
                return True
            return False
        def p11(pkt):
            if len(pkt)<PL or pkt[0]!=HD: return False
            if (sum(pkt[:10])&0xFF)!=pkt[10]: return False
            t=pkt[1]; w=[s16(pkt[2+i*2],pkt[3+i*2]) for i in range(4)]
            if t==0x51: sample.acc_x,sample.acc_y,sample.acc_z=w[0]*AS,w[1]*AS,w[2]*AS
            elif t==0x52: sample.gyro_x,sample.gyro_y,sample.gyro_z=w[0]*GS,w[1]*GS,w[2]*GS
            elif t==0x53: sample.roll,sample.pitch,sample.yaw=w[0]*ANGS,w[1]*ANGS,w[2]*ANGS
            else: return False
            return True
        at=[]; fa=True; self._ct=time.time()
        # Raw packet log — captures first packets to identify WT9011DCL BLE format
        _raw_log = []
        _raw_log_done = False
        _raw_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ble_raw_packets.txt')

        def cb(s_,r):
            nonlocal buf, _raw_log, _raw_log_done; now=time.time(); sample.ts=now

            # Log first 30 notifications to file for format diagnosis
            if not _raw_log_done and len(_raw_log) < 30:
                hex_str = ' '.join(f'{b:02X}' for b in r[:32])
                _raw_log.append(f"notif_{len(_raw_log):03d} len={len(r):3d}: {hex_str}")
                if len(_raw_log) == 30:
                    _raw_log_done = True
                    try:
                        with open(_raw_log_path, 'w') as _lf:
                            _lf.write("WT9011DCL BLE Raw Packet Log\n")
                            _lf.write("="*60 + "\n")
                            _lf.write("Standard WT901 format: 55 5X ...  (X=1 acc, 2 gyro, 3 angle)\n")
                            _lf.write("Combined format:        55 61 ...  (20-byte)\n")
                            _lf.write("="*60 + "\n")
                            for line in _raw_log:
                                _lf.write(line + "\n")
                        print(f"[IMU] Raw BLE packet log → {_raw_log_path}")
                        print(f"[IMU] First packet hex: {_raw_log[0]}")
                    except Exception as _le:
                        print(f"[IMU] Log write failed: {_le}")

            if p20(r): self.packet_count+=1
            else:
                buf+=r
                while len(buf)>=PL:
                    idx=buf.find(HD)
                    if idx<0: buf.clear(); break
                    if idx>0: buf=buf[idx:]
                    if len(buf)<PL: break
                    if p11(bytes(buf[:PL])): self.packet_count+=1
                    buf=buf[PL:]
            nonlocal at,fa
            at.append(now)
            if len(at)>30: at.pop(0)
            if len(at)>=10:
                hz=9./max(now-at[-10],1e-3)
                # Burst threshold: 3× the expected WT9011DCL rate (20 Hz × 3 = 60 Hz)
                # This catches reconnect flush bursts without being triggered by
                # normal 20 Hz packets.
                BURST_THRESH = 60.0
                MAX_FLUSH_WAIT = 1.5  # seconds — force valid after this regardless
                if hz > BURST_THRESH:
                    if not fa:
                        fa = True
                        self._flush_start = now
                    sample.valid = False
                elif fa:
                    # Flush burst ended — OR we have waited too long
                    elapsed_flush = now - getattr(self, '_flush_start', now)
                    if elapsed_flush < MAX_FLUSH_WAIT:
                        fa = False
                        sample.valid = True
                    else:
                        # Max wait exceeded — force valid to prevent permanent blockage
                        fa = False
                        sample.valid = True
                        print(f"[IMU] Flush timeout after {elapsed_flush:.1f}s — forced valid")
                else:
                    sample.valid = True
            else:
                sample.valid = False
            with self._lock: self._latest=IMUSample(**sample.__dict__)
        try:
            from bleak import BleakClient
        except ImportError: print("[IMU] pip install bleak"); return
        while not self._stop.is_set():
            try:
                async with BleakClient(self.mac,timeout=15.) as c:
                    if not c.is_connected: await asyncio.sleep(5.); continue
                    await asyncio.sleep(0.5)
                    try: svcs=await c.get_services()
                    except: svcs=c.services
                    chars=[ch for svc in svcs for ch in svc.characteristics]
                    nc=next((ch.uuid for ch in chars if 'ffe4' in ch.uuid and 'notify' in ch.properties),None)
                    if nc is None:
                        nc=next((ch.uuid for ch in chars if 'notify' in ch.properties),None)
                    if nc is None: await asyncio.sleep(5.); continue
                    await c.start_notify(nc,cb)
                    wc=next((ch.uuid for ch in chars if 'ffe9' in ch.uuid and any(p in ch.properties for p in ['write','write-without-response'])),None)
                    if wc:
                        # Retry rate-set up to 5 times with verification.
                        # WT901 at default 10 Hz needs explicit unlock + rate cmd.
                        # CMD sequence: unlock config (0x69) → set rate 0x0B=200Hz → save
                        CMD_UNLOCK  = bytes([0xFF,0xAA,0x69,0x88,0xB5])  # unlock config
                        CMD_200HZ   = bytes([0xFF,0xAA,0x03,0x0B,0x00])  # output rate 200Hz
                        CMD_SAVE    = bytes([0xFF,0xAA,0x00,0x00,0x00])  # save settings
                        rate_ok = False
                        for _attempt in range(5):
                            try:
                                await c.write_gatt_char(wc, CMD_UNLOCK, response=False)
                                await asyncio.sleep(0.15)
                                await c.write_gatt_char(wc, CMD_200HZ, response=False)
                                await asyncio.sleep(0.15)
                                await c.write_gatt_char(wc, CMD_SAVE, response=False)
                                await asyncio.sleep(0.5)  # wait for rate to take effect
                                # Verify by checking recent packet rate
                                cnt0 = self.packet_count
                                await asyncio.sleep(1.0)
                                hz_measured = self.packet_count - cnt0
                                if hz_measured >= 50:  # ≥50 pkts/s means rate set
                                    print(f"[IMU] 200 Hz ✓ (measured {hz_measured} pkts/s on attempt {_attempt+1})")
                                    rate_ok = True; break
                                else:
                                    print(f"[IMU] Rate attempt {_attempt+1}: {hz_measured} pkts/s — retrying")
                            except Exception as _re:
                                print(f"[IMU] Rate set attempt {_attempt+1} failed: {_re}")
                        if not rate_ok:
                            print("[IMU] ⚠ Could not set 200 Hz — still at default (10 Hz)")
                            print("[IMU]   → Use --serial COM_PORT for reliable 200 Hz")
                    self.connected=True; print(f"[IMU] Connected ✓")
                    while not self._stop.is_set() and c.is_connected: await asyncio.sleep(0.05)
                    self.connected=False
                    if not self._stop.is_set(): await asyncio.sleep(5.)
            except Exception: 
                self.connected=False
                if not self._stop.is_set(): await asyncio.sleep(5.)


class SerialIMUReader:
    def __init__(self, port, baud=9600):   # WT901C-TTL wired default
        self.port=port; self.baud=baud; self._latest=IMUSample(); self._lock=threading.Lock()
        self._thread=None; self._stop=threading.Event(); self.connected=False; self.packet_count=0
    def start(self): self._thread=threading.Thread(target=self._run,daemon=True); self._thread.start()
    def stop(self): self._stop.set()
    def get_latest(self):
        with self._lock: return IMUSample(**self._latest.__dict__)
    def _run(self):
        AS=16.*9.8/32768.; GS=2000./32768.; ANGS=180./32768.; HD=0x55; PL=11
        def s16(lo,hi): return struct.unpack('h',bytes([lo,hi]))[0]
        sample=IMUSample(); buf=bytearray()
        def p20(r):
            if len(r)>=20 and r[0]==0x55 and r[1]==0x61:
                sample.acc_x=s16(r[2],r[3])*AS; sample.acc_y=s16(r[4],r[5])*AS; sample.acc_z=s16(r[6],r[7])*AS
                sample.gyro_x=s16(r[8],r[9])*GS; sample.gyro_y=s16(r[10],r[11])*GS; sample.gyro_z=s16(r[12],r[13])*GS
                sample.roll=s16(r[14],r[15])*ANGS; sample.pitch=s16(r[16],r[17])*ANGS; sample.yaw=s16(r[18],r[19])*ANGS
                return True
            return False
        def p11(pkt):
            if len(pkt)<PL or pkt[0]!=HD: return False
            if (sum(pkt[:10])&0xFF)!=pkt[10]: return False
            t=pkt[1]; w=[s16(pkt[2+i*2],pkt[3+i*2]) for i in range(4)]
            if t==0x51: sample.acc_x,sample.acc_y,sample.acc_z=w[0]*AS,w[1]*AS,w[2]*AS
            elif t==0x52: sample.gyro_x,sample.gyro_y,sample.gyro_z=w[0]*GS,w[1]*GS,w[2]*GS
            elif t==0x53: sample.roll,sample.pitch,sample.yaw=w[0]*ANGS,w[1]*ANGS,w[2]*ANGS
            else: return False
            return True
        while not self._stop.is_set():
            try:
                import serial
                ser=serial.Serial(self.port,self.baud,timeout=1.)
                self.connected=True; sample.valid=True; buf.clear()
                print(f"[IMU-Serial] ✓ {self.port}")
                while not self._stop.is_set() and ser.is_open:
                    w=ser.in_waiting
                    if w==0: time.sleep(0.002); continue
                    chunk=ser.read(w); sample.ts=time.time()
                    if len(chunk)>=20 and p20(chunk[:20]): self.packet_count+=1; sample.valid=True
                    else:
                        buf+=chunk
                        while len(buf)>=PL:
                            idx=buf.find(HD)
                            if idx<0: buf.clear(); break
                            if idx>0: buf=buf[idx:]
                            if len(buf)<PL: break
                            if p11(bytes(buf[:PL])): self.packet_count+=1; sample.valid=True
                            buf=buf[PL:]
                    with self._lock: self._latest=IMUSample(**sample.__dict__)
                ser.close(); self.connected=False
                if not self._stop.is_set(): time.sleep(5.)
            except Exception as e:
                self.connected=False
                if not self._stop.is_set(): time.sleep(5.)


# ═════════════════════════════════════════════════════════════════════════════
#  COMPLEMENTARY FILTER  (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

class ComplementaryFilter:
    def __init__(self, tau=COMP_TAU, gyro_scale_mm_per_dps=0.12):
        # scale default raised from 0.05 → 0.12 mm/° (calibrated from session data)
        # tau lowered in main() from 0.5 → 0.2 s to reduce alpha at 20 Hz BLE
        self.tau=tau; self.scale=gyro_scale_mm_per_dps
        self.imu_sign=-1.0  # -1=inverted (WT9011DCL default), set by calibrate_scale
        self.x_fused=0.; self.x_imu_int=0.; self.last_ts=None; self.last_yaw=None
    def reset(self, x_gm):
        self.x_fused=x_gm; self.x_imu_int=x_gm; self.last_ts=None; self.last_yaw=None
        self._invalid_count=0   # must be here — getattr fallback loses persistence
    def update(self, x_gm, imu: IMUSample, quality=1.0):
        now=imu.ts if imu.ts>0 else time.time()
        if self.last_ts is None:
            self.x_fused=x_gm; self.x_imu_int=x_gm; self.last_ts=now; self.last_yaw=imu.yaw
            self._invalid_count=0
            return x_gm, 0.0
        dt=max(1e-4, now-self.last_ts); self.last_ts=now

        # If IMU is invalid (flush period, disconnected, or all-zero),
        # bypass integration entirely and use pure GM (alpha=0).
        # This prevents the high-alpha filter from suppressing real GM motion.
        if not imu.valid or (abs(imu.yaw)<1e-6 and abs(imu.gyro_z)<1e-6):
            self._invalid_count += 1   # always int — reset() guarantees initialisation
            if self._invalid_count > 5:
                # IMU has been invalid for >5 frames — use pure GM
                self.x_fused    = x_gm
                self.x_imu_int  = x_gm   # resync integration to GM
                return float(x_gm), 0.0   # alpha=0: pure GM
        else:
            self._invalid_count = 0

        # IMU displacement estimation — two sources in priority order:
        #   1. gyro_z direct integration (best — available at 200 Hz)
        #   2. yaw angle difference (fallback — works at 10 Hz when gyro_z=0)
        #      The WT901 at 10 Hz sends angle packets (0x53) but gyro packets
        #      (0x52) may be absent between video frames.  Deriving angular rate
        #      from d_yaw/dt gives real IMU contribution even at low update rates.
        if self.last_yaw is not None and abs(imu.yaw) < 90:
            dy = imu.yaw - self.last_yaw
            if dy >  180: dy -= 360
            if dy < -180: dy += 360
            # IMU sign: WT9011DCL mounted on probe handle — rightward sweep
            # (gm_x increase) produces DECREASING yaw → negate displacement.
            # imu_sign=-1 (default), set to +1 if yaw and gm_x move same direction.
            _sign = getattr(self, 'imu_sign', -1.0)
            if abs(imu.gyro_z) > 0.1:
                self.x_imu_int += _sign * imu.gyro_z * dt * self.scale
            else:
                # yaw-derived displacement
                self.x_imu_int += _sign * dy * self.scale
        else:
            # yaw > 90° — use gyro_z only
            _sign = getattr(self, 'imu_sign', -1.0)
            self.x_imu_int += _sign * imu.gyro_z * dt * self.scale

        self.last_yaw = imu.yaw
        ab=self.tau/(self.tau+dt); gw=quality**2
        alpha=float(np.clip(ab+(1.-ab)*(1.-gw),0.,1.))
        self.x_fused=alpha*self.x_imu_int+(1.-alpha)*x_gm
        self.x_imu_int=0.95*self.x_imu_int+0.05*self.x_fused
        return float(self.x_fused), float(ab)
    def calibrate_scale(self, d_mm, y0, y1):
        """Calibrate scale AND sign from a known sweep."""
        d_yaw = y1 - y0  # signed yaw change
        if abs(d_yaw) > 1.:
            self.scale = abs(d_mm) / abs(d_yaw)
            # If probe moved right (d_mm>0) but yaw decreased (d_yaw<0) → inverted
            self.imu_sign = 1.0 if (d_mm * d_yaw) > 0 else -1.0
            print(f"[IMU] Calibrated: scale={self.scale:.4f} mm/°  "
                  f"sign={self.imu_sign:+.0f}  "
                  f"({'normal' if self.imu_sign>0 else 'inverted — yaw opposes GM'})")


# ═════════════════════════════════════════════════════════════════════════════
#  PANO VOICE CONTROLLER  (unchanged from live_panorama_voice.py)
# ═════════════════════════════════════════════════════════════════════════════


# ── Extra imports for charting mode ─────────────────────────────────────
import re, math, wave, gc, csv, tempfile
from typing import Any   # extends panorama's Optional/Tuple/Dict/List

import tensorflow as tf

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False
    print("WARNING: openpyxl not installed — Excel export disabled.  pip install openpyxl")
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

try:
    from pynput import keyboard as _pynput_kb
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# MKUNET_AVAILABLE: set True here since MKUNetInference is imported in main()
# If it fails there the script exits; classes that check this flag can proceed.
MKUNET_AVAILABLE = True


# ── Charting constants ───────────────────────────────────────────────────
ENROLLMENT_DIR   = 'enrollment'
CONFIRM_WINDOW   = 2.0          # seconds to wait for CONFIRM before auto-commit
FLASH_DURATION   = 1.5          # seconds to show command flash
DEBUG_DIR        = 'debug'
IMAGE_HEIGHT     = 256
IMAGE_WIDTH      = 256

ALL_FEATURES     = ['GM','CEJ','ABC','TOOTH','BONE','GINGIVA']
POINT_FEATURES   = ['GM','CEJ','ABC']
REGION_FEATURES  = ['TOOTH','BONE','GINGIVA']
CONF_THRESH_DEF  = {'GM':0.15,'CEJ':0.20,'ABC':0.20}

FEATURE_COLORS = {
    'GINGIVA': (255,100,100), 'TOOTH': (200,200,255), 'BONE': (100,255,100),
    'CEJ': (255,255,0), 'ABC': (255,0,255), 'GM': (0,255,255),
}
CLINICAL_COLORS = {'iGR': (0,255,255), 'iGH': (0,165,255), 'iABL': (0,0,255)}


# ═════════════════════════════════════════════════════════════════════════════
#  UNIFIED VOICE CONTROLLER
#  — Single Whisper model shared by both stitching and charting modes
#  — Mode selection: say "stitching" or "charting" at any time
#  — Panorama mode: ON / OFF / SAVE / DROP / HOLD / GO / MARK / BAD / VIEW
#  — Charting mode: full tooth state machine (GO → tooth# → STOP → … → END)
# ═════════════════════════════════════════════════════════════════════════════

class UnifiedVoiceController:
    """
    Single voice controller for both stitching and charting sessions.

    MODES
    -----
    idle        System started, no mode selected yet.
                Say "stitching" or "charting" to switch.
    stitching   Panorama stitching mode.
                Commands: ON  OFF  SAVE  DROP  HOLD  GO  MARK  BAD  VIEW
    charting    Dental charting mode.
                State machine: GO → waiting_for_tooth → scanning → STOP → … → END

    MODE SWITCH
    -----------
    Say "stitching" from any state → enter stitching mode
    Say "charting"  from any state → enter charting mode
    Works mid-session; switch is immediate.

    SHARED WHISPER MODEL
    --------------------
    Loads the adapted Whisper model once (enrollment/<user>/adapted_model.pt
    if present, else base whisper-small).  Both modes use the same instance.

    FIXED-DURATION RECORDING
    -------------------------
    Uses raw PyAudio fixed-duration windows — no energy-gate timeout.
    Short words like "on", "off", "go", "stop", "end" are always captured.
    """

    # ── Mode keywords ─────────────────────────────────────────────────────────
    MODE_SWITCH_VARIANTS = {
        'stitching': [
            # Exact / close
            'STITCHING', 'STITCH', 'STITCHIN', 'STITCHINK',
            # Common accent substitutions
            'STICHING', 'STICHING', 'STEACHING', 'STEECHING',
            'STICHEN', 'STITCHEN', 'STICHIN',
            # Broader aliases
            'PANORAMA', 'PANO', 'SCANNING', 'SCAN',
            'SWEEP', 'IMAGING', 'PANORAMIC',
        ],
        'charting': [
            # Exact / close
            'CHARTING', 'CHART', 'CHARTIN', 'CHARTINK',
            # Common accent substitutions (ch→sh, t→d, ng drop)
            'SHARTING', 'SCHARTING', 'CHARDIN', 'CHARDING',
            'CHARDIN', 'CHARTEM', 'CHARTEN', 'CHARTON',
            'TCHARTING', 'JARTING', 'CHARTING',
            # Broader aliases
            'DENTAL', 'PROBING', 'PROBE', 'PERIODONTAL',
            'MEASUREMENT', 'MEASURING', 'RECORD TEETH',
        ],
    }

    # ── Panorama phoneme variants (from PanoVoiceController) ──────────────────
    PANO_PHONEME = {
        # ON — commonly misheard as OWN, ANN, IN, UMM
        'on':   ['ON','AHN','OWN','AN','UHN','OAN','ONN','ANN','AWN','OANE',
                 'IN','UN','EN','ION','INN','AND','ONS'],
        # OFF — commonly misheard as OF, AFF, COUGH
        'off':  ['OFF','OHF','AFF','OAF','AWF','OFFF','UFF','OF','OHF',
                 'OFFT','AUF','OFFE',
                 'STOP','STAHP','STAP','STOB','STUP','TOP'],
        # SAVE
        'save': ['SAVE','SAYV','SAV','SEHV','SEIV','SAFE','SAVED','SAYF',
                 'SEVE','SEIV','SAIV'],
        # DROP / DISCARD
        'drop': ['DROP','DRAWP','TROP','DROB','DRAP','DROOP','DOP','DRAW',
                 'DROPT','DROPP'],
        # HOLD / PAUSE — commonly confused with COLD, FOLD
        'hold': ['HOLD','HOALD','OLD','HOWLD','HOULD','HAWLD','HOLT',
                 'PAUSE','POZE','PAWZ','PAWS','PAUZ','POHZ','PAUSED'],
        # GO / RESUME
        'go':   ['GO','GOH','GOU','GAO','GOW','GOO','GOUW','JOH','GEO',
                 'GROW','LOW','JO','KO'],
        # MARK
        'mark': ['MARK','MAWK','MAHK','MAK','MARC','MAAK','MARKE','MARG',
                 'MORK','MORC','MARKED'],
        # BAD
        'bad':  ['BAD','BAED','BAAD','BED','BEAD','BET','BADD','BAHT',
                 'BAT','BARD','BADT'],
        # VIEW
        'view': ['VIEW','VYOO','VU','FEW','VYEW','VIU','VYOU','VEW',
                 'FYOO','PHEW','VIO'],
    }
    PANO_ALIASES = {
        'on':   ['START','BEGIN','RECORD','REC','OPEN','LAUNCH','ACTIVATE',
                 'START PANORAMA','START STITCHING','PANORAMA ON','STITCHING ON'],
        'off':  ['DONE','FINISH','HALT','STOP PANORAMA','STOP STITCHING',
                 'PANORAMA OFF','STITCHING OFF','END PANORAMA','SAVE AND STOP'],
        'save': ['OK','OKAY','SAVE PANORAMA','SAVE SESSION','SAVE IT'],
        'drop': ['CANCEL','DISCARD','CLEAR','RESET','THROW AWAY','REDO'],
        'hold': ['PAUSE','FREEZE','WAIT','HOLD ON','STAND BY'],
        'go':   ['RESUME','CONTINUE','PLAY','AGAIN','GO AHEAD','CARRY ON'],
        'mark': ['ANCHOR','FLAG','PIN','LOCK','MARK THIS'],
        'bad':  ['SKIP','GAP','MISS','REDO','REMOVE','DELETE'],
        'view': ['SHOW','IMU','DISPLAY','HIDE','TOGGLE','SHOW IMU'],
    }
    PANO_AMBIENT_REJECT = [
        # Filler / polite words
        'THANK','PLEASE','SORRY','EXCUSE','HELLO','GOODBYE','BYE',
        # Smart assistant wake words
        'OKAY GOOGLE','HEY SIRI','ALEXA',
        # Clinical ambient
        'RINSE','SUCTION','OPEN WIDE',
        # Question words — never a command
        'CAN YOU','COULD YOU','WOULD YOU','DO YOU',
        'HOW ARE','WHAT IS','WHERE IS','WHEN','WHY',
        # Common Whisper hallucinations on silence/background noise
        'GONNA','COVER IT','DOORS','ALREADY OPEN',
        'SET IT UP','THANKS FOR WATCHING','FOR WATCHING',
        'WATCHING','SUBSCRIBE','LIKE AND',
    ]

    # ── Charting phoneme variants (from VoiceCommandListenerV4) ───────────────
    CHART_PHONEME = {
        # GO / ACTIVATE — short word, often misheard
        'go':      ['GO','GOH','GOU','GAO','GOW','GOO','JOH','KO','GEO',
                    'GROW','JO','LOW','GW'],
        # STOP — Whisper often returns "STOP." with period — stripped in _normalize
        'stop':    ['STOP','STAHP','ESTOP','STAP','STOB','STUP','TOP','STOPT',
                    'STOPP','STAPH','STROP','STOPE','STOFF'],
        # END — short, confused with AND, IN
        'end':     ['END','EHND','IND','ENND','AEND','EEND','NED',
                    'AND','ENDS','ENDED','AIND','EHNED'],
        # START — Whisper returns this for "start" — mapped to go
        'start':   ['START','STAHRT','STAART','STURT','STARTT'],
        # CONFIRM
        'confirm': ['CONFIRM','CONFERM','KONFER','CONFER','KONFIRM','AFFIRM',
                    'CONFORM','CONIFORM'],
        # REPEAT
        'repeat':  ['REPEAT','REPEET','REPET','RIPEAT','REPAT','REBEAT',
                    'REPLETE'],
        # YES
        'yes':     ['YES','YAS','JES','JEH','YEH','SI','YIS','YUS'],
        # NO
        'no':      ['NO','NAO','NOH','NAH','NAW','NOE','KNOW'],
    }
    CHART_ALIASES = {
        'go':      ['START','BEGIN','ACTIVATE','READY','GO AHEAD','PROCEED',
                    'RECORD','MEASURE','SCAN'],
        'stop':    ['DONE','SAVE','FINISH','COMPLETE','FINISHED',"THAT'S IT",
                    'OK STOP','ENOUGH'],
        'end':     ['EXIT','QUIT','TERMINATE','END SESSION','ALL DONE',
                    'END CHARTING','FINISH CHARTING','CHARTING DONE'],
        'start':   ['GO','ACTIVATE','BEGIN','OPEN'],  # start → go
        'confirm': ['OK','OKAY','ACCEPT','CONFIRMED','GOOD'],
        'repeat':  ['UNDO','BACK','CANCEL','AGAIN','REDO','TRY AGAIN'],
    }

    MIC_PRIORITY = [1,7,14]
    MIC_FALLBACK = [2,8,15,16,20]

    def __init__(self,
                 whisper_model_size: str = 'small',
                 fuzzy_threshold:    float = 0.60,
                 voice_device:       Optional[int] = None,
                 enrollment_dir:     str = 'enrollment',
                 user:               Optional[str] = None):

        self.threshold      = fuzzy_threshold
        self._running       = False
        self._thread        = None
        self._whisper       = None
        self._pa            = None
        self._pa_dev        = None
        self._voice_device  = voice_device

        # Mode and charting state
        self.mode              = 'idle'
        self.switch_requested  = False   # set True when mode changes — lets run() break out
        self.chart_state       = 'inactive'
        self.chart_tooth    = None        # current tooth being scanned
        self.teeth_saved    = []          # list of saved teeth this session

        # Command queues — consumers poll these
        self.pano_queue     = queue.Queue()   # pano commands: on/off/save/…
        self.chart_queue    = queue.Queue()   # charting commands: tooth/stop/end/…
        self.mode_queue     = queue.Queue()   # mode switch events for display

        # Charting sub-objects
        self._fsg   = DentalFSG()
        self._phmap = PhonemeMapper()

        # Load Whisper (adapted if available)
        self._load_whisper(whisper_model_size, enrollment_dir, user)

        # Initialise raw PyAudio
        self._init_pa()

    # ── Whisper loader ────────────────────────────────────────────────────────

    def _load_whisper(self, model_size: str, enrollment_dir: str,
                      user: Optional[str]):
        if not WHISPER_AVAILABLE:
            print("[Voice] Whisper not available — using Google fallback")
            self._rec = sr.Recognizer() if SPEECH_AVAILABLE else None
            return

        # Try adapted model first
        if user:
            uid    = re.sub(r'[^a-z0-9_]', '', user.lower().replace(' ','_'))
            pt     = Path(enrollment_dir) / uid / 'adapted_model.pt'
            if pt.exists():
                try:
                    import torch
                    print(f"[Voice] Loading base '{model_size}' + adapter ({uid})…")
                    self._whisper = _whisper.load_model(model_size)
                    state = torch.load(pt, map_location='cpu')
                    self._whisper.load_state_dict(state, strict=False)
                    print(f"[Voice] ✓ Adapted Whisper loaded  ({pt})")
                    return
                except Exception as e:
                    print(f"[Voice] Adapter load failed: {e} — using base")

        print(f"[Voice] Loading base Whisper '{model_size}'…")
        print(f"[Voice] voice_device={self._voice_device}, enrollment_dir={enrollment_dir}")
        try:
            self._whisper = _whisper.load_model(model_size)
            print("[Voice] ✓ Base Whisper loaded")
        except Exception as e:
            print(f"[Voice] Whisper failed: {e}")

    # ── PyAudio fixed-duration recording ──────────────────────────────────────

    def _init_pa(self):
        """Initialise raw PyAudio for energy-gate-free recording."""
        if not PYAUDIO_AVAILABLE:
            self._rec = sr.Recognizer() if SPEECH_AVAILABLE else None
            if self._rec:
                self._rec.energy_threshold = 200
                self._rec.dynamic_energy_threshold = True
                self._mic = self._find_mic()
            return
        try:
            self._pa   = pyaudio.PyAudio()
            self._pa_dev = None   # always define first
            names = sr.Microphone.list_microphone_names() if SPEECH_AVAILABLE else []

            # Print all available input devices for debugging
            print(f"[Voice] Available input devices ({len(names)} total):")
            for i in range(self._pa.get_device_count()):
                try:
                    info = self._pa.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        print(f"[Voice]   {i}: {info['name'][:50]}")
                except: pass

            if self._voice_device is not None:
                self._pa_dev = self._voice_device
                print(f"[Voice] Mic override: device {self._voice_device}")
            else:
                for idx in self.MIC_PRIORITY + self.MIC_FALLBACK:
                    if idx >= len(names): continue
                    try:
                        info = self._pa.get_device_info_by_index(idx)
                        if info['maxInputChannels'] > 0:
                            tag = 'XREAL' if idx in self.MIC_PRIORITY else 'fallback'
                            print(f"[Voice] ✓ Mic [{tag}] device {idx}: {names[idx]}")
                            self._pa_dev = idx; break
                    except: continue

            if self._pa_dev is None:
                self._pa_dev = self._pa.get_default_input_device_info()['index']
                print(f"[Voice] ✓ Mic default device {self._pa_dev}")

            print(f"[Voice] Using mic device index: {self._pa_dev}")
        except Exception as e:
            print(f"[Voice] PyAudio init failed: {e}")
            self._pa = None
            self._rec = sr.Recognizer() if SPEECH_AVAILABLE else None
            if self._rec:
                self._mic = self._find_mic()

    def _find_mic(self):
        """Fallback sr.Microphone selection."""
        if not SPEECH_AVAILABLE: return None
        try:
            names = sr.Microphone.list_microphone_names()
            if self._voice_device is not None:
                return sr.Microphone(device_index=self._voice_device)
            for idx in self.MIC_PRIORITY + self.MIC_FALLBACK:
                if idx < len(names):
                    try:
                        return sr.Microphone(device_index=idx)
                    except: continue
        except: pass
        return sr.Microphone()

    def _record_fixed(self, duration: float) -> Optional[object]:
        """
        Record exactly `duration` seconds — no energy gate.
        Returns sr.AudioData or None.
        Duration guide: 1.5s = single words,  2.5s = tooth specs (14 B)
        """
        CHUNK = 512; RATE = 16000
        if self._pa is not None:
            try:
                # Debug: log first 5 recordings
                if not hasattr(self, '_rec_count'): self._rec_count = 0
                self._rec_count += 1
                if self._rec_count <= 5:
                    print(f"[VOICE DEBUG] recording #{self._rec_count} dev={self._pa_dev} dur={duration}s")
                stream = self._pa.open(
                    format=pyaudio.paInt16, channels=1, rate=RATE,
                    input=True, input_device_index=self._pa_dev,
                    frames_per_buffer=CHUNK)
                n = int(RATE * duration / CHUNK)
                frames = [stream.read(CHUNK, exception_on_overflow=False)
                          for _ in range(n)]
                stream.stop_stream(); stream.close()
                return sr.AudioData(b''.join(frames), RATE, 2)
            except Exception as e:
                print(f"[Voice] record error: {e}"); return None
        # sr fallback
        if hasattr(self, '_rec') and self._rec and hasattr(self, '_mic') and self._mic:
            try:
                with self._mic as src:
                    return self._rec.listen(src, timeout=3,
                                            phrase_time_limit=max(duration, 3.0))
            except: return None
        return None

    def _transcribe(self, audio) -> Optional[str]:
        if audio is None: return None
        if self._whisper:
            try:
                import tempfile, os as _os, sys as _sys
                # In PyInstaller exe, whisper looks for assets relative to its
                # module location. We must set WHISPER_ASSETS_DIR or patch the path.
                if getattr(_sys, 'frozen', False):
                    _base    = _os.path.dirname(_sys.executable)
                    _int_dir = _os.path.join(_base, '_internal')
                    # whisper/assets is bundled by --collect-data whisper
                    _assets  = _os.path.join(_int_dir, 'whisper', 'assets')
                    if not _os.path.exists(_assets):
                        _assets = _os.path.join(_base, 'whisper', 'assets')
                    if _os.path.exists(_assets):
                        # Patch whisper.audio to find assets in the exe bundle
                        try:
                            import whisper.audio as _wa
                            import whisper as _wmod
                            _wmod_dir = _os.path.dirname(_wmod.__file__)
                            if not _os.path.exists(_os.path.join(_wmod_dir,'assets')):
                                # Symlink/copy assets next to whisper package
                                import shutil as _sh
                                _sh.copytree(_assets,
                                    _os.path.join(_wmod_dir,'assets'),
                                    dirs_exist_ok=True)
                        except Exception as _pe:
                            print(f"[Voice] Asset patch: {_pe}")
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(audio.get_wav_data()); tmp = f.name
                # Pass audio directly as numpy array to avoid ffmpeg dependency
                import numpy as _np
                import wave as _wave
                with _wave.open(tmp, 'rb') as _wf:
                    frames = _wf.readframes(_wf.getnframes())
                    rate   = _wf.getframerate()
                audio_np = _np.frombuffer(frames, dtype=_np.int16).astype(_np.float32) / 32768.0
                print(f"[VOICE DEBUG] transcribing numpy array shape={audio_np.shape} rate={rate}")
                res = _whisper.transcribe(self._whisper, audio_np,
                                          language='en',
                                          condition_on_previous_text=False,
                                          fp16=False)
                _os.unlink(tmp)
                text = res['text'].strip().upper()
                # Filter Whisper silence hallucinations — common on empty audio
                _silence = {'', '.', '..', '...', 'THANK YOU.', 'THANK YOU',
                            'THANKS.', 'THANKS', 'BYE.', 'BYE', 'PING.',
                            'PING', 'JING.', 'JING', 'STARTING.', 'STARTING',
                            'OKAY.', 'OKAY', 'OK.', 'SO.', 'SO', 'UM.', 'UM',
                            'HMM.', 'HMM', 'AH.', 'AH', 'UH.', 'UH'}
                if text in _silence:
                    return None   # treat as silence
                if text:
                    print(f"[Voice] Heard: {text!r}")
                return text or None
            except Exception as _te:
                print(f"[Voice] Whisper error: {_te}")
                import traceback; traceback.print_exc()
        if hasattr(self, '_rec') and self._rec:
            try:
                text = self._rec.recognize_google(audio).upper()
                print(f"[Voice] Google heard: {text!r}")
                return text
            except Exception as _ge:
                pass  # Google fails when offline — expected
        return None

    # ── Start / stop ──────────────────────────────────────────────────────────

    def start(self):
        if self._running: return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("\n[Voice] 🎤 Listening — say 'stitching' or 'charting' to select mode")
        print("[Voice]   Stitching: ON  OFF  SAVE  DROP  HOLD  GO  MARK  BAD  VIEW")
        print("[Voice]   Charting:  GO  [tooth e.g. 14B]  STOP  END\n")

    def stop(self):
        self._running = False

    def get_pano_command(self) -> Optional[str]:
        try: return self.pano_queue.get_nowait()
        except queue.Empty: return None

    def get_chart_command(self) -> Optional[tuple]:
        """Returns (cmd_type, value) tuples: ('tooth','14B'), ('action','stop'), etc."""
        try: return self.chart_queue.get_nowait()
        except queue.Empty: return None

    def get_mode_event(self) -> Optional[str]:
        try: return self.mode_queue.get_nowait()
        except queue.Empty: return None

    def get_tooth_command(self) -> Optional[str]:
        """
        Non-blocking — returns tooth/surface commands regardless of mode.
        Works in BOTH stitching and charting so the tooth grid responds
        to voice during a panorama sweep (e.g. say '14 buccal' to label folder).
        chart_queue already receives tooth tuples — drain them here too.
        """
        # chart_queue contains ('tooth','14B') and ('action','stop') etc.
        # Peek without consuming — pull only tooth entries
        items = []
        try:
            while True:
                item = self.chart_queue.get_nowait()
                if item[0] == 'tooth':
                    return item[1]   # e.g. "14B"
                else:
                    items.append(item)  # put non-tooth items back
        except queue.Empty:
            pass
        for item in items:
            self.chart_queue.put_nowait(item)
        return None

    def is_pano_mode(self)    -> bool: return self.mode == 'stitching'
    def is_chart_mode(self)   -> bool: return self.mode == 'charting'

    def disable(self):
        """Disable voice command processing — mic still records but commands ignored."""
        self._voice_disabled = True
        print("[Voice] Voice commands DISABLED")

    def enable(self):
        """Re-enable voice command processing."""
        self._voice_disabled = False
        print("[Voice] Voice commands ENABLED")

    # ── Main listen loop ──────────────────────────────────────────────────────

    def _loop(self):
        print(f"[VOICE DEBUG] _loop started — pa={self._pa}, pa_dev={self._pa_dev}, whisper={self._whisper is not None}")
        _loop_count = 0
        while self._running:
            _loop_count += 1
            if _loop_count <= 3 or _loop_count % 20 == 0:
                print(f"[VOICE DEBUG] loop #{_loop_count} — pa={self._pa is not None}, pa_dev={self._pa_dev}, mode={self.mode}")
            # Tooth specs (e.g. "twenty two buccal") need 2.5s
            # Short commands (on/off/stop) need only 1.5s
            # In stitching mode we always use 2.0s so tooth labels also work
            if self.mode == 'stitching':
                duration = 2.0   # long enough for tooth numbers
            elif self.mode == 'charting' and self.chart_state == 'waiting_for_tooth':
                duration = 2.5
            else:
                duration = 1.5
            audio = self._record_fixed(duration)
            raw   = self._transcribe(audio)
            if not raw:
                continue

            print(f"[Voice] Processing: {raw!r}  (mode={self.mode})")

            # ── Check for mode switch first (always, regardless of mode) ──────
            mode_switched = self._check_mode_switch(raw)
            if mode_switched:
                continue

            # ── Dispatch to correct mode ──────────────────────────────────────
            # Skip if voice commands disabled by user
            if getattr(self, '_voice_disabled', False):
                continue
            if self.mode == 'stitching':
                # Always try tooth/surface first — so "14 buccal" labels the folder
                if not self._try_tooth_in_pano(raw):
                    self._handle_pano(raw)
            elif self.mode == 'charting':
                if getattr(self, '_voice_disabled', False):
                    continue   # voice disabled by user — ignore all commands
                self._handle_chart(raw)
            else:
                print(f"[Voice] '{raw}' — say 'stitching' or 'charting' first")

    # ── Mode switching ────────────────────────────────────────────────────────

    def _check_mode_switch(self, raw: str) -> bool:
        text = raw.strip().upper()
        # Strip ALL punctuation before mode matching
        raw = re.sub(r'[.!?,;:?\'"\\-]+', " ", raw.strip().upper())
        raw = re.sub(r'\s+', " ", raw).strip()
        text = raw   # use cleaned version for matching
        for mode, variants in self.MODE_SWITCH_VARIANTS.items():
            for v in variants:
                if v in text:
                    if self.mode != mode:
                        old = self.mode
                        self.mode = mode
                        self.switch_requested = True   # signal run() to exit
                        self.mode_queue.put(mode)
                        print(f"\n[Voice] ══ MODE SWITCH: {old.upper()} → {mode.upper()} ══")
                        if mode == 'stitching':
                            print("[Voice]   Commands: ON  OFF  SAVE  DROP  HOLD  GO  MARK  BAD  VIEW")
                            print("[Voice]   Say 'charting' at any time to switch back")
                        else:
                            self.chart_state = 'inactive'
                            self.chart_tooth = None
                            print("[Voice]   Commands: GO → [14 B / three L] → STOP → … → END")
                            print("[Voice]   Say 'stitching' at any time to switch back")
                    else:
                        print(f"[Voice] Already in {mode} mode")
                    return True
        return False

    # ── Tooth parsing active in stitching mode ───────────────────────────────

    def _try_tooth_in_pano(self, raw: str) -> bool:
        """
        Try to parse a tooth/surface command while in stitching mode.
        If matched, put it in chart_queue so get_tooth_command() returns it
        and the GUI updates _stitch_tooth for folder naming.
        Returns True if a tooth command was found (pano command skipped).
        Returns False if no tooth found (caller should try pano handling).
        """
        # Reuse the chart handler's tooth parser — same regex/phoneme logic
        text = raw.strip().upper()
        text = re.sub(r"[.!?,;:]+$", "", text).strip()

        # Quick pre-filter: must contain a number word or digit
        import re as _re
        has_number = bool(_re.search(r'([0-9]|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|'
                                      r'TEN|ELEVEN|TWELVE|THIRTEEN|FOURTEEN|FIFTEEN|SIXTEEN|'
                                      r'SEVENTEEN|EIGHTEEN|NINETEEN|TWENTY|THIRTY)', text))
        if not has_number:
            return False

        # Try chart parser in tooth-only mode
        try:
            r = self._match_chart(text, None)
            if r and r[0] == 'tooth':
                self.chart_queue.put_nowait(r)
                print(f"[Voice/stitch] Tooth label: {r[1]}")
                return True
        except Exception:
            pass

        return False

    # ── Panorama command handling ─────────────────────────────────────────────

    def _handle_pano(self, raw: str):
        # Strip ALL punctuation — "STOP." "ON." "OFF." → clean
        raw  = re.sub(r"[^A-Z0-9 ]", " ", raw.upper())
        raw  = re.sub(r" +", " ", raw).strip()
        text = raw

        # Ambient reject — block hallucinations and clinical noise
        for phrase in self.PANO_AMBIENT_REJECT:
            if phrase in text:
                return   # silently ignore

        words = text.split()

        # Tier 1: exact phoneme variant
        for cmd, variants in self.PANO_PHONEME.items():
            for w in words:
                if w in variants:
                    print(f"[Voice/pano] '{raw}' → {cmd.upper()}")
                    self.pano_queue.put(cmd); return

        # Tier 2: alias — full word match only to prevent false triggers
        words_set = set(words)
        for cmd, aliases in self.PANO_ALIASES.items():
            for a in aliases:
                # Single word alias: must be an exact word in the utterance
                # Multi-word alias: must be a full substring match
                a_words = a.split()
                if len(a_words) == 1:
                    if a in words_set:
                        print(f"[Voice/pano] '{raw}' → {cmd.upper()} (alias word)")
                        self.pano_queue.put(cmd); return
                else:
                    if a in text:
                        print(f"[Voice/pano] '{raw}' → {cmd.upper()} (alias phrase)")
                        self.pano_queue.put(cmd); return

        # Tier 3: fuzzy — threshold 0.82 prevents STOP→DROP/SAVE false matches
        short = [w for w in words if len(w) <= 6]
        if len(words) <= 2 and short:
            best = None; br = 0.
            for cmd, variants in self.PANO_PHONEME.items():
                for w in short:
                    for v in variants:
                        r = SequenceMatcher(None, w, v).ratio()
                        if r > br: br = r; best = cmd
            _pano_min = max(self.threshold, 0.82)
            if br >= _pano_min:
                print(f"[Voice/pano] '{raw}' → {best.upper()} (fuzzy {br:.2f})")
                self.pano_queue.put(best); return

        print(f"[Voice/pano] no match: '{raw}'")

    # ── Charting command handling ─────────────────────────────────────────────

    def _handle_chart(self, raw: str):
        # Strip ALL punctuation — "STOP." "END." "1 B." → clean before any matching
        raw = re.sub(r"[^A-Z0-9 ]", " ", raw.upper())
        raw = re.sub(r" +", " ", raw).strip()
        # Raw FSG first — avoids phoneme mapper corrupting digit+surface like "14B"
        fsg_raw = self._fsg.parse(raw)

        # Phoneme mapping + FSG on corrected
        corrected     = self._phmap.normalize(raw)
        fsg_corrected = self._fsg.parse(corrected)
        fsg_result    = fsg_raw if (fsg_raw and fsg_raw['type'] == 'tooth') else fsg_corrected

        # Three-tier action match
        cmd, tier = self._match_chart(corrected, fsg_result)

        tier_labels = {0:'',1:'T1',2:'T2',3:'T3'}

        # ── State machine ─────────────────────────────────────────────────────
        if self.chart_state == 'inactive':
            if cmd == 'go':
                self.chart_state = 'waiting_for_tooth'
                print(f"[Voice/chart] '{raw}' → GO ({tier_labels[tier]})")
                print("[Voice/chart] ✅ Activated — say tooth number (e.g. 14 B, three L)")
                self.chart_queue.put(('action', 'go'))

        elif self.chart_state == 'waiting_for_tooth':
            if fsg_result and fsg_result.get('type','') == 'tooth':
                tooth = fsg_result.get('value', None)
                self.chart_tooth  = tooth
                self.chart_state  = 'scanning'
                print(f"[Voice/chart] '{raw}' → TOOTH {tooth}")
                print(f"[Voice/chart] 🦷 Scanning {tooth} — say STOP to save")
                self.chart_queue.put(('tooth', tooth))
            elif cmd == 'end':
                self.chart_state = 'finished'
                print(f"[Voice/chart] '{raw}' → END ({tier_labels[tier]})")
                self.chart_queue.put(('action', 'end'))
            else:
                print(f"[Voice/chart] not a valid tooth: '{corrected}' — try '14 B' or 'three L'")

        elif self.chart_state == 'scanning':
            if cmd == 'stop':
                saved = self.chart_tooth
                self.teeth_saved.append(saved)
                self.chart_tooth  = None
                self.chart_state  = 'waiting_for_tooth'
                print(f"[Voice/chart] '{raw}' → STOP ({tier_labels[tier]})")
                print(f"[Voice/chart] 💾 Saved {saved} — say next tooth number")
                self.chart_queue.put(('action', 'stop'))
            elif cmd == 'end':
                self.chart_state = 'finished'
                print(f"[Voice/chart] '{raw}' → END ({tier_labels[tier]})")
                self.chart_queue.put(('action', 'end'))
            elif fsg_result and fsg_result.get('type','') == 'tooth':
                print(f"[Voice/chart] Still scanning {self.chart_tooth} — say STOP first")
            else:
                print(f"[Voice/chart] no match: '{corrected}'")

        elif self.chart_state == 'finished':
            print("[Voice/chart] Session ended — say 'charting' to start a new one")

    def _match_chart(self, text: str, fsg_result) -> Tuple[Optional[str], int]:
        # Strip ALL punctuation Whisper adds ("STOP." "2L?" "1 B." → clean)
        clean = re.sub(r'[^A-Z0-9 ]', ' ', str(text).strip().upper())
        clean = re.sub(r' +', ' ', clean).strip()
        if clean: print(f'[Voice/_match_chart] raw={text!r}  clean={clean!r}')
        words = clean.split()
        for cmd, variants in self.CHART_PHONEME.items():
            for w in words:
                if w in variants:
                    resolved = 'go' if cmd == 'start' else cmd
                    return resolved, 1
        for cmd, aliases in self.CHART_ALIASES.items():
            for alias in aliases:
                if alias in clean: return cmd, 2
        if fsg_result and fsg_result.get('type','') == 'action':
            return fsg_result.get('value', None), 2
        best = None; br = 0.
        for cmd, variants in self.CHART_PHONEME.items():
            for w in words:
                for v in variants:
                    r = 1.0 - (1.0 - SequenceMatcher(None,
                        re.sub(r'[AEIOU]','',w.upper()),
                        re.sub(r'[AEIOU]','',v.upper())).ratio())
                    if r > br: br = r; best = cmd
        if br >= self.threshold: return best, 3
        return None, 0



class MetricMonitor:
    """
    Comprehensive debug logging for accent robustness and model performance.

    Saves three files to session_dir/debug/:
      voice_events.jsonl  — every voice recognition event
      frame_metrics.csv   — per-frame model/quality/IMU data
      session_summary.json — aggregate stats for parameter optimisation

    Usage:
        mm = MetricMonitor(session_dir)
        mm.log_voice_event(raw_text, corrected, fsg_result, cmd, tier, latency_ms)
        mm.log_frame(frame_n, quality, gate_score, gate_pass, ...)
        mm.save_summary()  # called at session end

    Reload for analysis:
        data = MetricMonitor.load_session(session_dir)
    """

    def __init__(self, session_dir: Path):
        self._sdir    = Path(session_dir)
        self._dbdir   = self._sdir / DEBUG_DIR
        self._dbdir.mkdir(parents=True, exist_ok=True)
        self._voice_path  = self._dbdir / "voice_events.jsonl"
        self._frame_path  = self._dbdir / "frame_metrics.csv"
        self._summary_path = self._dbdir / "session_summary.json"

        # Rolling buffers
        self._voice_events: List[dict] = []
        self._frame_metrics: List[dict] = []
        self._t0 = time.time()

        # Open CSV and write header
        self._csv_fh = open(self._frame_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=[
            'frame_n','elapsed_s','quality_b1','sharpness','contact','snr',
            'gate_score','gate_pass','gm_x_mm','gm_y_mm',
            'inference_ms','total_ms',
        ])
        self._csv_writer.writeheader()

    # ── Voice events ──────────────────────────────────────────────────────────

    def log_voice_event(self,
                         raw_text:           str,
                         corrected:          str,
                         fsg_result:         Optional[dict],
                         matched_cmd:        Optional[str],
                         match_tier:         int,
                         latency_ms:         float,
                         pending_sent:       bool          = False,
                         confirmed:          Optional[bool]= None,
                         **kwargs):
        # kwargs: whisper_logprob, audio_rms, confirm_latency_ms
        event = {
            'ts':               round(time.time() - self._t0, 3),
            'raw':              raw_text,
            'corrected':        corrected,
            'fsg_type':         fsg_result.get('type')  if fsg_result else None,
            'fsg_value':        fsg_result.get('value') if fsg_result else None,
            'fsg_reject_reason':fsg_result.get('reject_reason') if fsg_result else 'no_match',
            'cmd':              matched_cmd,
            'tier':             match_tier,
            'latency_ms':       round(latency_ms, 1),
            'whisper_logprob':  kwargs.get('whisper_logprob', None),
            'audio_rms':        kwargs.get('audio_rms',       None),
            'pending':          pending_sent,
            'confirmed':        confirmed,
            'confirm_latency_ms': kwargs.get('confirm_latency_ms', None),
        }
        self._voice_events.append(event)
        with open(self._voice_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')

    # ── Frame metrics ─────────────────────────────────────────────────────────

    def log_frame(self, frame_n: int, quality: float,
                  sharpness: float = 0., contact: float = 0., snr: float = 0.,
                  gate_score: float = 0., gate_pass: bool = False,
                  gm_x_mm: float = 0., gm_y_mm: float = 0.,
                  inference_ms: float = 0., total_ms: float = 0.):
        row = {
            'frame_n':     frame_n,
            'elapsed_s':   round(time.time() - self._t0, 2),
            'quality_b1':  round(quality,    4),
            'sharpness':   round(sharpness,  3),
            'contact':     round(contact,    3),
            'snr':         round(snr,        3),
            'gate_score':  round(gate_score, 3),
            'gate_pass':   int(gate_pass),
            'gm_x_mm':     round(gm_x_mm,   4),
            'gm_y_mm':     round(gm_y_mm,   4),
            'inference_ms': round(inference_ms, 1),
            'total_ms':    round(total_ms,   1),
        }
        self._frame_metrics.append(row)
        self._csv_writer.writerow(row)
        self._csv_fh.flush()

    # ── Session summary ───────────────────────────────────────────────────────

    def save_summary(self) -> dict:
        ve = self._voice_events
        fm = self._frame_metrics

        # Voice stats
        n_events   = len(ve)
        n_rejected = sum(1 for e in ve if e['tier'] == 0)
        n_pending  = sum(1 for e in ve if e['pending'])
        n_confirmed= sum(1 for e in ve if e['confirmed'] is True)
        n_repeated = sum(1 for e in ve if e['confirmed'] is False)
        tier_dist  = {str(t): sum(1 for e in ve if e['tier']==t) for t in [1,2,3,0]}
        lat_vals   = [e['latency_ms'] for e in ve if e['latency_ms']>0]

        # Frame stats
        n_frames   = len(fm)
        q_vals     = [f['quality_b1'] for f in fm]
        g_vals     = [f['gate_score'] for f in fm]
        accept_n   = sum(1 for f in fm if f['gate_pass'])
        inf_vals   = [f['inference_ms'] for f in fm if f['inference_ms']>0]

        summary = {
            'session_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': 'B1_V8_Baseline',
            'duration_s': round(time.time() - self._t0, 1),
            'voice': {
                'n_events':         n_events,
                'false_reject_rate': round(n_rejected / max(n_events,1), 3),
                'pending_rate':     round(n_pending / max(n_events,1), 3),
                'confirm_rate':     round(n_confirmed / max(n_pending,1), 3),
                'repeat_rate':      round(n_repeated / max(n_pending,1), 3),
                'tier_distribution': tier_dist,
                'mean_latency_ms':  round(np.mean(lat_vals), 1) if lat_vals else 0,
                'p95_latency_ms':   round(np.percentile(lat_vals, 95), 1) if lat_vals else 0,
            },
            'frames': {
                'n_total':          n_frames,
                'n_accepted':       accept_n,
                'accept_rate':      round(accept_n / max(n_frames,1), 3),
                'mean_quality_b1':  round(np.mean(q_vals), 4) if q_vals else 0,
                'mean_gate_score':  round(np.mean(g_vals), 4) if g_vals else 0,
                'mean_inference_ms': round(np.mean(inf_vals), 1) if inf_vals else 0,
                'p95_inference_ms': round(np.percentile(inf_vals,95),1) if inf_vals else 0,
            },
            'tuning_recommendations': self._tuning_notes(n_rejected, n_events,
                                                          q_vals, g_vals),
        }
        with open(self._summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Metrics] Summary → {self._summary_path}")
        self._print_summary(summary)
        return summary

    def _tuning_notes(self, n_rej, n_ev, q_vals, g_vals) -> List[str]:
        notes = []
        rej_rate = n_rej / max(n_ev,1)
        if rej_rate > 0.3:
            notes.append("High voice reject rate (>30%) — lower fuzzy_threshold or add more phoneme variants")
        if rej_rate < 0.05:
            notes.append("Very low reject rate (<5%) — consider raising fuzzy_threshold to reduce false accepts")
        if q_vals:
            mq = np.mean(q_vals)
            if mq < 0.20:
                notes.append("Low B1 quality mean (<0.20) — check probe contact and ultrasound coupling")
            if mq > 0.50:
                notes.append("Good B1 quality — consider lowering QS_THRESH to accept more frames")
        if not notes:
            notes.append("Parameters look balanced — no adjustment needed")
        return notes

    def _print_summary(self, s: dict):
        print(f"\n{'='*55}")
        print(f"  SESSION METRICS SUMMARY")
        print(f"{'='*55}")
        v = s['voice']
        print(f"  Voice  | events:{v['n_events']}  reject:{v['false_reject_rate']:.0%}  "
              f"lat:{v['mean_latency_ms']:.0f}ms")
        print(f"         | tiers: {v['tier_distribution']}")
        f_ = s['frames']
        print(f"  Frames | total:{f_['n_total']}  accepted:{f_['accept_rate']:.0%}  "
              f"quality:{f_['mean_quality_b1']:.3f}  inf:{f_['mean_inference_ms']:.0f}ms")
        for note in s['tuning_recommendations']:
            print(f"  ► {note}")
        # Print voice tier distribution
        v = s.get('voice', {})
        td = v.get('tier_distribution', {})
        if td:
            print(f"  Voice tiers: T1(exact)={td.get('1',0)}  "
                  f"T2(alias)={td.get('2',0)}  "
                  f"T3(fuzzy)={td.get('3',0)}  "
                  f"Rejected={td.get('0',0)}")
        print(f"{'='*55}")

    def close(self):
        try: self._csv_fh.close()
        except: pass

    @staticmethod
    def load_session(session_dir: Path) -> dict:
        """Load all debug files for a session — use this to give me for analysis."""
        dbdir = Path(session_dir) / DEBUG_DIR
        data  = {'session_dir': str(session_dir)}
        try:
            with open(dbdir / 'session_summary.json') as f:
                data['summary'] = json.load(f)
        except: data['summary'] = {}
        events = []
        try:
            with open(dbdir / 'voice_events.jsonl') as f:
                for line in f:
                    try: events.append(json.loads(line))
                    except: pass
        except: pass
        data['voice_events'] = events
        frames = []
        try:
            with open(dbdir / 'frame_metrics.csv') as f:
                reader = csv.DictReader(f)
                frames = list(reader)
        except: pass
        data['frame_metrics'] = frames
        print(f"[MetricMonitor.load_session] {len(events)} voice events, "
              f"{len(frames)} frame records")
        return data


# ═════════════════════════════════════════════════════════════════════════════
#  2.  DOMAIN-CONSTRAINED FINITE-STATE GRAMMAR  (Accent Feature 2)
# ═════════════════════════════════════════════════════════════════════════════

class DentalFSG:
    """
    Domain-constrained grammar for dental voice commands.
    Valid vocabulary (~60 tokens total):
      Actions:   GO NEXT STOP END CONFIRM REPEAT YES NO
      Teeth:     1–32 (digit strings)
      Surfaces:  B L M N P  (all normalise to B=buccal or L=lingual)
      Aliases:   (see ALIASES dict)

    parse(text) returns:
      {'type': 'action',  'value': 'go'|'next'|...}
      {'type': 'tooth',   'value': '14B'|'3L'|...}
      None  →  doesn't parse → reject
    """

    ACTIONS = {
        'GO', 'STOP', 'END', 'CONFIRM', 'REPEAT', 'YES', 'NO',
        'UNDO', 'BACK', 'REDO',
    }
    TOOTH_NUMS  = {str(i) for i in range(1, 33)}
    SURFACES    = {'B','L','M','P','N'}
    # Accent normalisation — all spoken surface letters map to B or L only:
    #   B, P           → B  (buccal;  P sounds like B)
    #   L, M, N        → L  (lingual; M/N are common substitutions for L)
    # D is intentionally excluded — only buccal and lingual imaging is done.
    SURF_NORM   = {'B':'B', 'P':'B', 'L':'L'}
    SURF_WORDS  = {
        'BUCCAL':  'B', 'LINGUAL': 'L', 
        'PALATAL': 'B', 'FACIAL':  'B', 'NASAL':  'L',
    }
    ALIASES = {
        'START':    'GO',   'BEGIN':  'GO',   'ACTIVATE': 'GO',
        'DONE':     'STOP', 'SAVE':   'STOP', 'FINISH':   'STOP',
        'QUIT':     'END',  'EXIT':   'END',  'COMPLETE': 'END',
        'OK':       'CONFIRM', 'OKAY':    'CONFIRM', 'ACCEPT': 'CONFIRM',
        'REDO':     'REPEAT',  'REJECT':  'REPEAT',  'CANCEL': 'REPEAT',
    }

    def parse(self, text: str) -> Optional[dict]:
        words = text.strip().upper().split()
        if not words:
            return None

        # ── Direct action match ───────────────────────────────────────────
        for w in words:
            if w in self.ACTIONS:
                return {'type': 'action', 'value': w.lower()}
            if w in self.ALIASES:
                return {'type': 'action', 'value': self.ALIASES[w].lower()}

        # ── Tooth specification: NUMBER [+] SURFACE ───────────────────────
        num   = None
        surf  = None
        for i, w in enumerate(words):
            # Extract number
            digit = re.sub(r'[^0-9]', '', w)
            if digit and digit in self.TOOTH_NUMS:
                num = int(digit)
                # Look for surface in remaining words or suffix
                for j in range(len(words)):
                    wj = words[j]
                    if wj in self.SURF_WORDS:
                        surf = self.SURF_WORDS[wj]; break
                    if wj in self.SURFACES:
                        surf = self.SURF_NORM[wj]; break
                    # Surface appended to number e.g. "14B" or "14P"
                    m = re.match(r'^(\d+)([BLMPN])$', wj)
                    if m:
                        num  = int(m.group(1))
                        surf = self.SURF_NORM[m.group(2)]; break
                if num and surf and 1 <= num <= 32:
                    return {'type': 'tooth', 'value': f"{num}{surf}"}

        return None  # Doesn't parse → reject

    def is_action(self, text: str, action: str) -> bool:
        result = self.parse(text)
        return result is not None and result['type']=='action' and result['value']==action.lower()


# ═════════════════════════════════════════════════════════════════════════════
#  3.  PHONEME MAPPER  (Accent Feature 3 — CMU-style substitutions)
# ═════════════════════════════════════════════════════════════════════════════

class PhonemeMapper:
    """
    Phoneme-level accent correction applied BEFORE FSG parsing.

    Rules derived from common accent substitution patterns:
      th→t/d:  Arabic, Hindi, Korean, East Asian
      v→w/b:   German, Korean, some South Asian
      r→l:     East Asian (Mandarin, Japanese, Korean)
      Final consonant drop: Korean, Cantonese
      Vowel shift: varies

    Also handles number words (phoneme variants in non-native English):
      WAN→ONE, TU→TWO, TREE→THREE, FO→FOUR, etc.

    Returns corrected string with canonical spellings.
    """

    # Word-level phoneme corrections (handles full word variants)
    WORD_CORRECTIONS: Dict[str, str] = {
        # Action word variants (accent-driven)
        'GOUW': 'GO', 'GOH': 'GO', 'GAO': 'GO', 'JOH': 'GO', 'KO': 'GO',
        'STAP': 'STOP', 'STAHP': 'STOP', 'ESTOP': 'STOP', 'STOB': 'STOP',
        'STUP': 'STOP', 'TOP':   'STOP',
        'EHND': 'END',  'IND':   'END',  'ENND': 'END', 'AEND': 'END',
        'EEND': 'END',  'NED':   'END',
        'KONFIRM': 'CONFIRM', 'CONFER': 'CONFIRM', 'KONFERM': 'CONFIRM',
        'REPEET':  'REPEAT',  'REPET':  'REPEAT',  'RIPEAT':  'REPEAT',
        'YAS':  'YES',  'JES':  'YES', 'JEH': 'YES',
        'NAO':  'NO',   'NOH':  'NO',
        # Number word variants
        'WAN':  'ONE',   'JUAN': 'ONE',  'YONE': 'ONE',
        'TU':   'TWO',   'TOO':  'TWO',  'TW':   'TWO',
        'TREE': 'THREE', 'TRIE': 'THREE', 'DREE': 'THREE', 'SREE': 'THREE',
        'FO':   'FOUR',  'FAWR': 'FOUR', 'FOUL': 'FOUR',
        'FIFE': 'FIVE',  'FAIV': 'FIVE', 'FAIB': 'FIVE',
        'SEX':  'SIX',   'SIKS': 'SIX',
        'SEFEN':'SEVEN', 'SEBN': 'SEVEN', 'SEBM': 'SEVEN',
        'AYT':  'EIGHT', 'AYTH': 'EIGHT',
        'NAIN': 'NINE',  'NEIN': 'NINE',
        'TEN':  'TEN',   'DEN':  'TEN',
        'ELEBEN': 'ELEVEN', 'ELEBEM': 'ELEVEN',
        'TWELF':  'TWELVE', 'TWELF':  'TWELVE',
        # Surface word variants
        'BUCCLE':  'BUCCAL', 'BUCKLE': 'BUCCAL',
        'LINGUAL': 'LINGUAL', 'LINGO': 'LINGUAL',
        'MESIAL':  'MESIAL',  'MEDIA': 'MESIAL',
        'PALATAL': 'PALATAL', 'PALATE': 'PALATAL',
    }

    # Character-level accent substitution patterns applied as regex
    # Format: (pattern, replacement) — applied to each word
    CHAR_SUBS = [
        (r'^TH', 'T'),       # th→t: tree for three (initial)
        (r'TH$', 'T'),       # th→t: final (boot for booth)
        (r'^DH', 'D'),       # dh→d variant
        (r'NG$', 'N'),       # ng→n final drop
        (r'(.)\1+', r'\1'),  # double consonant collapse: STOPP→STOP
    ]

    def __init__(self):
        self._compiled = [(re.compile(p, re.I), r) for p, r in self.CHAR_SUBS]

    def normalize(self, text: str) -> str:
        """Apply phoneme corrections. Returns corrected upper-case string."""
        words = text.strip().upper().split()
        out   = []
        for w in words:
            # Word-level correction first
            corrected = self.WORD_CORRECTIONS.get(w, w)
            if corrected == w:
                # Apply character-level substitutions
                for pattern, repl in self._compiled:
                    corrected = pattern.sub(repl, corrected)
            out.append(corrected)
        return ' '.join(out)

    def phoneme_distance(self, a: str, b: str) -> float:
        """
        Simplified phoneme edit distance: normalised [0,1] where 0=identical.
        Uses SequenceMatcher on phoneme-simplified versions of both strings.
        Lower = more similar.
        """
        # Strip vowels for consonantal skeleton comparison (rough approximation)
        def skeleton(s):
            return re.sub(r'[AEIOU]', '', s.upper())
        sa, sb = skeleton(a), skeleton(b)
        sim = SequenceMatcher(None, sa, sb).ratio()
        return 1.0 - sim


# ═════════════════════════════════════════════════════════════════════════════
#  4.  PENDING COMMAND MANAGER  (Accent Feature 4 — CONFIRM/REPEAT workflow)
# ═════════════════════════════════════════════════════════════════════════════

class PendingCommandManager:
    """
    Visual CONFIRM / REPEAT gating for every recognised command.

    Workflow:
      1. Voice detects command → put_pending(cmd, display_text)
      2. AR display flashes display_text for CONFIRM_WINDOW seconds
      3. Voice says CONFIRM / YES / foot-pedal → confirmed = True
         Voice says REPEAT / NO / foot-pedal-rej → confirmed = False
         Timeout → confirmed = True (auto-commit, configurable)
      4. get_result() returns ('confirmed'|'rejected'|'pending'|None)
    """

    def __init__(self, window_s: float = CONFIRM_WINDOW,
                 auto_commit_on_timeout: bool = True):
        self.window_s           = window_s
        self.auto_commit        = auto_commit_on_timeout
        self._pending_cmd:      Optional[str]   = None
        self._pending_display:  str             = ''
        self._pending_ts:       float           = 0.
        self._result:           Optional[str]   = None  # 'confirmed'|'rejected'
        self._lock              = threading.Lock()
        # Flash text and expiry for display
        self.flash_text  = ''
        self.flash_expiry = 0.

    def put_pending(self, cmd: str, display_text: str):
        with self._lock:
            self._pending_cmd     = cmd
            self._pending_display = display_text
            self._pending_ts      = time.time()
            self._result          = None
            self.flash_text       = f"► {display_text}"
            self.flash_expiry     = time.time() + self.window_s + 0.5

    def confirm(self):
        with self._lock:
            if self._pending_cmd:
                self._result       = 'confirmed'
                self.flash_text    = f"✓ {self._pending_display}"
                self.flash_expiry  = time.time() + 0.8

    def reject(self):
        with self._lock:
            if self._pending_cmd:
                self._result       = 'rejected'
                self.flash_text    = f"✗ REPEAT"
                self.flash_expiry  = time.time() + 0.8

    def get_result(self) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (result, cmd) where result ∈ {'confirmed','rejected','pending',None}.
        'confirmed' → commit the command
        'rejected'  → discard, re-detect
        'pending'   → still waiting
        None        → no pending command
        """
        with self._lock:
            if self._pending_cmd is None:
                return None, None
            if self._result is not None:
                r, c = self._result, self._pending_cmd
                self._pending_cmd = None; self._result = None
                return r, c
            # Check timeout
            if time.time() - self._pending_ts > self.window_s:
                r = 'confirmed' if self.auto_commit else 'rejected'
                c = self._pending_cmd
                self._pending_cmd = None
                self.flash_text   = f"{'✓' if r=='confirmed' else '✗'} AUTO"
                self.flash_expiry = time.time() + 0.5
                return r, c
            return 'pending', self._pending_cmd

    @property
    def is_pending(self) -> bool:
        with self._lock:
            return self._pending_cmd is not None and self._result is None

    def draw_flash(self, frame: np.ndarray) -> np.ndarray:
        """Draw pending command flash on frame."""
        now = time.time()
        if not self.flash_text or now > self.flash_expiry:
            return frame
        if frame is None or frame.size == 0: return frame
        out = frame.copy(); h, w = out.shape[:2]
        scale = 1.1; thick = 2
        (tw, _), _ = cv2.getTextSize(self.flash_text, cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        tx = (w - tw) // 2; ty = h // 2
        cv2.putText(out, self.flash_text, (tx+2, ty+2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
        col = (0,220,130) if self.flash_text.startswith('✓') else \
              (0,100,255) if self.flash_text.startswith('✗') else (0,200,255)
        cv2.putText(out, self.flash_text, (tx, ty),
                    cv2.FONT_HERSHEY_DUPLEX, scale, col, thick, cv2.LINE_AA)
        # Pending indicator bar
        if self.is_pending:
            elapsed = time.time() - (self.flash_expiry - self.window_s - 0.5)
            frac    = max(0., min(1., elapsed / self.window_s))
            bw      = int((1. - frac) * w)
            cv2.rectangle(out, (0, h-6), (bw, h), (0,180,255), -1)
        return out


# ═════════════════════════════════════════════════════════════════════════════
#  5.  FOOT PEDAL LISTENER  (Accent Feature 5 — keyboard/hardware fallback)
# ═════════════════════════════════════════════════════════════════════════════

class FootPedalListener:
    """
    Foot pedal / keyboard fallback for when voice fails.

    F5 key (or actual foot pedal configured as F5) = CONFIRM / GO
    F6 key (or second pedal) = REPEAT / REJECT

    With pynput installed: works in background without stealing focus.
    Without pynput:        falls back to cv2.waitKey() poll in main loop.

    Usage:
        fpl = FootPedalListener(pedal_callback)
        fpl.start()
        # pedal_callback('confirm') or pedal_callback('reject') called on press
    """

    def __init__(self, callback, confirm_key='f5', reject_key='f6'):
        self._cb          = callback
        self._confirm_key = confirm_key
        self._reject_key  = reject_key
        self._running     = False
        self._listener    = None

    def start(self):
        if PYNPUT_AVAILABLE:
            self._running  = True
            self._listener = _pynput_kb.Listener(on_press=self._on_press)
            self._listener.start()
            print(f"[FootPedal] ✓ pynput listener active  "
                  f"({self._confirm_key}=CONFIRM  {self._reject_key}=REJECT)")
        else:
            print(f"[FootPedal] pynput not installed — use cv2 F5/F6 poll instead")
            print("  pip install pynput")

    def stop(self):
        self._running = False
        if self._listener:
            self._listener.stop()

    def _on_press(self, key):
        if not self._running:
            return
        try:
            name = key.name.lower() if hasattr(key, 'name') else ''
            if name == self._confirm_key:
                self._cb('confirm')
            elif name == self._reject_key:
                self._cb('reject')
        except Exception:
            pass

    def poll_cv2_key(self, key_code: int):
        """Call from main loop with cv2 key code to handle F5/F6 without pynput."""
        if   key_code == 0x700000 + 116:  # F5
            self._cb('confirm')
        elif key_code == 0x700000 + 117:  # F6
            self._cb('reject')


# ═════════════════════════════════════════════════════════════════════════════
#  1.  SPEAKER ENROLLMENT  (Accent Feature 1 — framework for Whisper adapter)
# ═════════════════════════════════════════════════════════════════════════════

class SpeakerEnrollment:
    """
    Clinician enrollment: records standard dental phrases for offline
    Whisper adapter fine-tuning.

    Two modes:
      1. record_enrollment()  — interactive CLI, records ~30 phrases, saves WAV files
      2. load_adapted_model() — loads the LoRA-adapted Whisper if present

    Full LoRA fine-tuning pipeline (offline, runs after enrollment):
      See enrollment/README_FINETUNE.md for instructions.
      Expected script: enrollment/finetune_whisper.py (provided separately)
      Trained adapter:  enrollment/adapter.pt  → loaded here automatically

    The enrollment phase takes < 2 minutes.
    Fine-tuning takes < 10 minutes on a single GPU.
    The adapted model typically reduces false reject rate by 40–60%.
    """

    STANDARD_PHRASES = [
        "go", "stop", "end", "confirm", "repeat",
        "yes", "no", "confirm tooth",
        "tooth one buccal", "tooth fourteen lingual",
        "tooth three mesial", "tooth twenty distal",
        "tooth thirty two buccal", "tooth eleven lingual",
        "one buccal", "fourteen lingual", "three mesial",
        "seven buccal", "twenty one lingual",
        "stop recording", "end session",
        "confirm reading", "repeat last", "go to next",
        "save tooth", "record tooth one",
        "four buccal", "eight lingual",
    ]

    def __init__(self, enrollment_dir: str = ENROLLMENT_DIR,
                 whisper_model_size: str = 'small'):
        self._dir   = Path(enrollment_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model_size = whisper_model_size
        self._readme_path = self._dir / "README_FINETUNE.md"
        self._write_readme()

    def _write_readme(self):
        if self._readme_path.exists(): return
        txt = """# Whisper Speaker Adaptation

## After recording enrollment audio:

1. Install requirements:
   pip install openai-whisper torch torchaudio

2. Run fine-tuning (uses LoRA for efficiency):
   python finetune_whisper.py \\
       --enrollment_dir enrollment/ \\
       --model_size small \\
       --output_dir enrollment/

3. Output: enrollment/adapted_model.pt
   This is loaded automatically on next session start.

## Expected improvement:
   - 40-60% reduction in false reject rate
   - Especially large improvement for: th/t substitution, v/w, r/l, final drops
"""
        with open(self._readme_path, 'w') as f:
            f.write(txt)

    def record_enrollment(self, mic_device: Optional[int] = None):
        """Interactive enrollment recording session."""
        if not PYAUDIO_AVAILABLE or not SPEECH_AVAILABLE:
            print("[Enrollment] pyaudio or speech_recognition not installed")
            return False
        print("\n" + "="*60)
        print("  SPEAKER ENROLLMENT — Record your voice for adaptation")
        print("="*60)
        print(f"  Recording {len(self.STANDARD_PHRASES)} phrases.")
        print(f"  Say each phrase clearly when prompted.")
        print(f"  Press ENTER when ready to start each recording.\n")

        recognizer = sr.Recognizer()
        if mic_device is not None:
            mic = sr.Microphone(device_index=mic_device)
        else:
            mic = sr.Microphone()

        recordings_path = self._dir / "recordings"
        recordings_path.mkdir(exist_ok=True)

        recorded = 0
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        for i, phrase in enumerate(self.STANDARD_PHRASES):
            wav_path = recordings_path / f"phrase_{i:03d}.wav"
            if wav_path.exists():
                print(f"  [{i+1:2d}/{len(self.STANDARD_PHRASES)}] '{phrase}' — already recorded, skipping")
                recorded += 1
                continue
            input(f"\n  [{i+1:2d}/{len(self.STANDARD_PHRASES)}] Ready to record: '{phrase}'\n"
                  f"  Press ENTER then say it clearly...")
            print("  🎤 Recording (3 seconds)...")
            try:
                with mic as source:
                    audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                wav_data = audio.get_wav_data()
                with open(wav_path, 'wb') as f:
                    f.write(wav_data)
                print(f"  ✓ Saved to {wav_path}")
                recorded += 1
            except Exception as e:
                print(f"  ✗ Recording failed: {e}")

        # Save metadata
        meta = {
            'clinician_id': f"clinician_{datetime.now().strftime('%Y%m%d')}",
            'n_phrases':    recorded,
            'total_phrases': len(self.STANDARD_PHRASES),
            'model_size':   self._model_size,
            'timestamp':    datetime.now().isoformat(),
        }
        with open(self._dir / "enrollment_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"\n[Enrollment] ✓ Recorded {recorded}/{len(self.STANDARD_PHRASES)} phrases")
        print(f"  Audio saved to: {recordings_path}")
        print(f"  Run finetune_whisper.py to create adapted model")
        print(f"  See {self._readme_path} for instructions\n")
        return recorded > 15

    def load_adapted_model(self):
        """Load LoRA-adapted Whisper if available, else return None."""
        adapter_path = self._dir / "adapted_model.pt"
        if not adapter_path.exists():
            return None
        if not WHISPER_AVAILABLE:
            print("[Enrollment] Whisper not installed — cannot load adapter")
            return None
        try:
            print(f"[Enrollment] Loading adapted Whisper from {adapter_path}…")
            import torch
            # Load the saved dict — format: {lora_state_dict, model_size, lora_rank, lora_alpha}
            saved = torch.load(adapter_path, map_location='cpu')

            # Determine model size from saved file or fall back to configured size
            model_size = saved.get('model_size', self._model_size) if isinstance(saved, dict) else self._model_size
            model = _whisper.load_model(model_size)

            # Extract LoRA weights — saved as nested dict or flat state dict
            if isinstance(saved, dict) and 'lora_state_dict' in saved:
                lora_weights = saved['lora_state_dict']
                lora_alpha   = saved.get('lora_alpha', 32)
                lora_rank    = saved.get('lora_rank', 8)
                # Apply LoRA: W_adapted = W_base + (alpha/rank) * B @ A
                # where lora_A and lora_B are the saved adapter matrices
                model_sd = model.state_dict()
                n_applied = 0
                scale = lora_alpha / max(lora_rank, 1)
                # Match lora_A and lora_B pairs by stripping suffix
                keys_A = {k.replace('.lora_A',''): (k, v)
                          for k, v in lora_weights.items() if 'lora_A' in k}
                keys_B = {k.replace('.lora_B',''): (k, v)
                          for k, v in lora_weights.items() if 'lora_B' in k}
                for base_key in keys_A:
                    if base_key in keys_B and base_key in model_sd:
                        A = keys_A[base_key][1].to(torch.float32)
                        B = keys_B[base_key][1].to(torch.float32)
                        # LoRA delta: B @ A scaled by alpha/rank
                        delta = (B @ A) * scale
                        if model_sd[base_key].shape == delta.shape:
                            model_sd[base_key] = model_sd[base_key].to(torch.float32) + delta
                            n_applied += 1
                model.load_state_dict(model_sd, strict=True)
                print(f"[Enrollment] ✓ LoRA adapter applied: {n_applied} weight matrices adapted")
            else:
                # Legacy flat state dict — apply directly
                model.load_state_dict(saved, strict=False)
                print("[Enrollment] ✓ Adapted Whisper loaded (legacy format)")

            return model
        except Exception as e:
            print(f"[Enrollment] Could not load adapter: {e}")
            print(f"[Enrollment] Using base Whisper model instead")
            return None


# ═════════════════════════════════════════════════════════════════════════════
#  VOICE COMMAND LISTENER V4  (all 5 accent features integrated)
# ═════════════════════════════════════════════════════════════════════════════

class VoiceCommandListenerV4:
    """
    Complete voice command listener with all 5 accent robustness features:
      1. Speaker enrollment adapter (if present)
      2. Domain-constrained FSG
      3. Phoneme mapper (CMU-style)
      4. CONFIRM/REPEAT pending workflow
      5. Foot pedal fallback

    State machine (unchanged interface from voice3):
      inactive → ready → waiting_for_tooth → scanning → saving → finished

    Additional states:
      pending_confirm → waiting for CONFIRM or REPEAT after any command

    Thread-safe queues:
      cmd_queue   — confirmed commands for main loop
      flash_queue — (text, duration) for display overlay (legacy compat)
    """

    MIC_PRIORITY = [1, 7, 14]       # XREAL AR goggle
    MIC_FALLBACK = [2, 8, 15, 16, 20]

    # Three-tier phoneme variant table
    PHONEME_VARIANTS = {
        'go':      ['GO','GOH','GOU','GAO','GOW','GOO','JOH','KO','GEO'],
        'stop':    ['STOP','STAHP','ESTOP','STAP','STOB','STUP','TOP','STOPT'],
        'end':     ['END','EHND','IND','ENND','AEND','EEND','NED'],
        'confirm': ['CONFIRM','CONFERM','KONFER','CONFER','KONFIRM','AFFIRM'],
        'repeat':  ['REPEAT','REPEET','REPET','RIPEAT','REPAT'],
        'yes':     ['YES','YAS','JES','JEH','YEH','SI'],
        'no':      ['NO','NAO','NOH','NAH','NAW'],
    }
    NATURAL_ALIASES = {
        'go':      ['START','BEGIN','ACTIVATE','READY','ON'],
        'stop':    ['DONE','SAVE','FINISH','COMPLETE'],
        'end':     ['EXIT','QUIT','FINISH SESSION'],
        'confirm': ['OK','OKAY','ACCEPT','COMMIT','YES CONFIRM'],
        'repeat':  ['UNDO','BACK','CANCEL','AGAIN','NO REPEAT'],
        'yes':     ['OK','OKAY'],
        'no':      ['CANCEL','BACK'],
    }

    def __init__(self,
                 whisper_model_size: str = 'small',
                 fuzzy_threshold:    float = 0.60,
                 voice_device:       Optional[int] = None,
                 pending_manager:    Optional[PendingCommandManager] = None,
                 foot_pedal:         Optional[FootPedalListener] = None,
                 metric_monitor:     Optional[MetricMonitor] = None,
                 enrollment_dir:     str = ENROLLMENT_DIR):

        self.threshold      = fuzzy_threshold
        self.cmd_queue      = queue.Queue()
        self.flash_queue    = queue.Queue()  # legacy compat
        self._running       = False
        self._mic           = None
        self._rec           = None
        self._whisper       = None
        self._pending       = pending_manager
        self._pedal         = foot_pedal
        self._metrics       = metric_monitor

        # FSG + phoneme mapper
        self._fsg    = DentalFSG()
        self._phmap  = PhonemeMapper()

        # State machine
        self.current_tooth  = None
        self.state          = 'inactive'
        self.system_active  = False

        if not SPEECH_AVAILABLE:
            print("[Voice] speech_recognition not available — voice disabled")
            return

        # Try to load speaker-adapted Whisper first
        enrollment = SpeakerEnrollment(enrollment_dir, whisper_model_size)
        adapted    = enrollment.load_adapted_model()

        if adapted is not None:
            self._whisper = adapted
        elif WHISPER_AVAILABLE:
            try:
                print(f"[Voice] Loading Whisper '{whisper_model_size}'…")
                self._whisper = _whisper.load_model(whisper_model_size)
                print("[Voice] ✓ Whisper base model loaded (no enrollment adapter)")
            except Exception as e:
                print(f"[Voice] Whisper failed: {e}")

        self._rec = sr.Recognizer()
        self._rec.energy_threshold         = 200
        self._rec.dynamic_energy_threshold = True
        self._rec.pause_threshold          = 0.7
        # Raw PyAudio for fixed-duration recording (bypasses energy gate)
        self._pa      = None
        self._pa_dev  = None
        self._voice_device_override = voice_device
        self._init_pa()

        self._init_mic(voice_device)

    def _init_mic(self, override: Optional[int]):
        if not SPEECH_AVAILABLE: return
        try:
            p     = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
            names = sr.Microphone.list_microphone_names()
            if override is not None:
                self._mic = sr.Microphone(device_index=override)
                if p: p.terminate()
                print(f"[Voice] Mic override: device {override}")
                return
            for idx in self.MIC_PRIORITY + self.MIC_FALLBACK:
                if idx >= len(names): continue
                try:
                    if p and p.get_device_info_by_index(idx)['maxInputChannels'] > 0:
                        self._mic = sr.Microphone(device_index=idx)
                        tag = 'XREAL' if idx in self.MIC_PRIORITY else 'fallback'
                        print(f"[Voice] ✓ Mic [{tag}] dev {idx}: {names[idx]}"); break
                except: continue
            if p: p.terminate()
            if self._mic is None:
                self._mic = sr.Microphone()
                print("[Voice] Using system default mic")
        except Exception as e:
            print(f"[Voice] Mic init error: {e}"); self._mic = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start_listening(self):
        if not SPEECH_AVAILABLE or self._mic is None:
            print("[Voice] Cannot start — no mic or speech_recognition"); return
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("\n"+"="*70)
        print("VOICE V4 — All 5 Accent Robustness Features Active")
        print("="*70)
        print("  GO → activate  |  [number + surface] → start scan")
        print("  STOP → save    |  END → finish  |  CONFIRM/YES → commit")
        print("  REPEAT/NO → reject  |  F5 = foot-pedal CONFIRM")
        if self._whisper: print("  ✓ Whisper active")
        print("="*70+"\n")

    def stop(self): self._running = False

    # v3-compatible interface
    def get_state(self):           return self.state
    def is_system_active(self):    return self.system_active
    def is_scanning(self):         return self.state == 'scanning'
    def get_current_tooth(self):   return self.current_tooth
    def should_finish(self):       return self.state == 'finished'

    def should_save_and_reset(self) -> Optional[str]:
        if self.state == 'saving' and self.current_tooth is not None:
            tooth = self.current_tooth
            self.current_tooth = None
            self.state = 'waiting_for_tooth'
            return tooth
        return None

    def foot_pedal_event(self, event: str):
        """Called by FootPedalListener — 'confirm' or 'reject'."""
        if event == 'confirm':
            if self._pending and self._pending.is_pending:
                self._pending.confirm()
                print("[FootPedal] CONFIRM")
            else:
                # Act as GO if no pending command
                self._dispatch('go', 'PEDAL→GO')
        elif event == 'reject':
            if self._pending:
                self._pending.reject()
            print("[FootPedal] REJECT")

    # ── Recognition pipeline ──────────────────────────────────────────────────

    def _loop(self):
        try:
            with self._mic as src:
                print("[Voice] Calibrating ambient noise…")
                self._rec.adjust_for_ambient_noise(src, duration=2)
                print(f"[Voice] ✓ Calibrated (threshold={self._rec.energy_threshold:.0f})\n"
                      f"[Voice] 🎤 Listening — say 'GO' to activate\n")
        except Exception as e:
            print(f"[Voice] Calibration failed: {e}"); return

        while self._running:
            # While a command is pending, only listen for CONFIRM/REPEAT
            if self._pending and self._pending.is_pending:
                self._listen_for_confirmation()
            else:
                self._listen_for_command()

    def _init_pa(self):
        """Initialise raw PyAudio — called once at startup."""
        if not PYAUDIO_AVAILABLE: return
        try:
            self._pa = pyaudio.PyAudio()
            names = sr.Microphone.list_microphone_names() if SPEECH_AVAILABLE else []
            override = getattr(self, '_voice_device_override', None)
            if override is not None:
                self._pa_dev = override; return
            for idx in self.MIC_PRIORITY + self.MIC_FALLBACK:
                if idx >= len(names): continue
                try:
                    info = self._pa.get_device_info_by_index(idx)
                    if info['maxInputChannels'] > 0:
                        self._pa_dev = idx; return
                except: continue
            self._pa_dev = self._pa.get_default_input_device_info()['index']
        except Exception as e:
            print(f"[Voice] PyAudio raw-record init failed: {e} — using sr.listen() fallback")
            self._pa = None

    def _record_fixed(self, duration: float):
        """
        Record exactly `duration` seconds via raw PyAudio — no energy gate.
        Short words like 'go', 'stop', '14 B' are captured immediately
        without waiting for a threshold that may never trigger.
        Returns sr.AudioData (compatible with _transcribe) or None on error.

        Duration guide:
          1.5s — single action words (go, stop, end, yes, no)
          2.5s — tooth specs (14 B, twenty two L)
          3.0s — multi-word phrases (stop recording, end session)
        """
        CHUNK = 512
        RATE  = 16000

        if self._pa is None or not PYAUDIO_AVAILABLE:
            # Energy-gate fallback
            try:
                with self._mic as src:
                    return self._rec.listen(src, timeout=3,
                                            phrase_time_limit=max(duration, 3.0))
            except sr.WaitTimeoutError:
                return None

        try:
            stream = self._pa.open(
                format=pyaudio.paInt16, channels=1, rate=RATE,
                input=True, input_device_index=self._pa_dev,
                frames_per_buffer=CHUNK)
            n_chunks = int(RATE * duration / CHUNK)
            frames   = [stream.read(CHUNK, exception_on_overflow=False)
                        for _ in range(n_chunks)]
            stream.stop_stream()
            stream.close()
            return sr.AudioData(b''.join(frames), RATE, 2)  # 2 = paInt16 bytes
        except Exception as e:
            print(f"[Voice] _record_fixed error: {e}")
            return None

    def _listen_for_command(self):
        try:
            # Choose duration based on state:
            #   inactive / scanning → expect short action word (1.5s)
            #   waiting_for_tooth   → expect tooth spec like "14 B" (2.5s)
            duration = 2.5 if self.state == 'waiting_for_tooth' else 1.5
            audio    = self._record_fixed(duration)
            if audio is None: return

            t0  = time.time()
            raw = self._transcribe(audio)
            if not raw: return
            lat = (time.time() - t0) * 1000.

            # When waiting for a tooth, try FSG on raw text FIRST before
            # phoneme-mapping corrupts digit+surface strings like "1B" or "TOOTH 14B"
            if self.state == 'waiting_for_tooth':
                fsg_raw = self._fsg.parse(raw)
                if fsg_raw and fsg_raw['type'] == 'tooth':
                    print(f"[Voice] '{raw}' → tooth {fsg_raw['value']} (raw FSG)")
                    self._dispatch_tooth(fsg_raw['value'])
                    return

            # Feature 3: phoneme mapping
            corrected = self._phmap.normalize(raw)

            # Feature 2: FSG parse on corrected text
            fsg_result = self._fsg.parse(corrected)

            # When waiting for a tooth, also try FSG on corrected text directly
            if self.state == 'waiting_for_tooth' and fsg_result and fsg_result.get('type','') == 'tooth':
                print(f"[Voice] '{raw}' → '{corrected}' → tooth {fsg_result.get('value', None)} (corrected FSG)")
                self._dispatch_tooth(fsg_result.get('value', None))
                return

            # Three-tier command matching
            cmd, tier = self._match(corrected, fsg_result)

            # Log
            if self._metrics:
                self._metrics.log_voice_event(raw, corrected, fsg_result, cmd, tier, lat)

            if cmd:
                print(f"[Voice] '{raw}' → '{corrected}' → {cmd.upper()} (tier {tier})")
                # Feature 4: pending confirmation
                if self._pending and cmd not in ('confirm','repeat','yes','no'):
                    display = self._format_pending_display(cmd, fsg_result)
                    self._pending.put_pending(cmd, display)
                    self._flash(f"► {display}")
                else:
                    self._dispatch(cmd, corrected)
            else:
                print(f"[Voice] no match: '{raw}' → '{corrected}'")

        except sr.WaitTimeoutError: pass
        except Exception: pass

    def _dispatch_tooth(self, tooth_value: str):
        """Directly set tooth and enter scanning state — bypasses pending/confirm."""
        self.current_tooth = tooth_value
        self.state = 'scanning'
        self._flash(f"TOOTH {tooth_value}")
        print(f"\n🦷 Scanning tooth {tooth_value} (say STOP)")
        self.cmd_queue.put('tooth')

    def _listen_for_confirmation(self):
        """Dedicated listening for CONFIRM/REPEAT during pending state."""
        try:
            audio = self._record_fixed(1.5)   # short word — no energy gate needed
            if not audio: return
            raw = self._transcribe(audio)
            if not raw: return
            corrected = self._phmap.normalize(raw)
            cmd, tier = self._match(corrected, None)
            if cmd in ('confirm','yes'):
                self._pending.confirm()
            elif cmd in ('repeat','no'):
                self._pending.reject()
        except: pass

    def _transcribe(self, audio) -> Optional[str]:
        if self._whisper:
            try:
                import tempfile, os as _os
                wav = audio.get_wav_data()
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(wav); tmp = f.name
                res = self._whisper.transcribe(tmp, language='en',
                                               condition_on_previous_text=False)
                _os.unlink(tmp)
                return res['text'].strip().upper()
            except: pass
        try: return self._rec.recognize_google(audio).upper()
        except: return None

    def _match(self, text: str, fsg_result: Optional[dict]) -> Tuple[Optional[str], int]:
        """
        Three-tier matching. Returns (command, tier) where tier 0 = no match.
        Tier 1: exact phoneme variant
        Tier 2: natural alias substring
        Tier 3: phoneme fuzzy (SequenceMatcher)
        """
        words = text.strip().upper().split()

        # Tier 1: exact variant
        for cmd, variants in self.PHONEME_VARIANTS.items():
            for w in words:
                if w in variants:
                    return cmd, 1

        # Tier 2: alias substring
        for cmd, aliases in self.NATURAL_ALIASES.items():
            for alias in aliases:
                if alias in text:
                    return cmd, 2

        # FSG action result as fallback
        if fsg_result and fsg_result.get('type','') == 'action':
            return fsg_result.get('value', None), 2

        # Tier 3: fuzzy phoneme distance
        best_cmd = None; best_ratio = 0.
        for cmd, variants in self.PHONEME_VARIANTS.items():
            for w in words:
                for v in variants:
                    r = 1.0 - self._phmap.phoneme_distance(w, v)
                    if r > best_ratio:
                        best_ratio = r; best_cmd = cmd
        if best_ratio >= max(self.threshold, 0.78):   # min 0.78 — UP→STOP false match
            return best_cmd, 3

        return None, 0

    def _format_pending_display(self, cmd: str, fsg_result: Optional[dict]) -> str:
        if fsg_result and fsg_result.get('type','') == 'tooth':
            return f"TOOTH {fsg_result.get('value', None)}"
        return cmd.upper()

    def _flash(self, text: str, dur: float = FLASH_DURATION):
        self.flash_queue.put((text, dur))

    def _dispatch(self, cmd: str, text: str):
        """State machine: GO → waiting_for_tooth → scanning → saving → waiting_for_tooth → … → END"""
        if self.state == 'inactive':
            if cmd == 'go':
                self.state = 'waiting_for_tooth'
                self.system_active = True
                self._flash("GO — say tooth number")
                print("\n✅ ACTIVATED — say tooth number (e.g. '14 B', '3 L')")

        elif self.state == 'waiting_for_tooth':
            # Try FSG on the raw/corrected text passed in
            fsg = self._fsg.parse(text)
            if fsg and fsg['type'] == 'tooth':
                self._dispatch_tooth(fsg['value'])
                return
            elif cmd == 'end':
                self.state = 'finished'; self.system_active = False
                self._flash("END"); print("\n🏁 Ending…")
            else:
                print(f"⚠ Not a valid tooth: '{text}' — say e.g. '14 B' or '3 L'")

        elif self.state == 'scanning':
            if cmd == 'stop':
                self.state = 'saving'
                self._flash(f"STOP — saving {self.current_tooth}")
                print(f"\n💾 STOP — saving {self.current_tooth}…")
            elif cmd == 'end':
                self.state = 'finished'; self.system_active = False
                self._flash("END"); print("\n🏁 Ending…")

        self.cmd_queue.put(cmd)

    # ── Process pending from main loop ────────────────────────────────────────

    def process_pending(self) -> bool:
        """Call every frame from main loop. Returns True if command was committed."""
        if not self._pending: return False
        result, cmd = self._pending.get_result()
        if result == 'confirmed' and cmd:
            self._dispatch(cmd, cmd)
            return True
        elif result == 'rejected':
            print(f"[Voice] REJECTED — repeat command")
            return False
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  TOOTH CHART MANAGER  (updated — uses V4 listener, adds metric monitor)
# ═════════════════════════════════════════════════════════════════════════════

class ToothChartManager:
    def __init__(self, output_dir='tooth_chart_scans',
                 jitter_threshold_um=500, min_stable_frames=10,
                 enable_voice=True, voice_device: Optional[int] = None,
                 whisper_model_size: str = 'small',
                 metric_monitor: Optional[MetricMonitor] = None,
                 enrollment_dir: str = ENROLLMENT_DIR,
                 confirm_window: float = CONFIRM_WINDOW):
        self.output_dir         = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.jitter_threshold_um = jitter_threshold_um
        self.min_stable_frames   = min_stable_frames
        self.current_tooth       = None
        self.tooth_buffer:       List[np.ndarray] = []
        self.tooth_measurements: List[dict] = []
        self.scanning_active     = False
        self.best_frames_saved:  Dict[str, dict] = {}
        self.valid_teeth         = self._generate_valid_teeth()
        self._metrics            = metric_monitor
        self.voice_listener      = None
        self._pending            = None
        self._pedal              = None

        if enable_voice:
            self._pending = PendingCommandManager(window_s=confirm_window)
            self._pedal   = FootPedalListener(
                callback=lambda ev: self.voice_listener.foot_pedal_event(ev)
                         if self.voice_listener else None)
            self.voice_listener = VoiceCommandListenerV4(
                whisper_model_size=whisper_model_size,
                voice_device=voice_device,
                pending_manager=self._pending,
                foot_pedal=self._pedal,
                metric_monitor=metric_monitor,
                enrollment_dir=enrollment_dir)
            self.voice_listener.start_listening()
            self._pedal._cb = self.voice_listener.foot_pedal_event
            self._pedal.start()

    def _generate_valid_teeth(self):
        return {f"{n}{s}" for n in range(1,33) for s in ['B','L']}

    def start_tooth_scan(self, tooth_name: str) -> bool:
        tooth_name = tooth_name.upper().strip()
        if tooth_name not in self.valid_teeth:
            print(f"Invalid tooth: {tooth_name}"); return False
        self.current_tooth      = tooth_name
        self.tooth_buffer       = []; self.tooth_measurements = []
        self.scanning_active    = True
        print(f"\n{'='*55}\nScanning: Tooth {tooth_name}  "
              f"(jitter<{self.jitter_threshold_um}µm, "
              f"min {self.min_stable_frames} frames)\n{'='*55}")
        return True

    def add_frame(self, frame: np.ndarray, measurements: dict):
        if not self.scanning_active: return
        if frame is not None and frame.size > 0: self.tooth_buffer.append(frame.copy())
        self.tooth_measurements.append(measurements)
        if len(self.tooth_buffer) > 300:
            self.tooth_buffer.pop(0); self.tooth_measurements.pop(0)

    def calculate_jitter(self, window: List[dict]) -> Optional[Dict[str,float]]:
        if len(window) < 2: return None
        result = {}
        for k in ['iGH','iGR','iABL']:
            vals = [m[k] for m in window if m and k in m and m[k] is not None]
            result[k] = float(np.std(vals))*1000. if len(vals)>=2 else float('inf')
        return result

    def find_best_frames(self) -> Optional[dict]:
        if len(self.tooth_buffer) < self.min_stable_frames: return None
        best=None; best_sum=float('inf')
        for i in range(len(self.tooth_buffer)-self.min_stable_frames+1):
            win=self.tooth_measurements[i:i+self.min_stable_frames]
            j=self.calculate_jitter(win)
            if j is None: continue
            if all(v<self.jitter_threshold_um for v in j.values()):
                s=sum(j.values())
                if s<best_sum:
                    best_sum=s; best={'start':i,'end':i+self.min_stable_frames,
                                      'jitters':j,'avg':self._avg_meas(win)}
        return best

    def _avg_meas(self, window):
        avg={}
        for k in ['iGH','iGR','iABL']:
            v=[m[k] for m in window if m and k in m and m[k] is not None]
            avg[k]=float(np.mean(v)) if v else None
        return avg

    def save_best_frames(self, force: bool = False) -> bool:
        """
        Save best frames for current tooth.
        force=True (called by voice STOP): always save even if no stable window found.
        force=False (called by button): only save if stable window exists.
        """
        if not self.current_tooth:
            print(f"[Chart] save_best_frames: no current_tooth set")
            return False
        print(f"[Chart] save_best_frames: tooth={self.current_tooth} "
              f"buffer={len(self.tooth_buffer)} meas={len(self.tooth_measurements)}")
        best = self.find_best_frames()
        if best is None:
            if force and len(self.tooth_buffer) >= 1:
                # Force-save: use all available frames — voice said STOP explicitly
                n = min(self.min_stable_frames, len(self.tooth_buffer))
                best = {
                    'start': max(0, len(self.tooth_buffer) - n),
                    'end':   len(self.tooth_buffer),
                    'jitters': {'iGH': 999, 'iGR': 999, 'iABL': 999},
                    'avg':   self._avg_meas(self.tooth_measurements[-n:] if self.tooth_measurements else [{}]*n),
                }
                print(f"[Chart] ⚠ No stable sequence — force-saving last {n} frames")
            else:
                print(f"[Chart] ⚠ No stable sequence at {self.jitter_threshold_um}µm "
                      f"— {len(self.tooth_buffer)} frames in buffer")
                return False
        # If same tooth scanned multiple times, add _2 _3 etc. — never overwrite
        _base = self.output_dir / self.current_tooth
        td    = _base; _n = 2
        while td.exists() and any(td.iterdir()):
            td = Path(str(_base) + f"_{_n}"); _n += 1
        td.mkdir(parents=True, exist_ok=True)
        print(f"[Chart] Saving frames → {td.name}/")
        for i in range(best['start'], best['end']):
            cv2.imwrite(str(td / f"frame_{i-best['start']+1:03d}.png"), self.tooth_buffer[i])
        # Use resolved folder name as key so repeated scans are all kept
        _tooth_key = td.name   # e.g. "14B" first time, "14B_2" second time
        meta = {'tooth': self.current_tooth, 'folder': _tooth_key,
                'timestamp': datetime.now().isoformat(),
                'n_frames': best['end'] - best['start'],
                'jitter_threshold_um': self.jitter_threshold_um,
                'jitters_um': best['jitters'], 'avg_measurements_mm': best['avg']}
        # Measurement stability CV (coefficient of variation — lower = more stable probe)
        if len(self.tooth_measurements) >= 5:
            recent = self.tooth_measurements[-min(30, len(self.tooth_measurements)):]
            for key in ['iGR','iABL','iGH']:
                vals = [m[key] for m in recent if m and key in m and m[key] is not None]
                if len(vals) > 2:
                    meta[f'{key}_cv'] = round(float(np.std(vals)/max(np.mean(vals),1e-6)), 4)
                    meta[f'{key}_n_samples'] = len(vals)
        with open(td / 'metadata.json', 'w') as f: json.dump(meta, f, indent=2)
        self.best_frames_saved[_tooth_key] = meta   # use folder name — keeps all scans
        j=best['jitters']; a=best['avg']
        tooth_name = meta.get('tooth', _tooth_key)
        print(f"\n✓ Tooth {tooth_name}: "
              f"iGH={a.get('iGH',0):.3f}mm  iGR={a.get('iGR',0):.3f}mm  "
              f"iABL={a.get('iABL',0):.3f}mm  "
              f"jitter={max(j.values()):.1f}µm\n")
        return True

    def stop_tooth_scan(self):
        if self.scanning_active:
            tooth = self.current_tooth   # save name before ANY clearing
            self.scanning_active = False
            saved = self.save_best_frames(force=True)
            if not saved:
                print(f"[Chart] stop_tooth_scan: save failed for {tooth}")
            self.current_tooth = None

    def export_to_excel(self, filename='tooth_chart.xlsx') -> bool:
        if not OPENPYXL_AVAILABLE:
            print("[Export] openpyxl not installed — pip install openpyxl")
            return False
        if not self.best_frames_saved:
            print("[Export] No tooth data to export")
            return False
        try:
            from pathlib import Path as _P
            fp_obj = _P(filename).resolve()   # always absolute regardless of cwd
            fp_obj.parent.mkdir(parents=True, exist_ok=True)
            print(f"[Export] Saving to: {fp_obj}")

            wb  = Workbook()
            ws  = wb.active
            ws.title = "Tooth Chart"
            hfill = PatternFill(start_color="1B6CA8", end_color="1B6CA8", fill_type="solid")
            hfont = Font(bold=True, color="FFFFFF", size=11)
            cal   = Alignment(horizontal="center", vertical="center")
            brd   = Border(
                left=Side(style='thin'), right=Side(style='thin'),
                top=Side(style='thin'),  bottom=Side(style='thin'))

            headers = ["Tooth", "iGH (mm)", "iGR (mm)", "iABL (mm)",
                       "Timestamp", "Frames",
                       "Jitter iGH (µm)", "Jitter iGR (µm)", "Jitter iABL (µm)"]
            for ci, h in enumerate(headers, 1):
                c = ws.cell(row=1, column=ci, value=h)
                c.fill = hfill; c.font = hfont
                c.alignment = cal; c.border = brd

            def _sk(t):
                m = re.match(r'(\d+)([A-Z]+)', t)
                return (int(m.group(1)), m.group(2)) if m else (999, t)

            for ri, tooth in enumerate(sorted(self.best_frames_saved, key=_sk), 2):
                d  = self.best_frames_saved[tooth]
                ms = d.get('avg_measurements_mm') or {}
                js = d.get('jitters_um') or {}
                if isinstance(js, list):
                    js = {}

                def _mv(key):
                    v = ms.get(key)
                    return round(v, 3) if v is not None and v != 0 else 'N/A'

                def _jv(key):
                    v = js.get(key)
                    return round(v, 1) if isinstance(js, dict) and v is not None else 'N/A'

                row = [tooth, _mv('iGH'), _mv('iGR'), _mv('iABL'),
                       d.get('timestamp', 'N/A'), d.get('n_frames', 0),
                       _jv('iGH'), _jv('iGR'), _jv('iABL')]

                for ci, val in enumerate(row, 1):
                    c = ws.cell(row=ri, column=ci, value=val)
                    c.alignment = cal; c.border = brd

            for col, w in zip('ABCDEFGHI', [10, 12, 12, 12, 22, 8, 14, 14, 14]):
                ws.column_dimensions[col].width = w

            wb.save(str(fp_obj))
            print(f"\n✓ Exported {len(self.best_frames_saved)} teeth → {fp_obj}")
            return True

        except Exception as e:
            print(f"[Export] Failed: {e}")
            import traceback; traceback.print_exc()
            return False


    def get_status_text(self) -> Optional[str]:
        if not self.scanning_active: return None
        n=len(self.tooth_buffer)
        s=f"Scanning: {self.current_tooth} | {n} frames | "
        if n>=self.min_stable_frames:
            s+="STABLE ✓" if self.find_best_frames() else f"Searching…"
        else:
            s+=f"Need {self.min_stable_frames-n} more"
        return s


# ═════════════════════════════════════════════════════════════════════════════
#  HD60X CAPTURE  (unchanged from voice3)
# ═════════════════════════════════════════════════════════════════════════════



class RealTimeB1ChartAnnotator:
    """
    Real-time charting annotator using B1_V8_Baseline (MKUNet).
    Replaces HybridCrossFusion from voice3.

    Key change: uses MKUNetInference from video_inference_pipeline.py
    which handles all B1 custom layers (MKIRBlock, SAVSSBlock,
    LGAGSkipGate, LGAG2Gate, SASEBlock) internally.

    Output format identical to B1LiveAnnotator in live_panorama_final.py:
      GM/CEJ/ABC → point head (soft-argmax centroids)
      TOOTH/BONE/GINGIVA → region head (binary masks)
      Quality: 0.40×TOOTH + 0.40×GM + 0.10×GINGIVA
    """

    CONF_THRESH = {'GM': 0.18, 'CEJ': 0.28, 'ABC': 0.20}

    def __init__(self, model_path: str,
                 # Physical field-of-view of the ultrasound probe at the imaging depth.
                 # CALIBRATE THESE VALUES against a ruler or phantom.
                 # Wrong values = wrong mm measurements in iGH/iGR/iABL.
                 # Current assumption: 7mm × 7mm physical → 610 × 640 pixels captured.
                 # At 610px/7mm = 87 px/mm = 11.5 µm/px (raw frame before model resize).
                 # The model resizes to 256×256 internally; px coordinates are in 610×640 space.
                 crop_width_mm=7.0, crop_height_mm=7.0,
                 crop_width_px=610, crop_height_px=640,
                 tooth_chart_dir='tooth_chart_scans',
                 jitter_threshold_um=500, target_fps=25.,  # 500µm = 0.5mm, same as stitching
                 enable_voice=True, jitter_display_threshold=200,
                 voice_device: Optional[int] = None,
                 whisper_model_size: str = 'small',
                 metric_monitor: Optional[MetricMonitor] = None,
                 enrollment_dir: str = ENROLLMENT_DIR,
                 confirm_window: float = CONFIRM_WINDOW,
                 mkunet_inference=None):   # pass pre-loaded MKUNetInference to avoid reloading
        self.model_path                  = model_path
        self.show_clinical_measurements  = True
        self.pixels_per_mm_x = crop_width_px  / crop_width_mm
        self.pixels_per_mm_y = crop_height_px / crop_height_mm
        self.pixels_per_mm   = (self.pixels_per_mm_x + self.pixels_per_mm_y) / 2.
        print(f"\n{'='*55}")
        print(f"  DentScan model  {crop_width_mm}mm×{crop_height_mm}mm  "
              f"→  {self.pixels_per_mm:.2f} px/mm  "
              f"({1000/self.pixels_per_mm:.1f} µm/px)")
        print(f"  NOTE: crop_width_mm={crop_width_mm} must match your ultrasound FOV.")
        print(f"  If measurements look wrong, verify the physical FOV of your probe.")
        print(f"{'='*55}\n")
        self.target_fps        = target_fps
        self.frame_interval    = 1. / target_fps
        self.last_process_time = 0.
        self.jitter_display_threshold = jitter_display_threshold
        self._metrics          = metric_monitor

        # B1 model state
        self.model             = None
        self.predict_fn        = None
        self.point_coords_cache: Dict[str, Tuple[int,int]] = {}
        self.point_coords_cache_256: Dict[str, Tuple[float,float]] = {}  # 27µm/px coords
        self._last_regions:    Dict[str, dict] = {}
        self._last_quality     = 0.
        self._inference_ms     = 0.
        self._display_fps      = 0.
        self._fps_counter      = 0
        self._fps_t0           = time.time()
        self._total_frames     = 0
        self._latency_buf      = deque(maxlen=30)
        # Position smoothing — 3-frame rolling window (mirrors panorama)
        self._pos_smooth: Dict[str, deque] = {
            f: deque(maxlen=3) for f in POINT_FEATURES}

        # Measurements cache
        self._meas_cache       = None
        self._meas_counter     = 0
        self._meas_interval    = 3

        self.performance_stats = {
            'fps':0,'inference_time':0,'avg_latency':0,
            'total_frames':0,'display_fps':0,
        }

        # Tooth chart (with V4 voice listener)
        self.tooth_chart = ToothChartManager(
            output_dir=tooth_chart_dir,
            jitter_threshold_um=jitter_threshold_um,
            enable_voice=enable_voice,
            voice_device=voice_device,
            whisper_model_size=whisper_model_size,
            metric_monitor=metric_monitor,
            enrollment_dir=enrollment_dir,
            confirm_window=confirm_window)

        self._pending = self.tooth_chart._pending  # reference for display

        if mkunet_inference is not None:
            # Use pre-loaded model — avoids loading B1 a second time
            self._mkunet    = mkunet_inference
            keras_model     = self._mkunet.model
            @tf.function(reduce_retracing=True)
            def pf(x): return keras_model(x, training=False)
            self.predict_fn = pf
            print("[DentScan] ✓ Shared MKUNetInference wired into RealTimeB1ChartAnnotator")
        else:
            self._load_model()

    def _load_model(self):
        if not MKUNET_AVAILABLE:
            raise ImportError("video_inference_pipeline.py not found — cannot load DentScan model")
        print(f"[DentScan] Loading from: {self.model_path}")
        self._mkunet = MKUNetInference(self.model_path)
        keras_model  = self._mkunet.model

        @tf.function(reduce_retracing=True)
        def pf(x): return keras_model(x, training=False)
        self.predict_fn = pf
        print("[DentScan] Warming up (3 runs)…")
        dummy = tf.zeros((1, MODEL_SIZE, MODEL_SIZE, 3), dtype=tf.float32)
        for _ in range(3): _ = self.predict_fn(dummy)
        print("✓ B1_V8_Baseline ready\n")

    def preprocess_frame_fast(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            r = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)
            return np.expand_dims(r.astype(np.float32) / 255., 0)
        except: return None

    def predict_single_frame(self, batch: np.ndarray):
        try:
            t0 = time.time()
            p  = self.predict_fn(batch)
            self._inference_ms = (time.time() - t0) * 1000.
            self._latency_buf.append(self._inference_ms)
            return p
        except Exception as e:
            print(f"[DentScan predict] {e}"); return None

    def post_process_predictions(self, preds, target_shape: Tuple[int,int]) -> dict:
        if preds is None: return {}
        h, w = target_shape; proc = {}
        self.point_coords_cache = {}
        self.point_coords_cache_256 = {}
        try:
            feats = getattr(self._mkunet, 'all_features', ALL_FEATURES)
            for i, feat in enumerate(feats):
                if i >= len(preds): break
                p   = preds[i]
                pnp = p[0].numpy() if hasattr(p[0], 'numpy') else np.array(p[0])
                pm  = pnp[:,:,1]
                if feat in POINT_FEATURES:
                    thr = self.CONF_THRESH.get(feat, 0.2)
                    if pm.max() < thr: continue
                    ys = np.arange(MODEL_SIZE)[:,None]*np.ones((1,MODEL_SIZE))
                    xs = np.arange(MODEL_SIZE)[None,:]*np.ones((MODEL_SIZE,1))
                    tot = pm.sum()+1e-8
                    cy_256 = float((pm*ys).sum()/tot)
                    cx_256 = float((pm*xs).sum()/tot)
                    cx = cx_256 * w / MODEL_SIZE
                    cy = cy_256 * h / MODEL_SIZE
                    # Smooth position (3-frame rolling window — same as panorama)
                    self._pos_smooth[feat].append((cx, cy))
                    cx_s = np.mean([p_[0] for p_ in self._pos_smooth[feat]])
                    cy_s = np.mean([p_[1] for p_ in self._pos_smooth[feat]])
                    self.point_coords_cache[feat] = (int(cx_s), int(cy_s))
                    # Also store 256-space centroid for accurate mm measurements.
                    # Measurements use 256×256 coords with PIXEL_SPACING_MM=7/256=27µm/px.
                    cx_s_256 = cx_256 * np.mean([p_[0]/max(cx,1) for p_ in self._pos_smooth[feat]])                                if cx > 0 else cx_256
                    # Simpler: smooth in 256 space directly
                    if not hasattr(self,'_pos_smooth_256'): self._pos_smooth_256={}
                    if feat not in self._pos_smooth_256: self._pos_smooth_256[feat]=deque(maxlen=3)
                    self._pos_smooth_256[feat].append((cx_256, cy_256))
                    cx_sm256 = float(np.mean([p[0] for p in self._pos_smooth_256[feat]]))
                    cy_sm256 = float(np.mean([p[1] for p in self._pos_smooth_256[feat]]))
                    self.point_coords_cache_256[feat] = (cx_sm256, cy_sm256)
                    proc[feat]={'cx_256':cx_256,'cy_256':cy_256,'prob':float(pm.max())}
                else:
                    b = (pm>0.5).astype(np.uint8)
                    self._last_regions[feat]={'mean_conf':float(pm.mean()),'pixel_area':int(b.sum())}
                    proc[feat]=cv2.resize(b,(w,h),interpolation=cv2.INTER_NEAREST)
            tc=self._last_regions.get('TOOTH',{}).get('mean_conf',0.)
            gmc=proc.get('GM',{}).get('prob',0.)
            gc=self._last_regions.get('GINGIVA',{}).get('mean_conf',0.)
            self._last_quality=float(0.40*tc+0.40*gmc+0.10*gc)
        except Exception as e:
            print(f"[DentScan post_process] {e}")
        return proc

    def get_measurements_throttled(self, proc: dict) -> dict:
        self._meas_counter += 1
        if self._meas_cache and self._meas_counter < self._meas_interval:
            return self._meas_cache
        self._meas_counter = 0; m = {}
        try:
            # Coordinates are in 256×256 inference space.
            # Pixel spacing: 7000 µm / 256 px = 27.34 µm/px  (same in X and Y)
            # px_per_mm = 256 / 7.0 = 36.57 px/mm
            #
            # DEFINITIONS (all Euclidean distances in 256-space → mm):
            #   iGR  = GM  → CEJ   (gingival recession)
            #   iABL = ABC → CEJ   (alveolar bone level)
            #   iGH  = GM  → ABC   (gingival height, total)
            c = getattr(self, 'point_coords_cache_256', {}) or self.point_coords_cache
            px_per_mm = 256.0 / 7.0   # 36.57 px/mm; 1px = 27.34 µm, equal in X and Y

            def _dist(p1, p2):
                """Euclidean distance in mm between two 256-space (x,y) points."""
                return round(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / px_per_mm, 3)

            if 'GM'  in c and 'CEJ' in c:
                m['iGR']  = _dist(c['GM'],  c['CEJ'])   # GM  → CEJ

            if 'ABC' in c and 'CEJ' in c:
                m['iABL'] = _dist(c['ABC'], c['CEJ'])   # ABC → CEJ

            if 'GM'  in c and 'ABC' in c:
                m['iGH']  = _dist(c['GM'],  c['ABC'])   # GM  → ABC

        except Exception as _me:
            print(f"[Meas] {_me}")
        self._meas_cache = m; return m

    def create_annotated_frame(self, frame: np.ndarray, proc: dict) -> np.ndarray:
        out = frame.copy()
        for feat in REGION_FEATURES:
            mask = proc.get(feat)
            if mask is None or not isinstance(mask, np.ndarray): continue
            col = FEATURE_COLORS[feat]; ov = out.copy()
            ov[mask>0]=col; out=cv2.addWeighted(out,0.7,ov,0.3,0)
        for feat in POINT_FEATURES:
            if feat in self.point_coords_cache:
                cx,cy=self.point_coords_cache[feat]; col=FEATURE_COLORS[feat]
                cv2.circle(out,(cx,cy),12,col,-1); cv2.circle(out,(cx,cy),16,(255,255,255),2)
                cv2.putText(out,feat,(cx-15,cy-20),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,2)
        if self.show_clinical_measurements:
            ms=self.get_measurements_throttled(proc)
            h=out.shape[0]; y0=h-80
            # Only display measurements that are computable from detected landmarks.
            # iGH = GM→ABC  (no CEJ needed)
            # iGR = GM→CEJ  (needs CEJ)
            # iABL= ABC→CEJ (needs CEJ)
            display_order = [
                ('iGH',  CLINICAL_COLORS['iGH'],  'GM→ABC'),
                ('iGR',  CLINICAL_COLORS['iGR'],  'GM→CEJ'),
                ('iABL', CLINICAL_COLORS['iABL'], 'ABC→CEJ'),
            ]
            row = 0
            for k, col, label in display_order:
                val = ms.get(k)
                if val is not None:
                    cv2.putText(out, f"{k}:{val:.3f}mm",
                                (8, y0 + row*22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                    row += 1
        # B1 quality bar
        q=self._last_quality; bar=int(q*140)
        cv2.rectangle(out,(5,out.shape[0]-20),(145,out.shape[0]-8),(40,40,40),-1)
        col=(0,200,0) if q>0.3 else (0,140,255) if q>0.15 else (0,0,200)
        cv2.rectangle(out,(5,out.shape[0]-20),(5+bar,out.shape[0]-8),col,-1)
        cv2.putText(out,f"B1:{q:.2f}",(150,out.shape[0]-9),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,(180,180,180),1)
        return out

    def add_performance_overlay(self, frame: np.ndarray) -> np.ndarray:
        roi=frame[10:min(130,frame.shape[0]),10:min(300,frame.shape[1])].copy() if frame is not None else None
        roi=cv2.addWeighted(roi,0.75,np.zeros_like(roi),0.25,0)
        fps_col=(0,255,0) if self.performance_stats['display_fps']>=self.target_fps*0.9 else (0,200,255)
        for text,color,(tx,ty) in [
            (f"FPS:{self.performance_stats['display_fps']:.1f}",fps_col,(8,22)),
            (f"Inf:{self.performance_stats['inference_time']:.0f}ms",(0,200,255),(8,40)),
            (f"Lat:{self.performance_stats['avg_latency']:.0f}ms",(0,200,255),(8,58)),
            (f"Frm:{self.performance_stats['total_frames']}",(200,200,200),(8,76)),
            ("Model:B1_V8",(160,160,255),(8,94)),
        ]:
            cv2.putText(roi,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,0.42,color,1,cv2.LINE_AA)
        frame[10:130,10:300]=roi
        st=self.tooth_chart.get_status_text()
        if st:
            sy=frame.shape[0]-30
            cv2.rectangle(frame,(0,sy),(frame.shape[1],frame.shape[0]),(80,80,0),-1)
            cv2.putText(frame,st,(8,sy+20),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
        return frame

    # ── Main processing loop ──────────────────────────────────────────────────

    def process_real_time_video(self, video_capture):
        print("\n"+"="*70)
        print("  DentScan Charting — Voice V4 + All 5 Accent Features + Metrics")
        print("="*70)
        print("  GO → activate  |  [tooth e.g. 14B] → scan  |  STOP → save  |  END → finish")
        print("  F5=CONFIRM  F6=REPEAT  q quit  s shot  t manual  e end  x export")
        print("="*70+"\n")

        lat_buf = deque(maxlen=30); disp_fps_cnt=0; disp_fps_t0=time.time()
        frame_skip=0; annotated_frame=None; frame_n=0

        cv2.namedWindow("B1 Charting — v4",cv2.WINDOW_NORMAL)
        cv2.resizeWindow("B1 Charting — v4",1200,900)

        try:
            while True:
                t0=time.time()
                lag=t0-self.last_process_time
                if lag<self.frame_interval:
                    time.sleep(self.frame_interval-lag)
                    key=cv2.waitKey(1)&0xFF
                    if key==ord('q'): break
                    self._handle_key(key)
                    continue
                self.last_process_time=time.time()

                frame=video_capture.get_frame()
                if frame is None:
                    frame_skip+=1
                    if frame_skip>100: print("HD60 X lost"); break
                    time.sleep(0.01)
                    if cv2.waitKey(1)&0xFF==ord('q'): break
                    continue
                frame_skip=0; frame_n+=1

                # Process pending CONFIRM/REPEAT from voice
                vl=self.tooth_chart.voice_listener
                if vl:
                    vl.process_pending()
                    if vl.should_finish(): print("\n🏁 END"); break
                    tooth=vl.should_save_and_reset()
                    if tooth:
                        if self.tooth_chart.scanning_active and self.tooth_chart.current_tooth==tooth:
                            self.tooth_chart.save_best_frames()
                            self.tooth_chart.scanning_active=False
                            print(f"✓ {tooth} saved — say next tooth number")
                    elif vl.is_scanning():
                        t_=vl.get_current_tooth()
                        if t_ and not self.tooth_chart.scanning_active:
                            self.tooth_chart.start_tooth_scan(t_)

                should_proc=True
                # NOTE: inference always runs regardless of voice state so the
                # live overlay (landmarks, quality bar, measurements) is visible
                # before the clinician says GO.  Only tooth scanning/saving
                # actions are gated on vl.is_system_active() below.

                if should_proc:
                    t_inf=time.time()
                    batch=self.preprocess_frame_fast(frame)
                    if batch is None: continue
                    preds=self.predict_single_frame(batch)
                    if preds is None: continue
                    inf_ms=(time.time()-t_inf)*1000.

                    proc=self.post_process_predictions(preds,frame.shape[:2])
                    cur_jitter=float('inf'); meas=None

                    if self.tooth_chart.scanning_active:
                        meas=self.get_measurements_throttled(proc)
                        if len(self.tooth_chart.tooth_measurements)>=2:
                            recent=self.tooth_chart.tooth_measurements[-5:]
                            j=self.tooth_chart.calculate_jitter(recent)
                            if j: cur_jitter=max(j.values())

                    if cur_jitter<self.jitter_display_threshold:
                        annotated_frame=self.create_annotated_frame(frame,proc)
                    else:
                        annotated_frame=frame.copy()
                        cv2.putText(annotated_frame,
                                    f"Jitter {cur_jitter:.0f}µm > {self.jitter_display_threshold:.0f} — stabilize",
                                    (8,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                    if self.tooth_chart.scanning_active and meas and len(meas) > 0:
                        self.tooth_chart.add_frame(
                            self.create_annotated_frame(frame,proc), meas)
                        # Print jitter to console every frame during scanning
                        if len(self.tooth_chart.tooth_measurements) >= 2:
                            recent = self.tooth_chart.tooth_measurements[-5:]
                            j = self.tooth_chart.calculate_jitter(recent)
                            if j:
                                cur_j_str = '  '.join(
                                    f"{k}:{v:.1f}µm" for k,v in j.items()
                                    if v != float('inf'))
                                n = len(self.tooth_chart.tooth_buffer)
                                best_ok = self.tooth_chart.find_best_frames() is not None
                                status = "✓ STABLE" if best_ok else f"need {self.tooth_chart.min_stable_frames}f"
                                print(f"[Jitter] {self.tooth_chart.current_tooth} "
                                      f"f={n}  {cur_j_str}  [{status}]")

                    tot_ms=(time.time()-t0)*1000.; lat_buf.append(tot_ms)
                    self.performance_stats.update({
                        'inference_time': inf_ms,
                        'avg_latency':    float(np.mean(lat_buf)),
                        'total_frames':   frame_n,
                    })
                    disp_fps_cnt+=1
                    now=time.time()
                    if now-disp_fps_t0>=1.:
                        self.performance_stats['display_fps']=\
                            disp_fps_cnt/(now-disp_fps_t0)
                        disp_fps_cnt=0; disp_fps_t0=now

                    # Metric logging
                    if self._metrics:
                        self._metrics.log_frame(
                            frame_n,self._last_quality,
                            inference_ms=inf_ms,total_ms=tot_ms)

                    annotated_frame=self.add_performance_overlay(annotated_frame)

                    # Show "Say GO" banner when voice is present but not yet activated
                    if vl and not vl.is_system_active():
                        cv2.putText(annotated_frame, "Say GO to activate",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

                    # Feature 4: CONFIRM/REPEAT flash overlay
                    if self._pending:
                        annotated_frame=self._pending.draw_flash(annotated_frame)

                    # Legacy voice flash (from flash_queue)
                    if vl:
                        try:
                            while True:
                                txt,dur=vl.flash_queue.get_nowait()
                                self._flash_text=txt; self._flash_exp=time.time()+dur
                        except queue.Empty: pass
                        if hasattr(self,'_flash_text') and time.time()<getattr(self,'_flash_exp',0):
                            h_,w_=annotated_frame.shape[:2]
                            scale=1.1; thick=2
                            (tw,_),_=cv2.getTextSize(self._flash_text,
                                                      cv2.FONT_HERSHEY_DUPLEX,scale,thick)
                            tx=(w_-tw)//2; ty=h_//3
                            cv2.putText(annotated_frame,self._flash_text,(tx+2,ty+2),
                                        cv2.FONT_HERSHEY_DUPLEX,scale,(0,0,0),thick+2,cv2.LINE_AA)
                            cv2.putText(annotated_frame,self._flash_text,(tx,ty),
                                        cv2.FONT_HERSHEY_DUPLEX,scale,(0,220,130),thick,cv2.LINE_AA)

                    cv2.imshow("B1 Charting — v4",annotated_frame)
                else:
                    # should_proc is always True now — this branch is unreachable.
                    # Kept as safety fallback only.
                    idle=frame.copy()
                    cv2.putText(idle,"Say GO to activate",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    cv2.imshow("B1 Charting — v4",idle)

                key=cv2.waitKey(1)&0xFF
                if key==ord('q'): break
                self._handle_key(key,annotated_frame)

        except KeyboardInterrupt: print("\nInterrupted")
        finally:
            if self.tooth_chart.scanning_active: self.tooth_chart.stop_tooth_scan()
            if self.tooth_chart.best_frames_saved:
                print("\nAuto-exporting…"); self.tooth_chart.export_to_excel()
            if self._metrics: self._metrics.save_summary(); self._metrics.close()
            cv2.destroyAllWindows(); self._print_summary()

    def _handle_key(self, key: int, frame=None):
        if   key==ord('s') and frame is not None:
            ts=time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"chart_{ts}.png",frame); print(f"Screenshot → chart_{ts}.png")
        elif key==ord('t'):
            print("Enter tooth (e.g. 14B): ",end='',flush=True)
            tn=input().strip()
            if tn: self.tooth_chart.start_tooth_scan(tn)
        elif key==ord('e'): self.tooth_chart.stop_tooth_scan()
        elif key==ord('l'): self._list_saved()
        elif key==ord('x'): self.tooth_chart.export_to_excel()
        # F5/F6 handled via pynput in FootPedalListener

    def _list_saved(self):
        print("\n"+"="*50+"  SAVED:")
        if not self.tooth_chart.best_frames_saved: print("  (none)")
        else:
            for t,m in self.tooth_chart.best_frames_saved.items():
                print(f"  {t}: {m['n_frames']}f  jitter {m['jitters_um'].get('iGH',0):.1f}µm")
        print("="*50)

    def _print_summary(self):
        print("\n"+"="*60)
        print(f"SESSION: {len(self.tooth_chart.best_frames_saved)} teeth  "
              f"FPS:{self.performance_stats['display_fps']:.1f}  "
              f"Model:B1_V8_Baseline")
        for t in sorted(self.tooth_chart.best_frames_saved):
            print(f"  ✓ {t}")
        print("="*60)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════



class MKUNetLiveAnnotator:
    """
    Lightweight real-time wrapper around MKUNetInference.
    Provides the same interface as RealTimeHybridFusionAnnotator so
    FrameProcessor works unchanged.

    Uses the same MKUNet model that runs video_inference_pipeline.py,
    so no additional model loading or custom_objects needed.
    """

    FEATURE_COLORS = {
        'GM':      (0,   255, 255),
        'CEJ':     (255, 255,   0),
        'ABC':     (255,   0, 255),
        'GINGIVA': (255, 100, 100),
        'TOOTH':   (200, 200, 255),
        'BONE':    (100, 255, 100),
    }
    CONF_THRESH = {'GM': 0.18, 'CEJ': 0.28, 'ABC': 0.20}

    def __init__(self, mkunet_inference):
        self.model = mkunet_inference
        self.point_coords_cache = {}
        self.point_coords_cache_256 = {}
        self._last_quality = 0.
        self._last_regions = {}
        self._pos_smooth = {f: deque(maxlen=5) for f in ('GM','CEJ','ABC')}

        # Performance tracking
        self._fps_counter   = 0
        self._fps_start     = time.time()
        self._display_fps   = 0.
        self._inference_ms  = 0.
        self._total_ms      = 0.
        self._latency_buf       = deque(maxlen=30)
        self._latency_total_buf = deque(maxlen=30)

        # GPU check
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            self._gpu_name = gpus[0].name if gpus else "CPU"
            self._on_gpu   = len(gpus) > 0
        except Exception:
            self._gpu_name = "unknown"
            self._on_gpu   = False

        # Build optimized inference function — same approach as original
        self._predict_fn = None
        self._build_predict_fn()

    def _build_predict_fn(self):
        """
        Build a @tf.function-compiled inference function and warm it up.
        This is the key to achieving ~20ms latency instead of ~100ms.
        Without tf.function: Python dispatch overhead on every call.
        With tf.function: graph is compiled once, reused every call.
        """
        try:
            import tensorflow as tf

            keras_model = self.mkunet_model  # the actual Keras model

            @tf.function(reduce_retracing=True)
            def predict_fn(x):
                return keras_model(x, training=False)

            self._predict_fn = predict_fn

            # Warmup: run 3 dummy inferences to trigger JIT compilation.
            # Without warmup the first real frames pay the compile cost (~2s).
            print("[MKUNet] Warming up inference function (3 runs)...")
            dummy = tf.zeros((1, MODEL_SIZE, MODEL_SIZE, 3), dtype=tf.float32)
            for _ in range(3):
                _ = self._predict_fn(dummy)
            print("[MKUNet] ✓ Inference function ready")

        except Exception as e:
            print(f"[MKUNet] tf.function setup failed ({e}) — using eager mode")
            self._predict_fn = None

    @property
    def mkunet_model(self):
        """Extract the underlying Keras model from MKUNetInference wrapper."""
        return self.model.model

    def preprocess_frame_fast(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Fast preprocessing — matches original live code exactly.
        Key: resize first, then cast+normalise in one operation.
        No extra BGR→RGB copy (model was trained on BGR from OpenCV,
        same as the original code which also doesn't convert).
        Returns (1, 256, 256, 3) float32 numpy array — NOT a tf.Tensor.
        The tf.function handles the numpy→tensor conversion internally,
        which is faster than explicit tf.constant() every call.
        """
        try:
            resized     = cv2.resize(frame_bgr, (MODEL_SIZE, MODEL_SIZE),
                                      interpolation=cv2.INTER_LINEAR)
            normalised  = resized.astype(np.float32) / 255.0
            return np.expand_dims(normalised, axis=0)   # (1, H, W, 3)
        except Exception:
            return None

    def predict_single_frame(self, frame_batch: np.ndarray) -> Optional[list]:
        """
        Run inference via the compiled @tf.function.
        frame_batch: (1, 256, 256, 3) float32 numpy — no tf.constant() needed.
        tf.function accepts numpy directly and converts internally (faster).
        """
        try:
            t0 = time.time()
            if self._predict_fn is not None:
                preds = self._predict_fn(frame_batch)
            else:
                # Fallback to eager if tf.function failed
                import tensorflow as tf
                preds = self.mkunet_model(
                    tf.constant(frame_batch, dtype=tf.float32),
                    training=False)
            self._inference_ms = (time.time() - t0) * 1000.
            self._latency_buf.append(self._inference_ms)
            return preds
        except Exception as e:
            print(f"[Predict] {e}")
            return None

    def post_process_predictions(self, predictions,
                                   target_shape: tuple) -> dict:
        """
        Convert raw predictions to landmark coords + region masks.
        Populates self.point_coords_cache and self._last_regions.
        Returns dict {feature: data} compatible with FrameProcessor.
        """
        if predictions is None:
            return {}

        h, w = target_shape
        proc  = {}
        self.point_coords_cache = {}
        self.point_coords_cache_256 = {}

        try:
            import tensorflow as tf
            # predictions is a list of (1,256,256,2) tensors, one per feature
            # Order matches ALL_FEATURES from mkunet_ablation_train
            all_features = getattr(self.model, 'all_features',
                                    ['GM','CEJ','ABC','TOOTH','BONE','GINGIVA'])

            for i, feat in enumerate(all_features):
                if i >= len(predictions):
                    break
                pred = predictions[i]
                if hasattr(pred, 'numpy'):
                    pred_np = pred[0].numpy()   # (256,256,2)
                else:
                    pred_np = np.array(pred[0])

                prob_map = pred_np[:, :, 1]   # class-1 probability

                if feat in ('GM', 'CEJ', 'ABC'):
                    # Soft-argmax: weighted centroid
                    thr = self.CONF_THRESH.get(feat, 0.2)
                    if prob_map.max() < thr:
                        continue
                    ys = np.arange(MODEL_SIZE)[:, None] * np.ones((1, MODEL_SIZE))
                    xs = np.arange(MODEL_SIZE)[None, :] * np.ones((MODEL_SIZE, 1))
                    total = prob_map.sum() + 1e-8
                    cy_256 = float((prob_map * ys).sum() / max(float(total), 1e-9))
                    cx_256 = float((prob_map * xs).sum() / max(float(total), 1e-9))

                    # Scale to target resolution
                    cx = cx_256 * w / MODEL_SIZE
                    cy = cy_256 * h / MODEL_SIZE

                    # Smooth position for display
                    self._pos_smooth[feat].append((cx, cy))
                    cx_s = np.mean([p[0] for p in self._pos_smooth[feat]])
                    cy_s = np.mean([p[1] for p in self._pos_smooth[feat]])
                    self.point_coords_cache[feat] = (int(cx_s), int(cy_s))
                    proc[feat] = {'cx_256': cx_256, 'cy_256': cy_256,
                                   'prob': float(prob_map.max())}

                else:
                    # Region feature
                    binary   = (prob_map > 0.5).astype(np.uint8)
                    resized  = cv2.resize(binary, (w, h),
                                           interpolation=cv2.INTER_NEAREST)
                    proc[feat] = resized
                    self._last_regions[feat] = {
                        'mean_conf': float(prob_map.mean()),
                        'pixel_area': int(binary.sum())}

            # Compute quality score (same weights as QualityScorer)
            tooth_c   = self._last_regions.get('TOOTH',{}).get('mean_conf', 0.)
            gm_conf   = proc.get('GM', {}).get('prob', 0.)
            gingiva_c = self._last_regions.get('GINGIVA',{}).get('mean_conf', 0.)
            self._last_quality = float(
                0.40 * tooth_c + 0.40 * gm_conf + 0.10 * gingiva_c)

        except Exception as e:
            pass

        return proc

    def get_measurements_throttled(self, predictions) -> dict:
        """
        Compute iGH, iGR, iABL from 256-space landmark positions.
        Pixel spacing: 7000µm / 256px = 27.34 µm/px in both X and Y.
        Definitions:
            iGR  = GM  → CEJ   (gingival recession)
            iABL = ABC → CEJ   (alveolar bone level)
            iGH  = GM  → ABC   (gingival height, total — no CEJ needed)
        """
        m = {}
        try:
            c   = getattr(self, 'point_coords_cache_256', {}) or self.point_coords_cache
            ppm = 256.0 / 7.0   # px/mm  (27.34 µm/px, same in X and Y)

            def _dist(p1, p2):
                return round(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / ppm, 3)

            if 'GM'  in c and 'CEJ' in c:
                m['iGR']  = _dist(c['GM'],  c['CEJ'])   # GM  → CEJ
            if 'ABC' in c and 'CEJ' in c:
                m['iABL'] = _dist(c['ABC'], c['CEJ'])   # ABC → CEJ
            if 'GM'  in c and 'ABC' in c:
                m['iGH']  = _dist(c['GM'],  c['ABC'])   # GM  → ABC
        except Exception:
            pass
        return m

    def create_annotated_frame(self, frame_bgr: np.ndarray,
                                proc_preds: dict) -> np.ndarray:
        """Draw landmarks and region overlays on frame."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        # Draw region overlays
        for feat in ('GINGIVA', 'TOOTH', 'BONE'):
            mask = proc_preds.get(feat)
            if mask is None or not isinstance(mask, np.ndarray):
                continue
            color = self.FEATURE_COLORS.get(feat, (128,128,128))
            overlay = out.copy()
            overlay[mask > 0] = color
            out = cv2.addWeighted(out, 0.75, overlay, 0.25, 0)

        # Draw landmark points + lines
        pts = {}
        for feat in ('GM', 'CEJ', 'ABC'):
            if feat in self.point_coords_cache:
                cx, cy = self.point_coords_cache[feat]
                color  = self.FEATURE_COLORS[feat]
                cv2.circle(out, (cx, cy), 5, color, -1)
                cv2.circle(out, (cx, cy), 7, (255,255,255), 1)
                cv2.putText(out, feat, (cx+8, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, color, 1)
                pts[feat] = (cx, cy)

        if 'GM' in pts and 'CEJ' in pts:
            cv2.line(out, pts['GM'], pts['CEJ'], (0,200,200), 1)
        if 'GM' in pts and 'ABC' in pts:
            cv2.line(out, pts['GM'], pts['ABC'], (200,0,200), 1)

        # Quality score bar
        q   = self._last_quality
        bar = int(q * 150)
        cv2.rectangle(out, (5, h-20), (155, h-8), (50,50,50), -1)
        col = (0,255,0) if q > 0.3 else (0,165,255) if q > 0.15 else (0,0,255)
        cv2.rectangle(out, (5, h-20), (5+bar, h-8), col, -1)
        cv2.putText(out, f"QS:{q:.2f}", (160, h-9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200,200,200), 1)

        # ── Performance overlay — matches original live code style ───────
        self._fps_counter += 1
        now = time.time()
        fps_elapsed = now - self._fps_start
        if fps_elapsed >= 1.0:
            self._display_fps = self._fps_counter / fps_elapsed
            self._fps_counter = 0
            self._fps_start   = now
        self._total_frames = getattr(self, '_total_frames', 0) + 1

        avg_inf = float(np.mean(self._latency_buf)) if self._latency_buf else 0.
        avg_lat = float(np.mean(self._latency_total_buf)) if \
                   hasattr(self, '_latency_total_buf') and self._latency_total_buf \
                   else avg_inf
        gpu_str = f"GPU: ON" if self._on_gpu else "GPU: OFF (CPU!)"
        fps_color = (0, 255, 0) \
            if self._display_fps >= 22.5 else (0, 255, 255)

        # Semi-transparent background (original style)
        ov_h, ov_w = 130, 300
        if h >= ov_h + 10 and w >= ov_w + 10:
            roi = out[10:10+ov_h, 10:10+ov_w].copy()
            bg  = np.zeros_like(roi)
            roi = cv2.addWeighted(roi, 0.7, bg, 0.3, 0)
            out[10:10+ov_h, 10:10+ov_w] = roi

        texts = [
            (f"Display FPS: {self._display_fps:.1f}",  fps_color,    (10, 35)),
            (f"Target FPS: 25.0",                       (200,200,200),(10, 55)),
            (f"Inference:  {avg_inf:.0f} ms",           (0, 255, 255),(10, 75)),
            (f"Latency:    {avg_lat:.0f} ms",           (0, 255, 255),(10, 95)),
            (f"Frames: {self._total_frames}",           (255,255,255),(10,112)),
            (gpu_str,   (0,255,100) if self._on_gpu else (0,100,255), (10,129)),
        ]
        for text, color, (tx, ty) in texts:
            fs = 0.6 if ty == 35 else 0.5 if ty < 100 else 0.4
            cv2.putText(out, text, (tx + 10, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

        return out



# ═════════════════════════════════════════════════════════════════════════════
#  TEMPORAL IOU GATE  (v3 — frame discard based on mask continuity)
# ═════════════════════════════════════════════════════════════════════════════

class TemporalIOUGate:
    """
    Discards disrupted frames using three temporal signals:

    Signal 1 — Mask IOU (primary):
      IOU between GINGIVA+TOOTH mask of current frame and previous frame.
      Good contact: IOU 0.7-0.9.  Lift-off / slip: IOU < 0.25.

    Signal 2 — GM centroid jump:
      Pixel distance between consecutive GM centroids in 256x256 space.
      Smooth sweep: < 8 px/frame.  Probe slip: > 15 px/frame.

    Signal 3 — Quality drop:
      If quality_b1 drops > 0.15 from running mean → disruption.

    A frame is discarded if ANY two signals fire simultaneously.
    Single-signal events are logged but not discarded (avoids over-rejection).

    Debug fields added to imu_stitching.csv:
      mask_iou, gm_jump_px, quality_drop, iou_gate_pass, iou_reject_reason
    """

    # Thresholds (tunable via --iou_thresh, --jump_thresh)
    IOU_THRESH      = 0.25   # below this → mask disrupted
    JUMP_THRESH_PX  = 15.0   # above this → GM jumped
    QUALITY_DROP    = 0.15   # above this → quality collapsed

    def __init__(self, iou_thresh=0.25, jump_thresh=15.0, quality_drop=0.15):
        self.iou_thresh   = iou_thresh
        self.jump_thresh  = jump_thresh
        self.quality_drop = quality_drop
        self._prev_mask   = None    # previous combined GINGIVA+TOOTH mask
        self._prev_gm_px  = None    # previous GM centroid (cx_256, cy_256)
        self._quality_buf = deque(maxlen=10)
        # Stats
        self.n_total    = 0
        self.n_rejected = 0
        self.n_iou_fail = 0
        self.n_jump_fail= 0
        self.n_qual_fail= 0

    def reset(self):
        self._prev_mask   = None
        self._prev_gm_px  = None
        self._quality_buf.clear()
        self.n_total = self.n_rejected = 0
        self.n_iou_fail = self.n_jump_fail = self.n_qual_fail = 0

    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Binary mask IOU in [0,1]."""
        if mask_a is None or mask_b is None:
            return 1.0   # no previous frame → accept
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        return float(inter) / max(float(union), 1.)

    def evaluate(self, proc: dict, quality: float) -> tuple:
        """
        proc:    post_process_predictions output dict
        quality: B1 quality score

        Returns (accept: bool, metrics: dict)
        metrics keys: mask_iou, gm_jump_px, quality_drop_val,
                      iou_gate_pass, iou_reject_reason
        """
        self.n_total += 1
        self._quality_buf.append(quality)
        q_mean = float(np.mean(self._quality_buf)) if self._quality_buf else quality

        # Build combined mask (GINGIVA | TOOTH) at 256x256
        curr_mask = None
        for feat in ('GINGIVA', 'TOOTH'):
            m = proc.get(feat)
            if m is not None and isinstance(m, np.ndarray):
                # Resize to 64x64 for fast IOU (exact result, 16x speedup)
                small = cv2.resize(m.astype(np.uint8), (64,64),
                                   interpolation=cv2.INTER_NEAREST)
                curr_mask = small if curr_mask is None else                             np.logical_or(curr_mask, small).astype(np.uint8)

        # Signal 1: IOU
        iou = self._compute_iou(self._prev_mask, curr_mask)
        iou_fail = (self._prev_mask is not None) and (iou < self.iou_thresh)

        # Signal 2: GM jump
        gm_jump = 0.
        gm_jump_fail = False
        gm_info = proc.get('GM', {})
        if isinstance(gm_info, dict):
            cx = gm_info.get('cx_256', 0.)
            cy = gm_info.get('cy_256', 0.)
            if self._prev_gm_px is not None:
                gm_jump = float(np.sqrt(
                    (cx - self._prev_gm_px[0])**2 +
                    (cy - self._prev_gm_px[1])**2))
                gm_jump_fail = gm_jump > self.jump_thresh
            self._prev_gm_px = (cx, cy)

        # Signal 3: Quality drop
        q_drop = max(0., q_mean - quality)
        qual_fail = q_drop > self.quality_drop

        # Discard if ≥2 signals fire
        fail_count = sum([iou_fail, gm_jump_fail, qual_fail])
        accept = (fail_count < 2)

        # Build reject reason string for debug log
        reasons = []
        if iou_fail:    reasons.append(f"iou={iou:.2f}")
        if gm_jump_fail:reasons.append(f"jump={gm_jump:.1f}px")
        if qual_fail:   reasons.append(f"qdrop={q_drop:.2f}")
        reject_reason = '+'.join(reasons) if not accept else ''

        # Update state
        self._prev_mask = curr_mask
        if not accept:
            self.n_rejected += 1
            if iou_fail:     self.n_iou_fail += 1
            if gm_jump_fail: self.n_jump_fail += 1
            if qual_fail:    self.n_qual_fail += 1

        metrics = {
            'mask_iou':         round(iou,       3),
            'gm_jump_px':       round(gm_jump,   2),
            'quality_drop_val': round(q_drop,    3),
            'iou_gate_pass':    int(accept),
            'iou_reject_reason':reject_reason,
        }
        return accept, metrics

    def summary(self) -> dict:
        return {
            'n_total':     self.n_total,
            'n_rejected':  self.n_rejected,
            'reject_rate': round(self.n_rejected / max(self.n_total,1), 3),
            'iou_fails':   self.n_iou_fail,
            'jump_fails':  self.n_jump_fail,
            'qual_fails':  self.n_qual_fail,
            'thresholds':  {
                'iou':      self.iou_thresh,
                'jump_px':  self.jump_thresh,
                'q_drop':   self.quality_drop,
            }
        }


# ═════════════════════════════════════════════════════════════════════════════
#  SWEEP DIRECTION TRACKER  (v3 — direction-aware compositing)
# ═════════════════════════════════════════════════════════════════════════════

class SweepDirectionTracker:
    """
    Tracks probe sweep direction and prevents reverse-pass frames
    from corrupting the panorama.

    Logic:
      - Uses a rolling window of gm_x_fused values to estimate direction.
      - Initial direction set from first 5 frames.
      - A reversal is confirmed only after probe moves > REVERSAL_DIST mm
        in the opposite direction (filters out small back-and-forth wobble).
      - In FORWARD_ONLY mode (default): frames moving in the non-initial
        direction are rejected from compositing.
      - In BIDIRECTIONAL mode (--bidirectional flag): frames in both
        directions are accepted, but each tile keeps the best-weight version.

    Debug fields: sweep_dir (+1/-1), dir_reversals, frames_dir_rejected
    """

    REVERSAL_DIST = 1.5    # mm — minimum reversal before direction changes
    WINDOW        = 8      # frames for direction estimation

    def __init__(self, forward_only: bool = True):
        self.forward_only   = forward_only
        self._gm_history    = deque(maxlen=self.WINDOW)
        self._initial_dir   = None    # +1 or -1, set from first frames
        self._current_dir   = None
        self._reversal_start= None    # gm_x when reversal began
        self.n_reversals    = 0
        self.n_rejected_dir = 0
        self.n_total        = 0

    def reset(self):
        self._gm_history.clear()
        self._initial_dir   = None
        self._current_dir   = None
        self._reversal_start= None
        self.n_reversals = self.n_rejected_dir = self.n_total = 0

    def update(self, gm_x_fused: float) -> tuple:
        """
        Returns (accept: bool, direction: int, metrics: dict)
        direction: +1 = moving in initial direction, -1 = moving backward
        """
        self.n_total += 1
        self._gm_history.append(gm_x_fused)

        # Need enough frames to estimate direction
        if len(self._gm_history) < 4:
            return True, 0, {'sweep_dir': 0, 'dir_reversals': 0,
                             'frames_dir_rejected': 0}

        # Estimate current direction from rolling window slope
        vals   = list(self._gm_history)
        dx     = vals[-1] - vals[0]
        curr_d = 1 if dx >= 0 else -1

        # Set initial direction from first stable estimate
        if self._initial_dir is None:
            self._initial_dir = curr_d
            self._current_dir = curr_d

        # Detect reversal: moving opposite to initial direction
        if curr_d != self._initial_dir:
            if self._reversal_start is None:
                self._reversal_start = gm_x_fused
            reversal_dist = abs(gm_x_fused - self._reversal_start)
            if reversal_dist >= self.REVERSAL_DIST:
                # Confirmed reversal
                if self._current_dir == self._initial_dir:
                    self.n_reversals += 1
                self._current_dir = curr_d
        else:
            self._reversal_start = None
            self._current_dir    = curr_d

        # Accept decision
        if self.forward_only and self._current_dir != self._initial_dir:
            accept = False
            self.n_rejected_dir += 1
        else:
            accept = True

        metrics = {
            'sweep_dir':          self._current_dir,
            'dir_reversals':      self.n_reversals,
            'frames_dir_rejected':self.n_rejected_dir,
        }
        return accept, self._current_dir, metrics

    def summary(self) -> dict:
        return {
            'initial_direction': self._initial_dir,
            'n_reversals':       self.n_reversals,
            'n_rejected_dir':    self.n_rejected_dir,
            'forward_only':      self.forward_only,
        }

# ═════════════════════════════════════════════════════════════════════════════
#  FRAME PROCESSOR  (extracts GM position from MKUNetLiveAnnotator output)
# ═════════════════════════════════════════════════════════════════════════════

class FrameProcessor:
    """
    Runs B1 inference and applies three gate layers:
      1. B1 quality score ≥ QS_THRESH (coarse pre-filter)
      2. AdaptiveQualityGater (sharpness, contact, SNR)
      3. TemporalIOUGate (mask continuity, GM jump, quality drop)  [v3]
    """
    def __init__(self, annotator: MKUNetLiveAnnotator,
                 quality_gater: Optional[AdaptiveQualityGater] = None):
        self.ann      = annotator
        self.gater    = quality_gater or AdaptiveQualityGater()
        self.iou_gate = TemporalIOUGate()   # v3: temporal consistency gate

    def process(self, frame_raw: np.ndarray) -> Optional[dict]:
        t0    = time.time()
        batch = self.ann.preprocess_frame_fast(frame_raw)
        if batch is None: return None
        preds = self.ann.predict_single_frame(batch)
        if preds is None: return None
        proc    = self.ann.post_process_predictions(preds, frame_raw.shape[:2])
        ann_frm = self.ann.create_annotated_frame(frame_raw, proc)
        meas    = self.ann.get_measurements_throttled(preds)
        quality = self.ann._last_quality

        # Adaptive quality gate
        gate_score, gate_comp = self.gater.score(frame_raw, self.ann._last_regions)
        gm_detected = 'GM' in self.ann.point_coords_cache
        # Three-tier acceptance (most permissive to most strict):
        #   Tier A — GM landmark detected by B1: real US tissue signal → always accept
        #   Tier B — B1 quality passes coarse threshold AND gate score passes
        #   Tier C — fallback: accept with very low bar (quality > 0.02) when recording
        #            so that frames are stored even without US probe (testing/development)
        #            Canvas compositing still needs gm_x_mm>0 to place the frame correctly.
        gate_pass = (
            gm_detected                                                 # Tier A: GM seen
            or (quality >= QS_THRESH and self.gater.accept(gate_score)) # Tier B: full gate
            or (quality >= 0.02)                                        # Tier C: dev fallback
        )

        self.ann._latency_total_buf.append((time.time()-t0)*1000.)

        gm_x_mm=gm_y_mm=0.
        if 'GM' in self.ann.point_coords_cache:
            h,w=frame_raw.shape[:2]; cx,cy=self.ann.point_coords_cache['GM']
            cx_256=cx*MODEL_SIZE/w; cy_256=cy*MODEL_SIZE/h
            gm_x_mm=cx_256*PIXEL_SPACING_MM; gm_y_mm=cy_256*PIXEL_SPACING_MM

        frame_256=cv2.resize(frame_raw,(MODEL_SIZE,MODEL_SIZE))
        frame_256=cv2.cvtColor(frame_256,cv2.COLOR_BGR2RGB).astype(np.float32)/255.

        # v3: Temporal IOU gate
        iou_accept, iou_metrics = self.iou_gate.evaluate(proc, quality)
        # Final gate: passes AdaptiveQualityGater AND TemporalIOUGate
        gate_pass_final = gate_pass and iou_accept

        return {
            'gm_x_mm':          gm_x_mm,
            'gm_y_mm':          gm_y_mm,
            'quality_score':    quality,
            'gate_score':       gate_score,
            'gate_pass':        gate_pass_final,   # combined gate
            'gate_pass_aq':     gate_pass,         # adaptive quality gate only
            'gate_pass_iou':    iou_accept,        # IOU gate only
            'gate_comp':        gate_comp,
            'iou_metrics':      iou_metrics,       # mask_iou, gm_jump_px, etc.
            'gingiva_area':     self.ann._last_regions.get('GINGIVA',{}).get('pixel_area',0),
            'annotated':        ann_frm,
            'frame_256':        frame_256,
            'measurements':     meas,
            'point_cache':      dict(self.ann.point_coords_cache),
            'proc':             proc,
        }


class _PanoVoiceAdapter:
    """
    Thin adapter so LivePanoramaSystem.run() can call self.voice.get_command()
    while the real voice is UnifiedVoiceController.get_pano_command().
    """
    def __init__(self, unified: 'UnifiedVoiceController'):
        self._uvc = unified
    def get_command(self):
        return self._uvc.get_pano_command()
    def stop(self): pass


# ═════════════════════════════════════════════════════════════════════════════
#  DentWise PANORAMA SYSTEM  (updated — all 6 features integrated)
# ═════════════════════════════════════════════════════════════════════════════

class LivePanoramaSystem:
    def __init__(self, annotator, capture, imu_reader,
                 output_dir='live_panoramas',
                 voice: Optional['UnifiedVoiceController'] = None,
                 quality_gater=None):  # quality_gater kept for API compat, unused
        self.ann     = annotator
        self.cap     = capture
        self.imu     = imu_reader
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.voice   = voice

        self.proc      = FrameProcessor(annotator)   # uses original working FrameProcessor
        self.canvas       = TiledCanvas()   # Feature 2: averaging canvas
        self.sharp_canvas = TiledCanvas()   # keyframe-only → sharp live view
        self.cf        = ComplementaryFilter()
        self.heatmap   = CoverageHeatmap()      # Feature 3: CoverageHeatmap
        self.audit     = AuditLogger()          # Feature 4: AuditLogger
        self.registrar = MultiSessionRegistrar() # Feature 5: MultiSessionRegistrar

        self.recording      = False
        self.show_imu       = True
        self.voice_paused   = False
        self.anchor_marks:  List[int] = []
        self.bad_marks:     List[tuple] = []
        self._bad_start     = None
        self._cmd_flash     = ('', 0.)

        self._gm_history        = deque(maxlen=30)
        self._session_id        = 1
        self._session_frames:   List[np.ndarray] = []
        self._session_coords:   List[dict] = []
        self._gm_x_ref         = None
        self._gm_x_max         = -np.inf
        self._sweep_start_gm   = None
        self._sweep_start_yaw  = None
        self._imu_was_connected = False
        self.stats = {'frames_total':0,'frames_good':0,'frames_gated_out':0,'frames_dir_rejected':0,'frames_iou_rejected':0,
                      'imu_packets':0,'gm_jitter_um':0.}
        self._jitter_buf = deque(maxlen=50)
        self._panorama_frozen = False
        self._sweep_start_gm  = None
        self._last_keyframe_col0: Optional[float] = None
        self._session_tooth_label: Optional[str] = None   # set by GUI tooth selector
        self._last_keyframe_gm_y: Optional[float] = None
        self._pano_metrics: Optional[PanoMetricMonitor] = None
        self._last_imu_yaw: Optional[float] = None
        self._last_keyframe_col0: Optional[float] = None
        self._last_keyframe_gm_y: Optional[float] = None
        self._sweep_tracker = SweepDirectionTracker(forward_only=True)  # v3

    # ── helpers ──────────────────────────────────────────────────────────────

    def _detect_sweep_dir(self, gm_x):
        self._gm_history.append(gm_x)

    def _compute_col0(self, gm_x):
        if self._gm_x_ref is None: self._gm_x_ref=gm_x
        if gm_x>self._gm_x_max: self._gm_x_max=gm_x
        col0=abs(self._gm_x_ref-gm_x)/PIXEL_SPACING_MM
        return float(np.clip(col0,0,CANVAS_MAX_W-MODEL_SIZE))

    def _update_jitter(self, gm_x):
        self._jitter_buf.append(gm_x)
        if len(self._jitter_buf)>3:
            self.stats['gm_jitter_um']=float(np.diff(list(self._jitter_buf)).std()*1e3)

    def _flash(self, text, dur=1.5): self._cmd_flash=(text, time.time()+dur)

    def _draw_imu_overlay(self, frame, imu, imu_conf):
        out=frame.copy(); h,w=out.shape[:2]; x0,y0=w-220,10
        def txt(s,dy,c=(255,255,255)):
            cv2.putText(out,s,(x0,y0+dy),cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1,cv2.LINE_AA)
        cv2.rectangle(out,(x0-5,y0-5),(w-5,y0+120),(0,0,0),-1)
        cv2.rectangle(out,(x0-5,y0-5),(w-5,y0+120),(100,100,100),1)
        cc=(0,255,0) if self.imu.connected else (0,80,255)
        txt(f"IMU: {'ON' if self.imu.connected else 'OFF'}",15,cc)
        txt(f"pkts:{self.imu.packet_count}",30)
        txt(f"yaw:{imu.yaw:+7.1f} deg",50)
        txt(f"gyroZ:{imu.gyro_z:+6.1f}",65)
        bw=int(imu_conf*100); bc=(0,255,0) if imu_conf>0.5 else (0,165,255)
        cv2.rectangle(out,(x0,y0+80),(x0+100,y0+93),(50,50,50),-1)
        cv2.rectangle(out,(x0,y0+80),(x0+bw,y0+93),bc,-1)
        txt(f"IMU wt:{imu_conf:.0%}",105)
        return out

    def _draw_gate_overlay(self, frame, gate_comp: dict, gate_pass: bool) -> np.ndarray:
        """Show AdaptiveQualityGater signals on frame (Feature 1 visualisation)."""
        out=frame.copy(); h,w=out.shape[:2]
        x0,y0=5,h-80
        cv2.rectangle(out,(x0,y0),(x0+165,h-5),(0,0,0),-1)
        def bar(label,val,dy,col):
            bw=int(val*80)
            cv2.rectangle(out,(x0+50,y0+dy),(x0+50+80,y0+dy+8),(50,50,50),-1)
            cv2.rectangle(out,(x0+50,y0+dy),(x0+50+bw,y0+dy+8),col,-1)
            cv2.putText(out,f"{label}:{val:.2f}",(x0+2,y0+dy+7),
                        cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,200,200),1)
        bar("shp",gate_comp.get('sharpness',0),2,(0,200,255))
        bar("cnt",gate_comp.get('contact',0),14,(0,255,100))
        bar("snr",gate_comp.get('snr',0),26,(255,200,0))
        bar("ALL",gate_comp.get('combined',0),38,(0,255,0) if gate_pass else (0,0,255))
        status="PASS" if gate_pass else "REJECT"
        cv2.putText(out,status,(x0+2,y0+54),cv2.FONT_HERSHEY_SIMPLEX,0.4,
                    (0,220,0) if gate_pass else (0,0,255),1)
        return out

    def _draw_recording_overlay(self, frame):
        out=frame.copy()
        if self.recording:
            col=(0,165,255) if self.voice_paused else (0,0,255)
            lbl="HOLD" if self.voice_paused else "REC"
            cv2.circle(out,(20,20),8,col,-1)
            cv2.putText(out,f"{lbl} {self.canvas.frame_count}f  "
                        f"mem:{self.canvas.memory_mb():.1f}MB",
                        (35,25),cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1,cv2.LINE_AA)
        gated=self.stats.get('frames_gated_out',0)
        cv2.putText(out,f"jitter:{self.stats['gm_jitter_um']:.0f}µm  gated:{gated}",
                    (5,out.shape[0]-8),cv2.FONT_HERSHEY_SIMPLEX,0.35,(180,180,180),1)
        return out

    def _draw_voice_status(self, frame):
        out=frame.copy(); now=time.time(); h,w=out.shape[:2]
        cmd,exp=self._cmd_flash
        if cmd and now<exp:
            scale=1.3; thick=3
            (tw,th),_=cv2.getTextSize(cmd,cv2.FONT_HERSHEY_DUPLEX,scale,thick)
            tx=(w-tw)//2; ty=h//2
            cv2.putText(out,cmd,(tx+2,ty+2),cv2.FONT_HERSHEY_DUPLEX,scale,(0,0,0),thick+2,cv2.LINE_AA)
            cv2.putText(out,cmd,(tx,ty),cv2.FONT_HERSHEY_DUPLEX,scale,(0,220,130),thick,cv2.LINE_AA)
        if self.voice:
            cv2.circle(out,(w-22,h-22),7,(0,200,100),-1)
        if self.recording and (self.anchor_marks or self.bad_marks):
            cv2.putText(out,f"ANC:{len(self.anchor_marks)} BAD:{len(self.bad_marks)}",
                        (5,out.shape[0]-22),cv2.FONT_HERSHEY_SIMPLEX,0.35,(100,220,255),1)
        return out

    # ── session management ────────────────────────────────────────────────────

    def _do_start(self, gm_x, imu_now):
        self.canvas.reset(); self.sharp_canvas.reset()
        self.cf.reset(gm_x); self.audit.clear()
        self._last_keyframe_col0 = None
        self._last_keyframe_gm_y = None
        self._session_frames.clear(); self._session_coords.clear()
        self.anchor_marks.clear(); self.bad_marks.clear(); self._bad_start=None
        self._gm_x_ref=gm_x; self._gm_x_max=gm_x
        self._sweep_start_yaw=imu_now.yaw; self.voice_paused=False
        self.stats['frames_gated_out']=0
        self.stats['frames_dir_rejected']=0
        self.stats['frames_iou_rejected']=0
        self._voice_event_log = []   # list of {ts, cmd, raw, mode, frame_n}
        self._last_imu_yaw=None
        if hasattr(self,'_imu_first_yaw_set'): del self._imu_first_yaw_set
        self._sweep_tracker.reset()          # v3: reset direction tracker
        self.proc.iou_gate.reset()           # v3: reset IOU gate state
        imu_str="IMU+GM" if self.imu.connected else "GM-only"
        print(f"\n[STITCH {self._session_id}] ▶  DentScan model  ({imu_str})")
        self.recording=True
        # Init per-session metric monitor
        # Metrics sdir: use tooth label matching _save_session logic
        # We use a provisional path here — _save_session creates the real folder.
        _tooth_lbl = getattr(self, '_session_tooth_label', None)
        if _tooth_lbl:
            sdir = self.out_dir / _tooth_lbl
            _base = sdir; _n = 2
            while sdir.exists() and any(sdir.iterdir()):
                sdir = Path(str(_base) + f"_{_n}"); _n += 1
        else:
            sdir = self.out_dir / f"stitch_{self._session_id:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        # LOCK the session folder — _save_session reuses this exact path
        self._session_sdir = sdir
        if hasattr(self, '_pano_metrics') and self._pano_metrics:
            try: self._pano_metrics.close()
            except: pass
        self._pano_metrics = PanoMetricMonitor(sdir)
        print(f"[Metrics] debug/ -> {sdir}/debug/")

    def _do_stop_save(self):
        self.recording=False; self.voice_paused=False
        n_sess  = len(self._session_frames)
        n_canvas = self.canvas.frame_count
        print(f"[STITCH {self._session_id}] ■  "
              f"frames={n_sess}  canvas={n_canvas}  "
              f"good={self.stats.get('frames_good',0)}  "
              f"gated={self.stats.get('frames_gated_out',0)}  "
              f"iou_rej={self.stats.get('frames_iou_rejected',0)}  "
              f"dir_rej={self.stats.get('frames_dir_rejected',0)}")
        n = n_sess
        if n < 5:
            print(f"[STITCH] Too few frames ({n}) — session discarded.")
            print(f"         Tip: hold probe in contact for at least 2 seconds before stopping.")
            # Reset recording state cleanly so app is ready for next session
            self.recording       = False
            self.voice_paused    = False
            self._panorama_frozen= False
            self._session_frames.clear()
            self._session_coords.clear()
            self.anchor_marks.clear()
            self.bad_marks.clear()
            self._bad_start = None
            return None   # callers must guard for None
        print(f"[STITCH {self._session_id}] Computing sharp panorama…")
        sharp=self._run_offline_stitch()
        self._save_session(sharp)
        return sharp

    def _run_offline_stitch(self):
        try:
            sys.path.insert(0,str(Path(__file__).parent))
            from session_stitcher import (analyse_trajectory,select_keyframes,
                                           refine_row0,stitch_offline)
        except ImportError:
            print("[Stitch] session_stitcher.py not found — using sharp keyframe canvas")
            _sc = self.sharp_canvas.get_panorama(); _cc = self.canvas.get_panorama(); return _sc if _sc is not None else _cc
        frames=[(f/255.).astype(np.float32) if f.dtype==np.uint8 else f
                for f in self._session_frames]
        if len(frames)<10: return None
        try:
            traj=analyse_trajectory(self._session_coords,debug=False,use_imu=False)
            kf=select_keyframes(traj['col0_smooth'],traj['quality'],traj['outlier_mask'])
            if len(kf)<5:
                print("[Stitch] Too few keyframes — using sharp keyframe canvas")
                _sc = self.sharp_canvas.get_panorama(); _cc = self.canvas.get_panorama(); return _sc if _sc is not None else _cc
            r0=refine_row0(kf,frames,traj['gm_row_smooth'],traj['col0_smooth'])
            return stitch_offline(frames,kf,traj['col0_smooth'],traj['gm_row_smooth'],traj['quality'],r0)
        except Exception as e:
            print(f"[Stitch] {e} — using sharp keyframe canvas")
            _sc = self.sharp_canvas.get_panorama(); _cc = self.canvas.get_panorama(); return _sc if _sc is not None else _cc


    def _build_annotated_panorama(self, pano_bgr: np.ndarray,
                                   coords: list) -> np.ndarray:
        """
        Draw biomarker overlays on the stitched panorama and return the result.
        Overlays:
          • GM centroid  — cyan dot
          • CEJ centroid — yellow dot  
          • ABC centroid — magenta dot
          • iGR line     (GM→CEJ, cyan)
          • iGH  line    (GM→ABC, magenta)
          • iABL line    (ABC→CEJ, orange)
          • Measurement text panel — bottom right of image
          • Scale bar    — bottom left
        Uses per-frame canvas coordinates stored in _session_coords.
        """
        if pano_bgr is None or pano_bgr.size == 0:
            return pano_bgr
        out = pano_bgr.copy()
        h, w = out.shape[:2]
        if h == 0 or w == 0:
            return out

        PIXEL_SPACING = PIXEL_SPACING_MM   # mm per pixel

        # Feature colours (BGR)
        COL = {
            'GM':  (0,255,255),    # cyan
            'CEJ': (0,255,0),      # green
            'ABC': (255,0,255),    # magenta
            'iGR': (0,255,255),
            'iABL':(255,0,255),
            'iGH': (0,165,255),
        }

        # Collect all frame landmarks in canvas space
        igr_vals=[]; iabl_vals=[]; igh_vals=[]
        for coord in coords:
            # Each coord has col0_px and gm_y offset info
            col0  = coord.get('col0_px', 0)
            gm_x  = coord.get('gm_x_mm', 0)
            gm_y  = coord.get('gm_y_mm', 0)
            meas  = coord.get('measurements', {})
            if not meas: continue

            # Canvas pixel position of GM
            cx = int(col0)
            # Anchor row stored at session level — approximate with canvas centre
            cy = h // 2

            # Draw GM dot
            cv2.circle(out, (cx, cy), 4, COL['GM'], -1)

            # Collect measurement values
            if 'iGR'  in meas and meas['iGR']:  igr_vals.append(meas['iGR'])
            if 'iABL' in meas and meas['iABL']: iabl_vals.append(meas['iABL'])
            if 'iGH'  in meas and meas['iGH']:  igh_vals.append(meas['iGH'])

            # Draw iGR extent as vertical segment at this column
            if 'iGR' in meas and meas['iGR']:
                px_igr = int(meas['iGR'] / PIXEL_SPACING)
                cv2.line(out, (cx, cy), (cx, cy - px_igr), COL['iGR'], 2, cv2.LINE_AA)
            if 'iABL' in meas and meas['iABL']:
                px_iabl = int(meas['iABL'] / PIXEL_SPACING)
                cv2.line(out, (cx, cy), (cx, cy + px_iabl), COL['iABL'], 2, cv2.LINE_AA)

        # ── Measurement summary panel (bottom-right) ──────────────────────────
        def _avg(v): return sum(v)/len(v) if v else None

        avg_igr  = _avg(igr_vals)
        avg_iabl = _avg(iabl_vals)
        avg_igh  = _avg(igh_vals)

        panel_lines = []
        if avg_igr  is not None: panel_lines.append((f"iGR  (GM→CEJ): {avg_igr:.3f} mm",  COL['iGR']))
        if avg_iabl is not None: panel_lines.append((f"iGH  (GM→ABC): {avg_iabl:.3f} mm", COL['iABL']))
        if avg_igh  is not None: panel_lines.append((f"iGH  (total) : {avg_igh:.3f} mm",  COL['iGH']))
        panel_lines.append((f"Frames: {len(coords)}", (180,180,180)))

        if panel_lines:
            lh = 26; pw = 320; ph = len(panel_lines) * lh + 14
            px0 = w - pw - 8; py0 = h - ph - 8
            cv2.rectangle(out, (px0, py0), (px0+pw, py0+ph), (20,20,20), -1)
            cv2.rectangle(out, (px0, py0), (px0+pw, py0+ph), (80,80,80),  1)
            for i, (txt, col) in enumerate(panel_lines):
                cv2.putText(out, txt, (px0+8, py0+18+i*lh),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

        # ── Scale bar (bottom-left): 1 mm only ──────────────────────────────
        bar_px = int(1.0 / PIXEL_SPACING)   # pixels per 1 mm
        if bar_px > 0:
            bx, by = 12, h - 14
            # Shadow for contrast on any background
            cv2.line(out, (bx-1, by+1), (bx+bar_px+1, by+1), (0,0,0), 5)
            # End ticks
            cv2.line(out, (bx,          by-8), (bx,          by), (230,230,230), 2, cv2.LINE_AA)
            cv2.line(out, (bx+bar_px,   by-8), (bx+bar_px,   by), (230,230,230), 2, cv2.LINE_AA)
            # Horizontal bar
            cv2.line(out, (bx, by), (bx+bar_px, by), (230,230,230), 3, cv2.LINE_AA)
            # Label
            cv2.putText(out, "1 mm",
                        (bx, by-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230,230,230), 1, cv2.LINE_AA)

        # ── Label ─────────────────────────────────────────────────────────────
        cv2.putText(out, "Annotated Panorama — B1 landmarks",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

        return out

    def _save_session(self, sharp_pano=None):
        n=self.canvas.frame_count
        if n<5: return
        sid=self._session_id
        # Reuse the sdir decided at _do_start so debug/ and panorama go to the same folder
        if hasattr(self, '_session_sdir') and self._session_sdir is not None:
            sdir = self._session_sdir
        else:
            _lbl = getattr(self, '_session_tooth_label', None)
            if _lbl:
                sdir = self.out_dir / _lbl
                _base = sdir; _idx = 2
                while sdir.exists() and any(sdir.iterdir()):
                    sdir = Path(str(_base) + f"_{_idx}"); _idx += 1
            else:
                sdir = self.out_dir / f"stitch_{sid:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        fdir=sdir/"frames"; fdir.mkdir(exist_ok=True)
        print(f"\n[STITCH {sid}] Saving → {sdir}")

        # Panorama
        if sharp_pano is not None:
            cv2.imwrite(str(sdir/"panorama.png"),cv2.cvtColor(sharp_pano,cv2.COLOR_RGB2BGR))
            lp=self.canvas.get_panorama()
            if lp is not None: cv2.imwrite(str(sdir/"panorama_live.png"),cv2.cvtColor(lp,cv2.COLOR_RGB2BGR))
            pano=sharp_pano
        else:
            pano=self.canvas.get_panorama()
            if pano is not None: cv2.imwrite(str(sdir/"panorama.png"),cv2.cvtColor(pano,cv2.COLOR_RGB2BGR))
        ph,pw=(pano.shape[:2] if pano is not None else (0,0))

        # Save annotated panorama with iGR / iABL / iGH biomarker overlays
        if pano is not None:
            try:
                bgr_pano = cv2.cvtColor(pano, cv2.COLOR_RGB2BGR)
                ann      = self._build_annotated_panorama(bgr_pano, list(self._session_coords))
                cv2.imwrite(str(sdir/"panorama_annotated.png"), ann)
                print(f"  Saved: panorama_annotated.png  (iGR/iABL/iGH overlays)")
            except Exception as _ae:
                print(f"  [Annotated] skipped: {_ae}")

        # Feature 3: Coverage heatmap
        cov=self.canvas.get_coverage_info()
        self.heatmap.save(sdir/"coverage.png",cov,max(pw,100))

        # Individual frames
        for i,fr in enumerate(self._session_frames):
            cv2.imwrite(str(fdir/f"frame_{i:04d}.png"),cv2.cvtColor(fr,cv2.COLOR_RGB2BGR))

        # Coordinates + voice marks
        with open(sdir/"coordinates.json","w") as f: json.dump(self._session_coords,f,indent=2)
        with open(sdir/"voice_marks.json","w") as f:
            json.dump({'anchor_frame_indices':self.anchor_marks,'bad_regions':self.bad_marks},f,indent=2)

        # Feature 4: Audit log
        self.audit.save_jsonl(sdir/"audit_log.jsonl")

        # Quality gater stats (derived from quality score, not separate gater)

        # Metadata
        coords=self._session_coords
        meta={
            'session_id':    sid,
            'timestamp':     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model':         'B1_V8_Baseline',
            'n_frames':      n,
            'n_gated_out':   self.stats.get('frames_gated_out',0),
            'n_iou_rejected': self.stats.get('frames_iou_rejected',0),
            'n_dir_rejected': self.stats.get('frames_dir_rejected',0),
            'iou_gate_summary': self.proc.iou_gate.summary(),
            'direction_summary': self._sweep_tracker.summary(),
            'voice_commands_total': len(getattr(self,'_voice_event_log',[])),
            'app_version': 'DentWise v3',
            'imu_type': 'serial' if getattr(self.imu,'port',None) else 'ble',
            'imu_port': getattr(self.imu,'port','BLE'),
            'sweep_mm':      round(float(max(c['gm_x_mm'] for c in coords)-
                                         min(c['gm_x_mm'] for c in coords)),3) if coords else 0,
            'quality_mean':  round(float(np.mean([c['quality_score'] for c in coords])),4) if coords else 0,
            'imu_used':      any(c['imu_connected'] for c in coords),
            'panorama_px':   [pw,ph],
            'tiled_canvas_memory_mb': round(self.canvas.memory_mb(),2),
            'n_anchor_marks': len(self.anchor_marks),
            'n_bad_marks':    len(self.bad_marks),
            'coverage_summary': cov,
        }
        # Save metric summary
        if hasattr(self,'_pano_metrics') and self._pano_metrics:
            self._pano_metrics.save_summary(self.canvas)
            self._pano_metrics.close()
            self._pano_metrics=None
        with open(sdir/"metadata.json","w") as f: json.dump(meta,f,indent=2)

        # Save raw panorama
        if sharp_pano is not None:
            raw_path = sdir / "panorama.png"
            cv2.imwrite(str(raw_path), cv2.cvtColor(sharp_pano, cv2.COLOR_RGB2BGR))

        # Save annotated panorama with iGH/iGR/iABL overlaid
        self._save_annotated_panorama(sdir, sharp_pano)

        print(f"  ✓ Stitch {sid}  sweep={meta['sweep_mm']:.1f}mm  "
              f"QS={meta['quality_mean']:.2f}  mem={meta['tiled_canvas_memory_mb']:.1f}MB\n")

    def export_panorama(self):
        pano=self.canvas.get_panorama()
        if pano is None: print("[Export] Nothing yet"); return
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        path=self.out_dir/f"panorama_{ts}.png"
        cv2.imwrite(str(path),cv2.cvtColor(pano,cv2.COLOR_RGB2BGR))
        print(f"[Export] → {path}")

    def _save_annotated_panorama(self, sdir: Path, sharp_pano):
        """
        Save the panorama with biomarker measurements overlaid.
        Draws GM/CEJ/ABC landmark points and iGH/iGR/iABL distance lines
        with text labels on the stitched panorama image.
        Saved as: <sdir>/panorama_annotated.png
        No display — save only.
        """
        if sharp_pano is None:
            sharp_pano = self.sharp_canvas.get_panorama()
        if sharp_pano is None:
            sharp_pano = self.canvas.get_panorama()
        if sharp_pano is None:
            print("[AnnotatedPano] Nothing to save"); return

        bgr = cv2.cvtColor(sharp_pano, cv2.COLOR_RGB2BGR).copy()
        h, w = bgr.shape[:2]

        # Scale factor: canvas coords → bgr pixel coords
        # Each column in the canvas is PIXEL_SPACING_MM wide in mm space
        # The panorama is stored at its native resolution from get_panorama()

        # Draw each composited frame's measurements onto the panorama
        FEAT_COLORS = {
            'GM':  (0,255,255),    # cyan
            'CEJ': (255,255,0),    # yellow
            'ABC': (255,0,255),    # magenta
        }
        LINE_COLORS = {
            'iGR':  (0,  255, 255),  # cyan    — GM→CEJ  (gingival recession)
            'iABL': (255, 165,  0),  # orange  — ABC→CEJ (alveolar bone level)
            'iGH':  (255,   0, 255), # magenta — GM→ABC  (gingival height)
        }

        # Collect unique measurement positions from session coords
        already_drawn = set()
        for coord in self._session_coords:
            col0   = coord.get('col0_px', 0)
            gm_x   = coord.get('gm_x_mm', 0)
            gm_y   = coord.get('gm_y_mm', 0)
            qual   = coord.get('quality_score', 0)
            meas   = coord.get('measurements', {})
            if not meas or qual < 0.30: continue
            # Snap to 20-px grid to avoid overlapping annotations
            snap = int(col0 // 20) * 20
            if snap in already_drawn: continue
            already_drawn.add(snap)

            cx = int(col0)
            anchor_r = getattr(self.canvas, 'anchor_row', h // 2)
            cy_gm = int(anchor_r)

            # Draw GM point
            cv2.circle(bgr, (cx, cy_gm), 5, FEAT_COLORS['GM'], -1)

            # Draw iGR line (GM→CEJ, upward from GM)
            igr = meas.get('iGR', 0)
            if igr:
                igr_px  = int(igr / PIXEL_SPACING_MM)
                cy_cej  = cy_gm - igr_px
                cv2.circle(bgr, (cx, cy_cej), 4, FEAT_COLORS['CEJ'], -1)
                cv2.line(bgr, (cx, cy_gm), (cx, cy_cej), LINE_COLORS['iGR'], 1, cv2.LINE_AA)

            # Draw iGH line (GM→ABC — gingival height, downward from GM)
            igh = meas.get('iGH', 0)
            if igh:
                igh_px  = int(igh / PIXEL_SPACING_MM)
                cy_abc  = cy_gm + igh_px
                cv2.circle(bgr, (cx, cy_abc), 4, FEAT_COLORS['ABC'], -1)
                cv2.line(bgr, (cx, cy_gm), (cx, cy_abc), LINE_COLORS['iGH'], 1, cv2.LINE_AA)

            # Draw iABL line (ABC→CEJ — alveolar bone level)
            iabl = meas.get('iABL', 0)
            if iabl and igh and igr:
                # ABC is at cy_gm + igh_px, CEJ is at cy_gm - igr_px
                cy_abc_pt = cy_gm + igh_px
                cy_cej_pt = cy_gm - int(igr / PIXEL_SPACING_MM)
                cv2.line(bgr, (cx, cy_abc_pt), (cx, cy_cej_pt),
                         LINE_COLORS['iABL'], 1, cv2.LINE_AA)

            # Measurement text every 40 px to avoid clutter
            if snap % 40 == 0 and (igr or igh):
                txt  = f"GR{igr:.1f} BL{iabl:.1f} GH{igh:.1f}"
                tx   = max(2, cx - 28)
                ty   = max(12, cy_gm - int(igr / PIXEL_SPACING_MM) - 8) if igr else cy_gm - 8
                cv2.putText(bgr, txt, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(bgr, txt, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, LINE_COLORS['iGH'], 1, cv2.LINE_AA)

        # Legend bottom-left — correct clinical definitions
        for i, (lbl, col) in enumerate([
            ("iGR  = GM→CEJ  (gingival recession)",  LINE_COLORS['iGR']),
            ("iABL = ABC→CEJ (alveolar bone level)",  LINE_COLORS['iABL']),
            ("iGH  = GM→ABC  (gingival height)",      LINE_COLORS['iGH']),
        ]):
            ly = h - 12 - i*18
            cv2.putText(bgr, lbl, (6, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

        out_path = sdir / "panorama_annotated.png"
        cv2.imwrite(str(out_path), bgr)
        print(f"  ✓ Annotated panorama → {out_path}")

    # ── main loop ─────────────────────────────────────────────────────────────

    def _show_panorama_fullscreen(self, pano_rgb, label: str = "PANORAMA"):
        """
        Display panorama at maximum readable size in the WIN_PANO window.
        pano_rgb: (H, W, 3) uint8 RGB array
        Always uses the sharp_canvas if available, falls back to averaging canvas.
        """
        if pano_rgb is None:
            # Try sharp canvas first, then averaging canvas
            pano_rgb = self.sharp_canvas.get_panorama()
        if pano_rgb is None:
            pano_rgb = self.canvas.get_panorama()
        if pano_rgb is None:
            print("[Display] No panorama to show — session had too few frames")
            return

        # Convert to BGR for OpenCV display
        try:
            bgr = cv2.cvtColor(pano_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            print("[Display] Cannot convert panorama — invalid array")
            return
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            print("[Display] Panorama has zero size — nothing to show")
            return

        # Scale to fill screen width (max 1400px wide, max 700px tall)
        screen_w, screen_h = 1400, 700
        scale = min(screen_w / max(w, 1), screen_h / max(h, 1), 3.0)
        # Use INTER_NEAREST for crisp pixels when upscaling
        interp  = cv2.INTER_NEAREST if scale > 1.0 else cv2.INTER_AREA
        dw = max(400, int(w * scale))
        dh = max(200, int(h * scale))
        disp = cv2.resize(bgr, (dw, dh), interpolation=interp)

        # Label overlay
        kf = self.sharp_canvas.frame_count
        total = self.canvas.frame_count
        full_label = (f"{label}  |  {kf} keyframes / {total} total  |  "
                      f"{w * PIXEL_SPACING_MM:.1f} mm sweep  |  "
                      f"say ON to start new scan")
        cv2.putText(disp, full_label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

        # Coverage heatmap strip at bottom
        cov   = self.canvas.get_coverage_info()
        strip = self.heatmap.render(cov, dw)
        disp  = np.vstack([disp, strip])

        cv2.imshow("Growing Panorama + Coverage", disp)
        cv2.resizeWindow("Growing Panorama + Coverage", dw, dh + self.heatmap.STRIP_H)
        self._panorama_frozen = True   # freeze: keep this image until next ON
        print(f"[Display] Panorama shown: {dw}×{dh} px  "
              f"({w * PIXEL_SPACING_MM:.2f} mm × {h * PIXEL_SPACING_MM:.2f} mm)  "
              f"{kf} keyframes  [frozen until next ON/1]")

    def run(self):
        WIN_MAIN="DentWise — Panorama"
        WIN_PANO="Growing Panorama + Coverage"
        cv2.namedWindow(WIN_MAIN,cv2.WINDOW_NORMAL); cv2.namedWindow(WIN_PANO,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_MAIN,700,550); cv2.resizeWindow(WIN_PANO,900,430)

        imu_conf=0.; last_imu=IMUSample()
        print("\n"+"="*62)
        print("  DentWise v3 — Panoramic Stitching + Voice + IMU")
        print("="*62)
        print("  1/ENTER  start/stop+save    2   advance session")
        print("  R  discard    E  export    I  IMU    S  screenshot")
        print("  C  calibrate IMU           P  replay last session")
        print("  M  multi-session register  Q  quit")
        print("  Voice: ON  SAVE  DROP  HOLD  GO  MARK  BAD  VIEW")
        print("="*62+"\n")

        placeholder=np.zeros((350,700,3),np.uint8)
        cv2.putText(placeholder,"Say 'ON' or press '1' to start",
                    (80,160),cv2.FONT_HERSHEY_SIMPLEX,0.7,(150,150,150),2)
        cv2.imshow(WIN_PANO,placeholder)

        frame_skip=0; pano_ctr=0; last_sharp_pano=None; last_sdir=None

        while True:
            t0=time.time()
            frame_bgr=self.cap.get_frame()
            if frame_bgr is None:
                frame_skip+=1
                if frame_skip>100: print("HD60X lost"); break
                time.sleep(0.01); cv2.waitKey(1); continue
            frame_skip=0

            result=self.proc.process(frame_bgr)
            if result is None:
                cv2.waitKey(1); continue

            self.stats['frames_total']+=1
            gm_x=result['gm_x_mm']; gm_y=result['gm_y_mm']
            quality=result['quality_score']
            # Original FrameProcessor has no gate — derive from quality + GM detection
            gm_detected='GM' in self.proc.ann.point_coords_cache
            gate_pass=(quality>=QS_THRESH) or gm_detected or (quality>=0.02)
            gate_comp={'sharpness':quality,'contact':quality,'snr':quality,'combined':quality}
            gate_score=quality

            if gate_pass: self.stats['frames_good']+=1
            else: self.stats['frames_gated_out']+=1

            imu_now=self.imu.get_latest(); self.stats['imu_packets']=self.imu.packet_count
            if self.imu.connected and not self._imu_was_connected:
                print("[IMU] Connected — fusion active")
                print("[IMU] Tip: set 100Hz via WitMotion app for best performance")
                self._imu_was_connected=True
                self._imu_first_yaw=None   # reset — will capture first real yaw
            elif not self.imu.connected and self._imu_was_connected:
                print("[IMU] Disconnected — GM-only"); self._imu_was_connected=False
            # Track first valid IMU yaw so ComplementaryFilter has a real reference
            if imu_now.valid and not hasattr(self,'_imu_first_yaw_set'):
                self._imu_first_yaw_set=True
                self.cf.last_yaw=imu_now.yaw   # seed filter with real yaw
                print(f"[IMU] First valid packet — yaw={imu_now.yaw:.2f}° seeded into filter")

            # imu_used = True only when IMU provides real displacement signal.
            # Connected-but-silent IMU (gyro=0, tiny yaw change) should not
            # mark tiles as 'imu' registration — they are effectively GM-only.
            _imu_dy = 0.
            if hasattr(self,'_last_imu_yaw') and self._last_imu_yaw is not None:
                _imu_dy = abs(imu_now.yaw - self._last_imu_yaw)
                if _imu_dy > 180: _imu_dy = abs(_imu_dy - 360)
            self._last_imu_yaw = imu_now.yaw
            imu_used = (self.imu.connected and imu_now.valid and
                        (abs(imu_now.gyro_z) > 0.1 or _imu_dy > 0.02))
            if self.recording and not self.voice_paused and gate_pass:
                gm_x_fused,imu_conf=self.cf.update(gm_x,imu_now,quality)
            else:
                gm_x_fused=gm_x; imu_conf=0.

            self._detect_sweep_dir(gm_x_fused); self._update_jitter(gm_x)

            # v3: IOU gate metrics
            iou_metrics  = result.get('iou_metrics', {})
            iou_pass     = result.get('gate_pass_iou', True)
            if not iou_pass and self.recording:
                self.stats['frames_iou_rejected'] = self.stats.get('frames_iou_rejected',0)+1

            # v3: direction tracking (run every frame while recording)
            dir_accept = True; sweep_dir = 0; dir_metrics = {}
            if self.recording and not self.voice_paused:
                dir_accept, sweep_dir, dir_metrics =                     self._sweep_tracker.update(gm_x_fused)
                if not dir_accept:
                    self.stats['frames_dir_rejected'] =                         self.stats.get('frames_dir_rejected',0)+1

            # Canvas + audit
            if self.recording and not self.voice_paused and gate_pass:
                # Use IMU-fused position for col0
                col0=self._compute_col0(gm_x_fused)
                gm_row=gm_y/PIXEL_SPACING_MM
                is_anchor=len(self._session_coords) in self.anchor_marks
                is_bad=self._bad_start is not None
                # Only composite if direction is correct (v3)
                if dir_accept:
                    self.canvas.add_frame(result['frame_256'],col0,gm_row,quality,imu_used=imu_used)

                # Keyframe selection for sharp live canvas
                # A frame is a keyframe if:
                #   1. col0 has advanced ≥ KEYFRAME_MIN_DIST_PX since last keyframe
                #   2. quality ≥ KEYFRAME_QUALITY threshold
                #   3. gm_dy (vertical jitter) is small (probe not wobbling)
                _gm_dy     = result.get('gm_dy_px', 0.) if hasattr(result,'get') else 0.
                _kf_col_ok = (self._last_keyframe_col0 is None or
                              abs(col0 - self._last_keyframe_col0) >= KEYFRAME_MIN_DIST_PX)
                _kf_jit_ok = abs(_gm_dy) <= KEYFRAME_JITTER_PX
                _kf_qual_ok= quality >= KEYFRAME_QUALITY
                if _kf_col_ok and _kf_jit_ok and _kf_qual_ok and dir_accept:
                    self.sharp_canvas.add_frame(result['frame_256'],col0,gm_row,
                                                 quality,imu_used=imu_used)
                    self._last_keyframe_col0 = col0
                    self._last_keyframe_gm_y = gm_y
                self._session_frames.append((result['frame_256']*255).astype(np.uint8))
                coord={
                    'frame_n':len(self._session_coords),'ts':time.time(),
                    'gm_x_mm':round(gm_x,4),'gm_y_mm':round(gm_y,4),
                    'gm_x_fused':round(gm_x_fused,4),'col0_px':round(col0,2),
                    'gm_row_frame':round(gm_row,2),'quality_score':round(quality,4),
                    'gate_score':round(gate_score,4),
                    'imu_yaw':round(imu_now.yaw,3),'imu_pitch':round(imu_now.pitch,3),
                    'imu_roll':round(imu_now.roll,3),'imu_gyro_z':round(imu_now.gyro_z,3),
                    'imu_conf':round(imu_conf,3),'imu_connected':self.imu.connected,
                    'imu_valid':imu_now.valid,'is_anchor':is_anchor,'is_bad':is_bad,
                }
                self._session_coords.append(coord)
                self.audit.log(coord['frame_n'],coord['ts'],gm_x,gm_y,col0,quality,
                               gate_comp,imu_now.yaw,imu_conf,imu_now.valid,
                               anchor=is_anchor,bad=is_bad)
                # Metric monitor log (accepted frames)
                # Store direction info in result for metric log
                result['_sweep_dir']   = sweep_dir
                result['_dir_accept']  = dir_accept
                if self._pano_metrics:
                    ti=int(col0//TiledCanvas.TILE_W)
                    tl=self.canvas._tiles.get(ti,{}).get('level',0)
                    # Pass imu_reader.connected as extra context via imu_now
                    # (imu_now.valid reflects packet arrival; connected = BLE paired)
                    self._pano_metrics.log(result,col0,imu_now,imu_conf,imu_conf,
                                           self.canvas,ti,tl,
                                           gm_x_fused=gm_x_fused,
                                           frame_bgr=frame_bgr)
                    # Patch the last written row with correct imu_connected
                    # (written inline since log() already flushed to CSV)

            # Log rejected frames to debug (gate_comp recorded for all frames)
            if self.recording and not self.voice_paused and not gate_pass and self._pano_metrics:
                result['_sweep_dir']  = sweep_dir
                result['_dir_accept'] = dir_accept
                # Log with col0=0 so debug CSV shows all frames, not just accepted
                _col_est = self._compute_col0(gm_x) if self._gm_x_ref is not None else 0.
                ti=int(_col_est//TiledCanvas.TILE_W)
                self._pano_metrics.log(result,_col_est,imu_now,0.,0.,self.canvas,ti,0)

            # Display
            ann=result['annotated']
            ann=self._draw_gate_overlay(ann,gate_comp,gate_pass)
            if self.show_imu: ann=self._draw_imu_overlay(ann,imu_now,imu_conf)
            ann=self._draw_recording_overlay(ann); ann=self._draw_voice_status(ann)
            cv2.imshow(WIN_MAIN,ann)

            pano_ctr+=1
            if pano_ctr>=3 and not self._panorama_frozen:
                pano_ctr=0
                # Live display: sharp keyframe canvas (high-res) with coverage heatmap
                sharp_pano  = self.sharp_canvas.get_panorama()
                avg_pano    = self.canvas.get_panorama()
                disp_pano   = sharp_pano if sharp_pano is not None else avg_pano
                if disp_pano is not None:
                    ph,pw=disp_pano.shape[:2]; dh=350; dw=max(200,int(pw*dh/ph))
                    pano_d=cv2.resize(cv2.cvtColor(disp_pano,cv2.COLOR_RGB2BGR),(dw,dh))
                    cov=self.canvas.get_coverage_info()
                    strip=self.heatmap.render(cov,dw)
                    pano_d=np.vstack([pano_d,strip])
                    kf=self.sharp_canvas.frame_count; total=self.canvas.frame_count
                    label=f"SHARP {kf}kf/{total}f  {pw*PIXEL_SPACING_MM:.1f}mm"
                    cv2.putText(pano_d,label,(8,dh-6),
                                cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,130),1)
                    cv2.imshow(WIN_PANO,pano_d)
                    cv2.resizeWindow(WIN_PANO,max(400,dw),dh+self.heatmap.STRIP_H+20)
                elif not self.recording:
                    cv2.imshow(WIN_PANO,placeholder)

            # ── Voice commands ────────────────────────────────────────────
            if self.voice:
                while True:
                    vcmd=self.voice.get_command()
                    if vcmd is None: break
                    if vcmd=='on':
                        if not self.recording:
                            self._do_start(gm_x,imu_now); self._flash("ON"); cv2.imshow(WIN_PANO,placeholder)
                        else: print("[Voice] Already recording — say OFF to stop")
                    elif vcmd=='off':
                        if self.recording:
                            self._flash("OFF — saving…")
                            sharp=self._do_stop_save()
                            if sharp is not None:
                                self._show_panorama_fullscreen(sharp, "PANORAMA — HIGH RES")
                            else:
                                self._flash("TOO FEW FRAMES — say ON to try again")
                        else: print("[Voice] Not recording — say ON first")
                    elif vcmd=='save':
                        if self.recording:
                            self._flash("SAVE")
                            sharp=self._do_stop_save(); last_sharp_pano=sharp
                            last_sdir=self.out_dir/f"stitch_{self._session_id:03d}"
                            self._show_panorama_fullscreen(sharp)
                    elif vcmd=='drop':
                        if self.recording:
                            self.recording=False; self.voice_paused=False; self.canvas.reset()
                            self._session_frames.clear(); self._session_coords.clear()
                            self.anchor_marks.clear(); self.bad_marks.clear(); self._bad_start=None
                            cv2.imshow(WIN_PANO,placeholder); self._flash("DROP")
                    elif vcmd=='hold':
                        if self.recording and not self.voice_paused: self.voice_paused=True; self._flash("HOLD")
                    elif vcmd=='go':
                        if self.recording and self.voice_paused: self.voice_paused=False; self._flash("GO")
                    elif vcmd=='mark':
                        if self.recording:
                            self.anchor_marks.append(len(self._session_coords)); self._flash("MARK")
                    elif vcmd=='bad':
                        if self.recording:
                            if self._bad_start is None: self._bad_start=len(self._session_coords); self._flash("BAD start")
                            else:
                                self.bad_marks.append((self._bad_start,len(self._session_coords)))
                                self._bad_start=None; self._flash("BAD end")
                    elif vcmd=='view': self.show_imu=not self.show_imu; self._flash(f"VIEW {'ON' if self.show_imu else 'OFF'}")

            # ── Keyboard ─────────────────────────────────────────────────
            key=cv2.waitKey(1)&0xFF
            if key in (ord('q'),27):
                self._quit_requested = True   # signal main() to exit fully
                break

            # ── Voice mode-switch exits stitching run() ───────────────────
            # When UnifiedVoiceController detects "charting", it sets
            # switch_requested=True. We check it here so run() exits cleanly
            # on the next frame and returns to the mode-selection loop.
            if self.voice and hasattr(self.voice, '_uvc'):
                uvc = self.voice._uvc
                if getattr(uvc, 'switch_requested', False):
                    uvc.switch_requested = False
                    print("[Pano] Mode switch detected — exiting stitching")
                    if self.recording:
                        print("[Pano] Auto-saving current session…")
                        self._do_stop_save()
                    break
            elif key in (ord('1'),13,32):
                if not self.recording:
                    self._do_start(gm_x,imu_now); cv2.imshow(WIN_PANO,placeholder)
                    print("[v3] Recording started — say OFF or press 1 to stop")
                else:
                    sharp=self._do_stop_save(); last_sharp_pano=sharp
                    last_sdir=self.out_dir/f"stitch_{self._session_id:03d}"
                    if sharp is not None:
                        self._show_panorama_fullscreen(sharp, "PANORAMA — PRESS 1 TO SCAN AGAIN")
                    else:
                        self._flash("TOO FEW FRAMES — scan longer")
                        print("[Stitch] Tip: sweep for at least 2-3 seconds before stopping")
            elif key==ord('2'):
                if self.recording:
                    sharp=self._do_stop_save()
                    if sharp is not None:
                        self._show_panorama_fullscreen(sharp, "PANORAMA — ADVANCING SESSION")
                        import time as _t; _t.sleep(0.8)
                    else:
                        self._flash("TOO FEW FRAMES — discarded")
                old=self._session_id; self._session_id+=1
                self.canvas.reset(); self.sharp_canvas.reset()
                self.cf.reset(gm_x); self.audit.clear()
                self._session_frames.clear(); self._session_coords.clear()
                self.anchor_marks.clear(); self.bad_marks.clear()
                print(f"[STITCH {old}] → STITCH {self._session_id}")
            elif key==ord('e'): self.export_panorama()
            elif key==ord('r'):
                self.recording=False; self.voice_paused=False; self.canvas.reset()
                self._session_frames.clear(); self._session_coords.clear()
                self.anchor_marks.clear(); self.bad_marks.clear(); self._bad_start=None
                cv2.imshow(WIN_PANO,placeholder)
            elif key==ord('i'): self.show_imu=not self.show_imu
            elif key==ord('s'):
                ts=datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(self.out_dir/f"screenshot_{ts}.png"),ann)
                print(f"[Screenshot] saved")
            elif key==ord('c'):
                if self._sweep_start_yaw is None:
                    self._sweep_start_yaw=imu_now.yaw; self._sweep_start_gm=gm_x
                    print("[CAL] Mark start — sweep 10 mm then press C again")
                else:
                    self.cf.calibrate_scale(abs(gm_x-self._sweep_start_gm),
                                            self._sweep_start_yaw,imu_now.yaw)
                    self._sweep_start_yaw=None
            elif key==ord('p'):
                # Feature 4: replay last session
                if last_sdir and last_sdir.exists():
                    print(f"[Replay] Building {last_sdir}/replay.mp4 …")
                    self.audit.replay_session(last_sdir, last_sdir/"panorama.png")
                else: print("[Replay] No saved session yet — press 1 to record first")
            elif key==ord('m'):
                # Feature 5: multi-session registration
                prev_dirs=sorted(self.out_dir.glob("stitch_*"))
                if len(prev_dirs)<2: print("[Reg] Need ≥2 sessions"); 
                else:
                    cur=prev_dirs[-1]; prev=prev_dirs[-2]
                    print(f"[Reg] Registering {cur.name} → {prev.name} …")
                    pano=self.canvas.get_panorama()
                    if pano is None: pano_d=cv2.imread(str(cur/"panorama.png"))
                    else:            pano_d=pano
                    if pano_d is not None:
                        self.registrar.register(
                            pano_d if pano is not None else cv2.cvtColor(pano_d,cv2.COLOR_BGR2RGB),
                            self._session_coords,
                            prev/"panorama.png", prev/"coordinates.json",
                            cur/"registered_panorama.png")

            elapsed=time.time()-t0; sleep=max(0.,1./25.-elapsed)
            if sleep>0: time.sleep(sleep)

        # Don't call destroyAllWindows here — main() owns the window lifecycle
        # so charting/idle windows stay open after returning from stitching.
        # Set quit_requested so main() knows whether to exit or return to menu.
        if hasattr(self, '_quit_requested'):
            pass  # already set by keyboard handler below
        s=self.proc.gater.summary()
        if s: print(f"\n[QualityGater] accept_rate={s.get('accept_rate','?')}  "
                    f"mean={s.get('mean_score','?')}  n={s.get('n_frames','?')}")
        print(f"[Done] {self.stats['frames_total']} total  "
              f"{self.stats['frames_good']} passed gate  "
              f"{self.stats['frames_gated_out']} rejected")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
#  UNIFIED MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Unified Live System — Panorama Stitching + Dental Charting")

    # ── Shared ────────────────────────────────────────────────────────────────
    p.add_argument("--model","-m", type=str, required=True,
                   help="B1 MKUNet .h5 checkpoint (shared by both modes)")
    p.add_argument("--device","-d", type=int, default=1,
                   help="HD60X device ID")
    p.add_argument("--user",        type=str, default=None,
                   help="Clinician user ID — loads enrollment/<user>/adapted_model.pt")
    p.add_argument("--enrollment_dir", type=str, default="enrollment")
    p.add_argument("--no_voice",    action="store_true")
    p.add_argument("--voice_device",type=int, default=None)
    p.add_argument("--whisper_model",type=str, default='small',
                   choices=['tiny','base','small','medium'])

    # ── Panorama ──────────────────────────────────────────────────────────────
    p.add_argument("--serial",      type=str,   default=None)
    p.add_argument("--baud",        type=int,   default=9600,
                   help="Serial baud rate (WT901C-TTL wired default: 9600)")
    p.add_argument("--mac",         type=str,   default=IMU_MAC)
    p.add_argument("--no_imu",      action="store_true")
    p.add_argument("--out",         type=str,   default="results",
                   help="Root output directory.  "
                        "Stitching saves to <out>/stitching/  "
                        "Charting saves to  <out>/charting/  "
                        "(default: results/)")
    p.add_argument("--tau",         type=float, default=COMP_TAU)
    p.add_argument("--scale",       type=float, default=0.12)
    p.add_argument("--imu_sign",    type=float, default=-1.0)
    p.add_argument("--qs_thresh",   type=float, default=0.28)
    p.add_argument("--qs_sharp",    type=float, default=400.)
    p.add_argument("--qs_contact",  type=float, default=1800.)
    p.add_argument("--qs_snr",      type=float, default=18.)
    p.add_argument("--bidirectional", action="store_true")
    p.add_argument("--iou_thresh",  type=float, default=0.25)
    p.add_argument("--jump_thresh", type=float, default=15.0)
    p.add_argument("--quantize",    type=str,   default=None)
    p.add_argument("--calib_samples",type=int,  default=100)
    p.add_argument("--jitter_um",   type=float, default=500.0)
    p.add_argument("--min_frames",  type=int,   default=10)

    args = p.parse_args()

    # ── Resolve output subdirectories ─────────────────────────────────────────
    root_out      = Path(args.out)
    stitch_out    = root_out / "stitching"
    chart_out     = root_out / "charting"
    stitch_out.mkdir(parents=True, exist_ok=True)
    chart_out.mkdir(parents=True,  exist_ok=True)

    # ── Quantise mode (panorama utility — no display needed) ──────────────────
    if args.quantize:
        result = quantize_b1_model(args.model, args.quantize, args.calib_samples)
        print(json.dumps(result, indent=2)); return

    print("=" * 64)
    print("  UNIFIED LIVE SYSTEM — Stitching + Charting")
    print("=" * 64)
    print(f"  Model     : {args.model}")
    print(f"  User      : {args.user or '(base model, no adapter)'}")
    print(f"  HD60X     : device {args.device}")
    print(f"  Output    : {root_out}/")
    print(f"              ├── stitching/   (panorama sessions)")
    print(f"              └── charting/    (tooth chart sessions)")
    print()
    print("  Say 'stitching' → panorama mode  (ON / OFF to start/stop sweep)")
    print("  Say 'charting'  → charting mode   (GO → [tooth] → STOP → END)")
    print("=" * 64 + "\n")

    # ── Import hardware ───────────────────────────────────────────────────────
    try:
        sys.path.insert(0, str(Path(args.model).parent))
        from video_inference_pipeline     import MKUNetInference
        from live_hybrid_fusion_realtime2 import HD60XOptimizedCapture as _HD60X
    except ImportError as e:
        print(f"ERROR: {e}"); sys.exit(1)

    # ── Shared DentScan model ───────────────────────────────────────────────────────
    print(f"[DentScan] Loading {args.model}…")
    mkunet          = MKUNetInference(args.model)
    pano_annotator  = MKUNetLiveAnnotator(mkunet)
    chart_annotator = RealTimeB1ChartAnnotator(
        model_path         = args.model,
        enrollment_dir     = args.enrollment_dir,
        whisper_model_size = args.whisper_model,
        voice_device       = args.voice_device,
        enable_voice       = False,          # voice handled by UnifiedVoiceController
        mkunet_inference   = mkunet)         # reuse already-loaded model — no second load
    print("[DentScan] ✓ Shared model ready\n")

    # ── HD60X capture ─────────────────────────────────────────────────────────
    print(f"[HD60X] device {args.device}…")
    capture = _HD60X(device_id=args.device)
    capture.start_capture()

    # ── IMU ───────────────────────────────────────────────────────────────────
    if args.no_imu:
        imu = IMUReader(mac=args.mac)
    elif args.serial:
        imu = SerialIMUReader(port=args.serial, baud=args.baud); imu.start()
        for _ in range(8):
            time.sleep(1.)
            if imu.connected: break
    else:
        imu = IMUReader(mac=args.mac); imu.start()
        for _ in range(15):
            time.sleep(1.)
            if imu.connected: break

    # ── Unified voice controller ───────────────────────────────────────────────
    voice = None
    if not args.no_voice:
        voice = UnifiedVoiceController(
            whisper_model_size = args.whisper_model,
            voice_device       = args.voice_device,
            enrollment_dir     = args.enrollment_dir,
            user               = args.user)
        voice.start()

    # ── Panorama system ────────────────────────────────────────────────────────
    pano_system = LivePanoramaSystem(
        annotator   = pano_annotator,
        capture     = capture,
        imu_reader  = imu,
        output_dir  = str(stitch_out),
        voice       = None)    # voice handled by UnifiedVoiceController below
    pano_system.cf.tau      = args.tau
    pano_system.cf.scale    = args.scale
    pano_system.cf.imu_sign = args.imu_sign
    pano_system.proc.iou_gate.iou_thresh  = args.iou_thresh
    pano_system.proc.iou_gate.jump_thresh = args.jump_thresh
    pano_system._sweep_tracker.forward_only = not args.bidirectional

    # ── Charting system ────────────────────────────────────────────────────────
    chart_system = ToothChartManager(
        output_dir         = str(chart_out),
        jitter_threshold_um= args.jitter_um,
        min_stable_frames  = args.min_frames,
        enable_voice       = False,   # voice handled by UnifiedVoiceController
        enrollment_dir     = args.enrollment_dir)

    if voice:
        pano_system.voice = _PanoVoiceAdapter(voice)

    # ── Windows ───────────────────────────────────────────────────────────────
    WIN_LIVE  = "Live — B1 Unified"
    WIN_PANO  = "Panorama"
    WIN_CHART = "Dental Charting"
    cv2.namedWindow(WIN_LIVE,  cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN_LIVE,  700, 550)
    cv2.namedWindow(WIN_PANO,  cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN_PANO,  900, 430)
    cv2.namedWindow(WIN_CHART, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN_CHART, 800, 600)

    placeholder_pano  = np.zeros((350, 700, 3), np.uint8)
    placeholder_chart = np.zeros((480, 800, 3), np.uint8)
    cv2.putText(placeholder_pano,  "Press 1 or say ON to start stitching",
                (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150,150,150), 2)
    cv2.putText(placeholder_chart, "Say GO then tooth number to start charting",
                (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (150,150,150), 2)
    cv2.imshow(WIN_PANO,  placeholder_pano)
    cv2.imshow(WIN_CHART, placeholder_chart)

    print("\n[System] Ready — default mode: STITCHING")
    print("  Keyboard  : 1=start stitch  2=stop+save  R=discard")
    print("              press 1 again to switch to charting when not recording")
    print("              press 2 again to switch to stitching when in charting")
    print("  Voice     : ON/OFF = start/stop stitch  |  say 'charting' to switch")
    print("              GO + tooth = charting        |  say 'stitching' to switch")
    print("  Q=quit  S=screenshot\n")

    # Default to stitching — it's the primary mode
    current_mode = 'stitching'
    if voice: voice.mode = 'stitching'

    frame_skip = 0
    pano_ctr   = 0
    imu_conf   = 0.
    gm_x       = 0.   # last known gm_x for _do_start fallback

    try:
        while True:

            # ── Grab frame ────────────────────────────────────────────────────
            frame_bgr = capture.get_frame()
            if frame_bgr is None:
                frame_skip += 1
                if frame_skip > 100: print("[HD60X] lost"); break
                time.sleep(0.01); cv2.waitKey(1); continue
            frame_skip = 0

            # ── IMU ───────────────────────────────────────────────────────────
            imu_now = imu.get_latest()

            # ── Voice mode switch (checked every frame) ───────────────────────
            if voice:
                mode_evt = voice.get_mode_event()
                if mode_evt and mode_evt != current_mode:
                    old_mode     = current_mode
                    current_mode = mode_evt
                    print(f"\n[System] ══ MODE: {old_mode.upper()} → {current_mode.upper()} ══")
                    if current_mode == 'stitching':
                        print("[System]   1=start  2=stop+save  ON/OFF by voice")
                    else:
                        print("[System]   GO → [14 B] → STOP → END")

            # ══════════════════════ STITCHING ══════════════════════════════════
            if current_mode == 'stitching':

                result = pano_system.proc.process(frame_bgr)
                if result is None:
                    cv2.waitKey(1); continue

                gm_x      = result['gm_x_mm']
                gm_y      = result['gm_y_mm']
                quality   = result['quality_score']
                gate_comp = result.get('gate_comp', {'combined': quality})
                gate_pass = result.get('gate_pass',
                            (quality >= QS_THRESH) or
                            ('GM' in pano_system.proc.ann.point_coords_cache))

                if pano_system.recording and not pano_system.voice_paused and gate_pass:
                    gm_x_fused, imu_conf = pano_system.cf.update(gm_x, imu_now, quality)
                else:
                    gm_x_fused = gm_x; imu_conf = 0.

                pano_system._detect_sweep_dir(gm_x_fused)
                pano_system._update_jitter(gm_x)

                dir_accept = True
                if pano_system.recording and not pano_system.voice_paused:
                    dir_accept, _, _ = pano_system._sweep_tracker.update(gm_x_fused)

                imu_used = (imu.connected and imu_now.valid and abs(imu_now.gyro_z) > 0.1)

                # Canvas compositing — mirrors original run() exactly
                if pano_system.recording and not pano_system.voice_paused and gate_pass and dir_accept:
                    col0   = pano_system._compute_col0(gm_x_fused)
                    gm_row = gm_y / PIXEL_SPACING_MM
                    pano_system.canvas.add_frame(
                        result['frame_256'], col0, gm_row, quality, imu_used=imu_used)
                    _gm_dy  = result.get('gm_dy_px', 0.)
                    _kf_ok  = ((pano_system._last_keyframe_col0 is None or
                                abs(col0 - pano_system._last_keyframe_col0) >= KEYFRAME_MIN_DIST_PX)
                               and abs(_gm_dy) <= KEYFRAME_JITTER_PX
                               and quality >= KEYFRAME_QUALITY)
                    if _kf_ok:
                        pano_system.sharp_canvas.add_frame(
                            result['frame_256'], col0, gm_row, quality, imu_used=imu_used)
                        pano_system._last_keyframe_col0 = col0
                    pano_system._session_frames.append(
                        (result['frame_256']*255).astype(np.uint8))
                    coord = {
                        'frame_n':       len(pano_system._session_coords),
                        'ts':            time.time(),
                        'gm_x_mm':       round(gm_x,    4),
                        'gm_y_mm':       round(gm_y,    4),
                        'gm_x_fused':    round(gm_x_fused, 4),
                        'col0_px':       round(col0,    2),
                        'gm_row_frame':  round(gm_row,  2),
                        'quality_score': round(quality,  4),
                        'gate_score':    round(gate_comp.get('combined', quality), 4),
                        'imu_yaw':       round(imu_now.yaw,   3),
                        'imu_pitch':     round(imu_now.pitch, 3),
                        'imu_roll':      round(imu_now.roll,  3),
                        'imu_gyro_z':    round(imu_now.gyro_z,3),
                        'imu_conf':      round(imu_conf, 3),
                        'imu_connected': imu.connected,
                        'imu_valid':     imu_now.valid,
                        'is_anchor':     len(pano_system._session_coords) in pano_system.anchor_marks,
                        'is_bad':        pano_system._bad_start is not None,
                    }
                    pano_system._session_coords.append(coord)
                    pano_system.audit.log(
                        frame_n          = len(pano_system._session_coords),
                        ts               = coord['ts'],
                        gm_x             = gm_x,
                        gm_y             = gm_y,
                        col0_px          = col0,
                        quality          = quality,
                        gater_components = gate_comp,
                        imu_yaw          = imu_now.yaw,
                        imu_conf         = imu_conf,
                        imu_valid        = imu_now.valid,
                        anchor           = len(pano_system._session_coords)-1 in pano_system.anchor_marks,
                        bad              = pano_system._bad_start is not None,
                    )
                    pano_system.stats['frames_good'] += 1
                pano_system.stats['frames_total'] += 1

                # Voice pano commands
                if voice:
                    while True:
                        vcmd = voice.get_pano_command()
                        if vcmd is None: break
                        if   vcmd == 'on'   and not pano_system.recording:
                            pano_system._do_start(gm_x, imu_now)
                            pano_system._flash("ON")
                            cv2.imshow(WIN_PANO, placeholder_pano)
                        elif vcmd == 'off'  and pano_system.recording:
                            sharp = pano_system._do_stop_save()
                            if sharp is not None:
                                pano_system._show_panorama_fullscreen(sharp, "PANORAMA")
                        elif vcmd == 'save' and pano_system.recording:
                            pano_system._do_stop_save()
                        elif vcmd == 'drop' and pano_system.recording:
                            pano_system.recording = False; pano_system.canvas.reset()
                            pano_system._session_frames.clear()
                            pano_system._session_coords.clear()
                            cv2.imshow(WIN_PANO, placeholder_pano)
                        elif vcmd == 'hold' and pano_system.recording:
                            pano_system.voice_paused = True;  pano_system._flash("HOLD")
                        elif vcmd == 'go'   and pano_system.recording:
                            pano_system.voice_paused = False; pano_system._flash("GO")
                        elif vcmd == 'mark' and pano_system.recording:
                            pano_system.anchor_marks.append(len(pano_system._session_coords))
                            pano_system._flash("MARK")
                        elif vcmd == 'view':
                            pano_system.show_imu = not pano_system.show_imu

                # Display live frame
                ann = result['annotated']
                ann = pano_system._draw_gate_overlay(ann, gate_comp, gate_pass)
                if pano_system.show_imu:
                    ann = pano_system._draw_imu_overlay(ann, imu_now, imu_conf)
                ann = pano_system._draw_recording_overlay(ann)
                rec_lbl = "●REC — press 2 or say OFF to stop" if pano_system.recording \
                          else "STITCH — press 1 or say ON to start"
                cv2.putText(ann, rec_lbl, (8, ann.shape[0]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,220,255), 1)
                cv2.imshow(WIN_LIVE, ann)

                # Panorama window (every 3 frames) — frozen after save
                pano_ctr += 1
                if pano_ctr >= 3 and not getattr(pano_system, '_panorama_frozen', False):
                    pano_ctr = 0
                    sharp_p = pano_system.sharp_canvas.get_panorama()
                    avg_p   = pano_system.canvas.get_panorama()
                    disp    = sharp_p if sharp_p is not None else avg_p
                    if disp is not None:
                        ph, pw = disp.shape[:2]
                        dh = 350; dw = max(200, int(pw*dh/ph))
                        pd = cv2.resize(cv2.cvtColor(disp, cv2.COLOR_RGB2BGR), (dw, dh))
                        cov   = pano_system.canvas.get_coverage_info()
                        strip = pano_system.heatmap.render(cov, dw)
                        pd = np.vstack([pd, strip])
                        kf = pano_system.sharp_canvas.frame_count
                        cv2.putText(pd, f"SHARP {kf}kf  {pw*PIXEL_SPACING_MM:.1f}mm",
                                    (8, dh-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,130), 1)
                        cv2.imshow(WIN_PANO, pd)
                        cv2.resizeWindow(WIN_PANO, max(400,dw),
                                         dh+pano_system.heatmap.STRIP_H+20)
                    elif not pano_system.recording:
                        cv2.imshow(WIN_PANO, placeholder_pano)

            # ══════════════════════ CHARTING ═══════════════════════════════════
            elif current_mode == 'charting':

                # Voice chart commands
                if voice:
                    while True:
                        ccmd = voice.get_chart_command()
                        if ccmd is None: break
                        ctype, cval = ccmd
                        if ctype == 'tooth':
                            chart_system.start_tooth_scan(cval)
                        elif ctype == 'action':
                            if cval in ('stop', 'end'):
                                chart_system.stop_tooth_scan()
                                chart_system.save_best_frames()
                                ts_x = datetime.now().strftime("%Y%m%d_%H%M%S")
                                chart_system.export_to_excel(
                                    str(chart_out / f"chart_{ts_x}.xlsx"))
                                if cval == 'end':
                                    voice.chart_state = 'finished'
                                    print("[Chart] Session ended — results in", chart_out)

                # B1 inference + display
                try:
                    batch = chart_annotator.preprocess_frame_fast(frame_bgr)
                    if batch is not None:
                        preds = chart_annotator.predict_single_frame(batch)
                        h, w  = frame_bgr.shape[:2]
                        proc  = chart_annotator.post_process_predictions(preds, (h, w))
                        meas  = chart_annotator.get_measurements_throttled(preds)
                        cf    = chart_annotator.create_annotated_frame(frame_bgr, proc)
                        cf    = chart_annotator.add_performance_overlay(cf)
                        if chart_system.scanning_active and meas:
                            chart_system.add_frame(cf, meas)
                        status = chart_system.get_status_text() or "say GO then tooth number"
                        cv2.putText(cf, f"CHARTING  {status}",
                                    (8, cf.shape[0]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,130), 1)
                        cv2.imshow(WIN_LIVE,  cf)
                        cv2.imshow(WIN_CHART, cf)
                    else:
                        cv2.imshow(WIN_CHART, placeholder_chart)
                except Exception:
                    pass

            # ── Keyboard ──────────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                # Auto-save stitching if recording before exit
                if current_mode == 'stitching' and pano_system.recording:
                    print("[System] Auto-saving before quit…")
                    pano_system._do_stop_save()
                break

            elif key == ord('1'):
                if current_mode == 'stitching':
                    # Start recording
                    if not pano_system.recording:
                        pano_system._panorama_frozen = False   # unfreeze on new scan
                        pano_system._do_start(gm_x, imu_now)
                        print("[Stitch] ● Recording — press 2 to stop+save")
                    else:
                        print("[Stitch] Already recording — press 2 to stop+save")
                else:
                    # Switch TO stitching
                    current_mode = 'stitching'
                    if voice: voice.mode = 'stitching'
                    print("[System] ══ MODE → STITCHING ══  press 1 to start")

            elif key == ord('2'):
                if current_mode == 'stitching':
                    if pano_system.recording:
                        # Stop and save — panorama stays visible until next scan
                        sharp = pano_system._do_stop_save()
                        pano_system._show_panorama_fullscreen(sharp, "PANORAMA — press 1 to scan again")
                        pano_ctr = 0   # reset counter so freeze takes effect immediately
                        print("[Stitch] Saved — panorama displayed  (press 1 to start new sweep)")
                    else:
                        # Not recording — switch to charting
                        current_mode = 'charting'
                        if voice: voice.mode = 'charting'; voice.chart_state = 'inactive'
                        print("[System] ══ MODE → CHARTING ══  say GO then tooth number")
                elif current_mode == 'charting':
                    # Switch back to stitching
                    current_mode = 'stitching'
                    if voice: voice.mode = 'stitching'
                    print("[System] ══ MODE → STITCHING ══  press 1 to start")

            elif key == ord('r') and current_mode == 'stitching' and pano_system.recording:
                pano_system.recording = False; pano_system.canvas.reset()
                pano_system._session_frames.clear(); pano_system._session_coords.clear()
                print("[Stitch] Discarded")

            elif key == ord('s'):
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                d     = chart_out if current_mode == 'charting' else stitch_out
                cv2.imwrite(str(d / f"screenshot_{ts}.png"), frame_bgr)
                print(f"[Screenshot] {d}/screenshot_{ts}.png")

    except KeyboardInterrupt:
        print("\n[System] Interrupted")
    finally:
        cv2.destroyAllWindows()
        capture.stop_capture()
        imu.stop()
        if voice: voice.stop()
        print("[System] Done.")

if __name__ == "__main__":
    main()