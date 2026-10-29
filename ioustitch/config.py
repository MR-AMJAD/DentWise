"""
Central configuration of the reconstruction (Table S2 of the Supplementary Material).

Every physical constant and parameter is defined here once. The values are identical for all simulated and clinical
sweeps of the paper.

Pixel spacing: the saved frames are 256 x 256 px resized from a 610 x 640 px capture crop at 11.475 um/px, i.e.
27.3 um (lateral) x 28.7 um (axial) per pixel. Confirm the spacing for another probe or depth setting.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StitchConfig:
    # ---------------- geometry ----------------
    px_mm_x: float = 610 * 0.011475 / 256   # lateral mm/px of the saved 256-px frames (crop 610 px @ 11.475 um)
    px_mm_y: float = 640 * 0.011475 / 256   # axial   mm/px (crop 640 px)
    frame_size: Tuple[int, int] = (256, 256)

    # ---------------- preprocessing (C1) ----------------
    smooth_sigma: float = 1.0            # speckle pre-smoothing before NCC (px)
    tissue_lo: float = 0.03              # soft tissue-weight ramp (intensity 0..1)
    tissue_hi: float = 0.10
    layer_separation: bool = True        # probe-fixed layer separation
    layer_iters: int = 1                 # refinements of the probe-fixed layer

    # ---------------- registration (C2, C3) ----------------
    lags: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)   # scheduled lag set
    search_x: int = 200                  # full search window, lateral (px)
    search_y: int = 40                   # full search window, axial (px)
    guided: bool = True                  # guided multi-lag registration with gap bridging
    guide_r0: float = 6.0                # guided window R(L) = guide_r0 + guide_r1 * L (px)
    guide_r1: float = 1.5
    gap_s: float = 0.30                  # time gap treated like a lift-off (s)
    min_overlap_frac: float = 0.25       # of the smaller mask mass
    min_ncc: float = 0.15                # below this a pair is discarded
    peak_exclusion: int = 4              # px radius around the peak when searching the second peak
    speckle_area_px: float = 0.0         # 0 -> estimated from the data

    # ---------------- factor graph (C4) ----------------
    use_image: bool = True
    use_landmark: bool = True            # GM landmark factors
    use_loop_closure: bool = True        # revisit closure
    motion_sigma_acc: float = 4000.0     # px/s^2 acceleration random walk (weak prior)
    landmark_sigma0: float = 4.0         # px, a priori (rescaled by VCE)

    # ---------------- robust estimation (C4) ----------------
    gnc: bool = True                     # graduated non-convexity (Geman-McClure)
    gnc_c: float = 3.0                   # inlier threshold on the Mahalanobis residual
    gnc_factor: float = 1.4
    gnc_max_iter: int = 60
    vce: bool = True                     # variance-component estimation
    vce_iters: int = 8

    # ---------------- revisit closure (C3) ----------------
    lc_min_lag: int = 20
    lc_max_pred_dx: float = 150.0
    lc_stride: int = 3
    lc_window: float = 25.0              # search half-width around the prediction (px)

    # ---------------- compositing (C5) ----------------
    blend: str = "twoband"               # 'twoband' | 'compound' | 'wta'
    lowband_sigma: float = 6.0
    highband_gamma: float = 6.0          # weight sharpening of the detail band
    feather_px: int = 24
    sigma_ref_px: float = 2.0            # uncertainty -> weight down-scaling

    # ---------------- misc ----------------
    verbose: bool = True
    seed: int = 0

    def px_to_mm(self, dx_px, axis="x"):
        return dx_px * (self.px_mm_x if axis == "x" else self.px_mm_y)
