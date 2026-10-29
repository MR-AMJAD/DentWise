"""
Speckle-aware, masked, uncertainty-quantified pairwise registration.

Convention (used EVERYWHERE in the package)
-------------------------------------------
Frame i is placed on the panorama canvas with its top-left corner at
(p_i, q_i) (x = lateral, y = axial, pixels).  A tissue point with canvas
coordinate X appears in frame i at image coordinate u = X - p_i.  Hence

        I_j(u) = I_i(u + d)      with   d = (p_j - p_i, q_j - q_i).

`register_pair(i, j)` returns the MAP estimate of d and its 2x2 covariance.

Method
------
1. Masked weighted NCC over ALL shifts at once in the Fourier domain
   (Padfield, IEEE TIP 2012), so dark background / gel / out-of-tissue
   pixels do not contribute and the overlap-normalisation is exact.
2. Sub-pixel peak by local quadratic fit.
3. Laplace approximation of a pseudo-likelihood  L(d) ~ exp(kappa * NCC(d))
   with kappa = n_eff * rho / (1 - rho), n_eff = overlap / speckle-cell area
   -> covariance  Sigma = (kappa * H)^-1, H = -Hessian of NCC at the peak.
   This captures the aperture problem: a horizontal structure (e.g. the
   gingival surface line) gives a flat NCC ridge along x -> large sigma_x.
4. Multi-modality: the second-highest local peak is moment-matched into the
   covariance (Gaussian approximation of a 2-component mixture), so
   periodic/ambiguous anatomy yields inflated uncertainty instead
   of a confidently wrong match.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy import fft as sfft


@dataclass
class PairResult:
    i: int
    j: int
    d: np.ndarray            # (dx, dy) px, estimate of (p_j - p_i, q_j - q_i)
    cov: np.ndarray          # 2x2 covariance (px^2)
    rho: float               # peak NCC
    rho2: float              # second peak NCC
    overlap: float           # weighted overlap mass at the peak
    ok: bool
    reason: str = ""
    role: str = ""           # provenance: 'short', 'bridge', 'long', 'long_gap', 'seq', 'loop'
    lag: int = 0             # intended lag (counted in sighted frames for guided pairs)


class FrameSpectra:
    """Pre-computed zero-padded spectra of (w, w*f, w*f^2) for one frame."""
    __slots__ = ("W", "WF", "WF2", "mass", "shape")

    def __init__(self, f: np.ndarray, w: np.ndarray, pad_shape):
        w = w.astype(np.float32)
        f = f.astype(np.float32)
        # centre + unit-scale intensities: NCC is invariant to this, and it
        # keeps the float32 moment sums well conditioned (no cancellation).
        m = float((f * w).sum() / max(w.sum(), 1e-6))
        s = float(np.sqrt(((f - m) ** 2 * w).sum() / max(w.sum(), 1e-6))) + 1e-6
        f = (f - m) / s
        self.shape = f.shape
        self.W = sfft.rfft2(w, s=pad_shape, workers=-1)
        self.WF = sfft.rfft2(w * f, s=pad_shape, workers=-1)
        self.WF2 = sfft.rfft2(w * f * f, s=pad_shape, workers=-1)
        self.mass = float(w.sum())


def pad_shape_for(shape):
    H, W = shape
    return (2 * H, 2 * W)


def _corr(A, B, pad_shape):
    """circular cross-correlation  C(d) = sum_u a(u+d) b(u)  via spectra."""
    return sfft.irfft2(A * np.conj(B), s=pad_shape, workers=-1)


def masked_ncc_map(Si: FrameSpectra, Sj: FrameSpectra, pad_shape, eps=1e-9):
    """Weighted NCC between I_i(u+d) and I_j(u) for every shift d.
    Returns (ncc, overlap) arrays indexed circularly by (dy, dx)."""
    S_ab = _corr(Si.W, Sj.W, pad_shape)
    S_fa = _corr(Si.WF, Sj.W, pad_shape)
    S_gb = _corr(Si.W, Sj.WF, pad_shape)
    S_fg = _corr(Si.WF, Sj.WF, pad_shape)
    S_f2 = _corr(Si.WF2, Sj.W, pad_shape)
    S_g2 = _corr(Si.W, Sj.WF2, pad_shape)
    ov = np.maximum(S_ab, eps)
    num = S_fg - S_fa * S_gb / ov
    vf = S_f2 - S_fa * S_fa / ov
    vg = S_g2 - S_gb * S_gb / ov
    den = np.sqrt(np.maximum(vf, eps) * np.maximum(vg, eps))
    ncc = num / den
    return ncc, S_ab


def _window(arr, ry, rx):
    """Extract shifts dy in [-ry, ry], dx in [-rx, rx] from a circular map.
    Returned array index [ry+dy, rx+dx]."""
    Hp, Wp = arr.shape
    ys = np.arange(-ry, ry + 1) % Hp
    xs = np.arange(-rx, rx + 1) % Wp
    return arr[np.ix_(ys, xs)]


def _quad_fit(C, y, x):
    """Local quadratic fit on 3x3 neighbourhood of C at (y, x).
    Returns (offset (dy,dx), Hessian A (2x2, in [y,x] order))."""
    c = C[y, x]
    gy = 0.5 * (C[y + 1, x] - C[y - 1, x])
    gx = 0.5 * (C[y, x + 1] - C[y, x - 1])
    gyy = C[y + 1, x] - 2 * c + C[y - 1, x]
    gxx = C[y, x + 1] - 2 * c + C[y, x - 1]
    gxy = 0.25 * (C[y + 1, x + 1] - C[y + 1, x - 1] - C[y - 1, x + 1] + C[y - 1, x - 1])
    A = np.array([[gyy, gxy], [gxy, gxx]])
    g = np.array([gy, gx])
    off = np.zeros(2)
    try:
        if np.all(np.linalg.eigvalsh(A) < 0):
            off = -np.linalg.solve(A, g)
            off = np.clip(off, -1.0, 1.0)
    except np.linalg.LinAlgError:
        pass
    return off, A


def register_pair(i: int, j: int, Si: FrameSpectra, Sj: FrameSpectra, pad_shape,
                  search_x: int, search_y: int, speckle_area: float,
                  min_overlap_frac=0.25, min_ncc=0.15, peak_exclusion=4,
                  center: Optional[Tuple[float, float]] = None,
                  radius: Optional[float] = None) -> PairResult:
    ncc, ov = masked_ncc_map(Si, Sj, pad_shape)
    C = _window(ncc, search_y, search_x)
    O = _window(ov, search_y, search_x)
    min_ov = min_overlap_frac * min(Si.mass, Sj.mass)
    valid = O >= min_ov
    # keep a 1-px border so the quadratic fit always has neighbours
    valid[0, :] = valid[-1, :] = False
    valid[:, 0] = valid[:, -1] = False
    if center is not None and radius is not None:
        yy, xx = np.mgrid[-search_y:search_y + 1, -search_x:search_x + 1]
        valid &= ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius ** 2
    if not valid.any():
        return PairResult(i, j, np.zeros(2), np.eye(2) * 1e6, 0., 0., 0., False, "no-overlap")
    Cv = np.where(valid, C, -np.inf)
    k = int(np.argmax(Cv))
    y0, x0 = np.unravel_index(k, C.shape)
    rho = float(C[y0, x0])
    if not np.isfinite(rho) or rho < min_ncc:
        return PairResult(i, j, np.zeros(2), np.eye(2) * 1e6, rho, 0., 0., False, "low-ncc")

    off, A = _quad_fit(C, y0, x0)
    dy = (y0 - search_y) + off[0]
    dx = (x0 - search_x) + off[1]
    overlap = float(O[y0, x0])

    # ---- Laplace covariance ------------------------------------------------
    rho_c = float(np.clip(rho, 0.05, 0.98))
    n_eff = max(overlap / max(speckle_area, 1.0), 4.0)
    kappa = n_eff * rho_c / (1.0 - rho_c)
    Hm = -A
    w, V = np.linalg.eigh(Hm)
    w = np.maximum(w, 1e-4)                    # flat direction -> huge variance
    cov_yx = V @ np.diag(1.0 / (kappa * w)) @ V.T
    cov = cov_yx[::-1, ::-1].copy()            # -> (x, y) order
    # quantisation / interpolation floor (sub-pixel fits are not better than ~0.05 px)
    cov += np.eye(2) * 0.05 ** 2

    # ---- second peak & mixture moment matching ----------------------------
    Cm = Cv.copy()
    r = peak_exclusion
    Cm[max(0, y0 - r):y0 + r + 1, max(0, x0 - r):x0 + r + 1] = -np.inf
    rho2 = 0.0
    if np.isfinite(Cm).any():
        k2 = int(np.argmax(Cm))
        y2, x2 = np.unravel_index(k2, C.shape)
        rho2 = float(C[y2, x2])
        if np.isfinite(rho2) and rho2 > 0:
            # posterior mass ratio of the two modes under L ~ exp(kappa*rho)
            w2 = np.exp(-kappa * max(rho - rho2, 0.0))
            if w2 > 1e-6:
                delta = np.array([(x2 - search_x) - dx, (y2 - search_y) - dy])
                p1, p2 = 1.0 / (1.0 + w2), w2 / (1.0 + w2)
                cov = cov + p1 * p2 * np.outer(delta, delta)
    return PairResult(i, j, np.array([dx, dy]), cov, rho, rho2, overlap, True)


def estimate_speckle_area(frames_f, weights, max_frames=12):
    """Speckle-cell area (px^2) from the normalised autocorrelation FWHM,
    averaged over a few frames. Used to convert overlap to effective
    independent samples (n_eff)."""
    idx = np.linspace(0, len(frames_f) - 1, min(max_frames, len(frames_f))).astype(int)
    areas = []
    for k in idx:
        f = frames_f[k].astype(np.float64); w = weights[k].astype(np.float64)
        if w.sum() < 100:
            continue
        m = (f * w).sum() / w.sum()
        g = (f - m) * w
        F = np.fft.rfft2(g, s=(2 * f.shape[0], 2 * f.shape[1]))
        ac = np.fft.irfft2(F * np.conj(F), s=(2 * f.shape[0], 2 * f.shape[1]))
        ac = ac / (ac[0, 0] + 1e-12)
        ax = ac[0, :]; ay = ac[:, 0]
        wx = np.argmax(ax[:64] < 0.5) if (ax[:64] < 0.5).any() else 32
        wy = np.argmax(ay[:64] < 0.5) if (ay[:64] < 0.5).any() else 32
        areas.append(max(np.pi * wx * wy, 1.0))
    return float(np.median(areas)) if areas else 12.0
