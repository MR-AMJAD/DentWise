"""
Evaluation metrics.

With ground truth (simulation):
    ATE   absolute trajectory error (RMS, after removing the unobservable
          global translation), lateral and axial
    RPE1s relative error of displacements over ~1 s windows (drift)
    SLE   sweep-length error |extent_est - extent_gt| / extent_gt
    NEES coverage: fraction of frames whose error lies inside the reported
          95 % interval (calibration of the uncertainty; ideal 0.95)
    PFID  panorama fidelity: NCC between the panorama composited with the
          estimated trajectory and the one composited with the true
          trajectory (same frames, same compositor)

Without ground truth (in vivo):
    PC    photometric consistency (Eq. 6 of the paper): mean tissue-weighted NCC
          between frame pairs at lags 1-10 at the displacement implied by the
          trajectory under test. It shares the image signal with the proposed
          method and measures internal consistency, not positional accuracy.
"""
import numpy as np


def _center(x):
    return np.asarray(x, float) - np.mean(x)


def trajectory_metrics(p, q, gt_p, gt_q, t, px_mm_x, px_mm_y, sd_p=None):
    ex = _center(p) - _center(gt_p)
    ey = _center(q) - _center(gt_q)
    out = dict(ate_x_um=float(np.sqrt(np.mean(ex ** 2)) * px_mm_x * 1e3),
               ate_y_um=float(np.sqrt(np.mean(ey ** 2)) * px_mm_y * 1e3),
               max_x_um=float(np.max(np.abs(ex)) * px_mm_x * 1e3))
    # RPE over ~1 s
    t = np.asarray(t, float); errs = []
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + 1.0))
        if j < len(t):
            errs.append((p[j] - p[i]) - (gt_p[j] - gt_p[i]))
    out["rpe1s_x_um"] = float(np.sqrt(np.mean(np.square(errs))) * px_mm_x * 1e3) if errs else np.nan
    ext_gt = np.ptp(gt_p)
    out["sle_pct"] = float(abs(np.ptp(p) - ext_gt) / max(ext_gt, 1e-6) * 100)
    if sd_p is not None:
        z = ex / np.maximum(np.asarray(sd_p), 1e-6)
        out["cover95"] = float(np.mean(np.abs(z) <= 1.96))
    return out


def _paste(canvas, img, x0, y0):
    H, W = img.shape
    canvas[y0:y0 + H, x0:x0 + W] = img
    return canvas


def panorama_fidelity(comp_est, comp_gt):
    """NCC between two composites given their canvas origins (x0, y0)."""
    A, (ax, ay) = comp_est["pano"], comp_est["origin"]
    B, (bx, by) = comp_gt["pano"], comp_gt["origin"]
    x0, y0 = min(ax, bx), min(ay, by)
    x1 = max(ax + A.shape[1], bx + B.shape[1]); y1 = max(ay + A.shape[0], by + B.shape[0])
    CA = np.full((y1 - y0, x1 - x0), np.nan, np.float32)
    CB = np.full_like(CA, np.nan)
    _paste(CA, np.where(comp_est["coverage"] > 0, A, np.nan), ax - x0, ay - y0)
    _paste(CB, np.where(comp_gt["coverage"] > 0, B, np.nan), bx - x0, by - y0)
    m = np.isfinite(CA) & np.isfinite(CB)
    if m.sum() < 100:
        return np.nan
    a = CA[m] - CA[m].mean(); b = CB[m] - CB[m].mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def photometric_consistency(frames_s, weights, p, q, max_lag=10, stride=1):
    """Mean weighted NCC between frame pairs at the trajectory-implied shift."""
    import cv2
    H, W = frames_s[0].shape
    vals = []
    for i in range(0, len(frames_s), stride):
        for L in range(1, max_lag + 1):
            j = i + L
            if j >= len(frames_s):
                break
            dx, dy = p[j] - p[i], q[j] - q[i]
            if abs(dx) > W * 0.75 or abs(dy) > H * 0.5:
                continue
            # frame i resampled into frame j's coordinates: I_i(u + d)
            M = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
            fi = cv2.warpAffine(frames_s[i], M, (W, H), flags=cv2.INTER_LINEAR)
            wi = cv2.warpAffine(weights[i], M, (W, H), flags=cv2.INTER_LINEAR)
            w = wi * weights[j]
            if w.sum() < 500:
                continue
            a = fi; b = frames_s[j]
            ma = (a * w).sum() / w.sum(); mb = (b * w).sum() / w.sum()
            num = (w * (a - ma) * (b - mb)).sum()
            den = np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()) + 1e-9
            vals.append(num / den)
    return float(np.mean(vals)) if vals else np.nan
