"""
Frame conditioning for registration.

* speckle pre-smoothing (Gaussian, sigma ~ speckle grain)
* soft tissue weight map (excludes dark background / gel / shadow, whose
  pixels carry no position information but do carry noise)
* probe-fixed layer separation (C1): every frame is modelled as the sum of a
  tissue layer that moves with the anatomy and a probe-fixed layer that does
  not (near-field ring-down, the coupling-interface line, reverberation bands):

        I_i(u) = T(u + p_i) + B(u) + noise

  Probe-fixed structure correlates with itself at zero displacement and biases
  registration toward d = 0. After a first trajectory estimate, T is the
  motion-compensated weighted mean of the frames and B is updated once as the
  temporal median of the non-negative motion-compensated residuals. If the
  probe barely moves, T absorbs all structure and B -> 0.
"""
import numpy as np
import cv2


def to_gray01(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0
    return img


def smooth(img, sigma):
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def tissue_weight(img_s, lo, hi, erode_px=2):
    """Soft 0..1 weight: 0 in background, ramps to 1 in echogenic tissue.
    A light blur of the weight avoids hard mask edges driving the NCC."""
    w = np.clip((img_s - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    w = cv2.GaussianBlur(w, (0, 0), 3.0)
    if erode_px > 0:
        k = np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8)
        w = cv2.erode(w, k)
    # suppress the 3 outermost columns/rows (scan-converter edge effects)
    w[:, :3] = 0; w[:, -3:] = 0; w[:3, :] = 0; w[-3:, :] = 0
    return w.astype(np.float32)


def warp_translate(img, dx, dy, out_shape, border=0.0):
    """Place `img` so that out(X) = img(X - (dx,dy)) (sub-pixel, bilinear)."""
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, M, (out_shape[1], out_shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)


def motion_compensated_mean(frames, weights, p, q, pad=4):
    """Tissue layer T on a canvas, given positions (p,q). Returns (T, W, x0, y0)."""
    H, Wd = frames[0].shape
    x0 = int(np.floor(np.min(p))) - pad; y0 = int(np.floor(np.min(q))) - pad
    CW = int(np.ceil(np.max(p) - x0 + Wd)) + pad
    CH = int(np.ceil(np.max(q) - y0 + H)) + pad
    acc = np.zeros((CH, CW), np.float32); wsum = np.zeros((CH, CW), np.float32)
    for f, w, pi, qi in zip(frames, weights, p, q):
        acc += warp_translate(f * w, pi - x0, qi - y0, (CH, CW))
        wsum += warp_translate(w, pi - x0, qi - y0, (CH, CW))
    T = acc / np.maximum(wsum, 1e-6)
    return T, wsum, x0, y0


def estimate_probe_fixed_layer(frames, weights, p, q, min_motion_px=20.0):
    """B(u) = median_i [ I_i(u) - T(u + p_i) ]_+   (probe-fixed additive layer).
    Returns zeros if the sweep is too short to separate the layers."""
    H, Wd = frames[0].shape
    if (np.max(p) - np.min(p)) < min_motion_px and (np.max(q) - np.min(q)) < min_motion_px:
        return np.zeros((H, Wd), np.float32)
    T, wsum, x0, y0 = motion_compensated_mean(frames, weights, p, q)
    res = []
    for f, pi, qi in zip(frames, p, q):
        # sample T back into frame coordinates: Tf(u) = T(u + p_i - x0)
        M = np.array([[1, 0, -(pi - x0)], [0, 1, -(qi - y0)]], np.float32)
        Tf = cv2.warpAffine(T, M, (Wd, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        res.append(f - Tf)
    B = np.median(np.stack(res), axis=0)
    B = np.maximum(B, 0.0)
    return cv2.GaussianBlur(B, (0, 0), 1.0).astype(np.float32)


def static_pattern_weight(frames, B, eps=0.02):
    """Down-weight pixels dominated by the probe-fixed layer."""
    mean = np.mean(np.stack(frames), axis=0)
    frac = B / (mean + eps)
    return np.clip(1.0 - frac, 0.0, 1.0).astype(np.float32)
