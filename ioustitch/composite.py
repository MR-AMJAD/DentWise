"""
Uncertainty-aware two-band compositing.

A seam-strip compositor paints each frame only inside a narrow band between
neighbour midpoints; with dense frames (probe dwell) the cosine fade never
reaches full weight, which produces vertical stripes. Here every frame
contributes wherever it overlaps, with a smooth weight

    w_i(u) = quality_i * feather(u) * tissue_i(u) / (1 + sd_i^2 / sd_ref^2)

and two frequency bands (as in multiband blending, Burt & Adelson 1983):
    low band  : weighted mean of Gaussian-low-passed frames      (no seams)
    high band : weighted mean with sharpened weights w^gamma      (keeps detail,
                                                                   limits ghosting)
Also returns a per-pixel positional-uncertainty map (weighted mean of the
contributing frames' marginal sd) and a coverage map.
"""
import numpy as np
import cv2
from .preprocess import warp_translate


def feather_mask(H, W, fx, fy=8):
    x = np.ones(W, np.float32)
    r = np.arange(fx, dtype=np.float32)
    ramp = 0.5 * (1 - np.cos(np.pi * (r + 0.5) / fx))
    x[:fx] = ramp; x[-fx:] = ramp[::-1]
    y = np.ones(H, np.float32)
    ry = np.arange(fy, dtype=np.float32)
    rampy = 0.5 * (1 - np.cos(np.pi * (ry + 0.5) / fy))
    y[:fy] = rampy; y[-fy:] = rampy[::-1]
    return np.outer(y, x)


def composite(frames, p, q, quality=None, sd=None, tissue=None, cfg=None,
              mode=None, pad=4):
    mode = mode or (cfg.blend if cfg else "twoband")
    N = len(frames); H, W = frames[0].shape
    quality = np.ones(N) if quality is None else np.asarray(quality, float)
    sd = np.zeros(N) if sd is None else np.asarray(sd, float)
    sref = cfg.sigma_ref_px if cfg else 2.0
    fx = cfg.feather_px if cfg else 24
    lsig = cfg.lowband_sigma if cfg else 6.0
    gam = cfg.highband_gamma if cfg else 6.0

    x0 = int(np.floor(np.min(p))) - pad; y0 = int(np.floor(np.min(q))) - pad
    CW = int(np.ceil(np.max(p) - x0 + W)) + pad
    CH = int(np.ceil(np.max(q) - y0 + H)) + pad
    shape = (CH, CW)
    fm = feather_mask(H, W, fx)
    lo_acc = np.zeros(shape, np.float32); lo_w = np.zeros(shape, np.float32)
    hi_acc = np.zeros(shape, np.float32); hi_w = np.zeros(shape, np.float32)
    sd_acc = np.zeros(shape, np.float32); cover = np.zeros(shape, np.float32)
    best_w = np.zeros(shape, np.float32); wta = np.zeros(shape, np.float32)
    for k in range(N):
        if quality[k] <= 0:
            continue                       # blind (lift-off) frame
        f = frames[k].astype(np.float32)
        w = fm * float(quality[k]) / (1.0 + (sd[k] / sref) ** 2)
        if tissue is not None:
            # keep dark anatomy (shadow) contributing, but prefer echogenic content
            w = w * (0.25 + 0.75 * tissue[k])
        w = w.astype(np.float32)
        dx, dy = p[k] - x0, q[k] - y0
        fw = warp_translate(f, dx, dy, shape)
        ww = warp_translate(w, dx, dy, shape)
        cover += warp_translate(fm, dx, dy, shape) > 0.5
        sd_acc += ww * float(sd[k])
        if mode == "compound":
            lo_acc += ww * fw; lo_w += ww
        elif mode == "wta":
            upd = ww > best_w
            wta[upd] = fw[upd]; best_w[upd] = ww[upd]
            lo_w += ww
        else:
            low = cv2.GaussianBlur(fw * (ww > 0), (0, 0), lsig)
            nrm = cv2.GaussianBlur((ww > 0).astype(np.float32), (0, 0), lsig)
            low = low / np.maximum(nrm, 1e-3)
            high = fw - low
            lo_acc += ww * low; lo_w += ww
            wh = ww ** gam
            hi_acc += wh * high; hi_w += wh
    if mode == "compound":
        pano = lo_acc / np.maximum(lo_w, 1e-6)
    elif mode == "wta":
        pano = wta
    else:
        pano = lo_acc / np.maximum(lo_w, 1e-6) + hi_acc / np.maximum(hi_w, 1e-12)
    pano[lo_w <= 1e-6] = 0
    sdmap = sd_acc / np.maximum(lo_w, 1e-6)
    sdmap[lo_w <= 1e-6] = np.nan
    return dict(pano=np.clip(pano, 0, 1), sd_map=sdmap, coverage=cover, origin=(x0, y0))


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def uncertainty_strip(sd_map, px_mm, height=18, max_um=500.0):
    """Column-wise positional uncertainty strip (colour-coded, um)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        col = np.nanmean(np.where(np.isfinite(sd_map), sd_map, np.nan), axis=0)
    col = np.nan_to_num(col, nan=np.nanmax(col) if np.isfinite(col).any() else 0.0)
    um = col * px_mm * 1000.0
    v = np.clip(um / max_um, 0, 1)
    rgb = cv2.applyColorMap((v * 255).astype(np.uint8)[None, :], cv2.COLORMAP_TURBO)
    return np.repeat(rgb, height, axis=0), um
