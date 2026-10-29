"""
End-to-end offline stitching of one intra-oral sweep.

    stitch_sweep(frames, coords, cfg) -> result dict

Stages (Algorithm S1 of the Supplementary Material)
  1. condition frames (gray, speckle smoothing, tissue weights); trajectory-free probe-fixed layer B0 (C1)
  2. blind (lift-off) frames from the relative tissue mass after layer separation (C3)
  3. guided multi-lag masked-NCC registration with gap bridging -> image factors with covariances (C2, C3)
  4. robust factor graph (GNC + VCE) with image, GM landmark, motion and gauge factors (C4)
  5. one refinement of the probe-fixed layer from the trajectory, re-registration, re-solve (C1)
  6. revisit closure (predicted overlap, windowed search) and final solve (C3, C4)
  7. uncertainty-weighted two-band compositing and uncertainty map (C5)
"""
import time
import numpy as np
from .config import StitchConfig
from . import preprocess as pp
from .registration import (FrameSpectra, pad_shape_for, register_pair,
                           estimate_speckle_area)
from .graph import Graph, solve
from .composite import composite


def _log(cfg, *a):
    if cfg.verbose:
        print(*a, flush=True)


def condition(frames_raw, cfg):
    g = [pp.to_gray01(f) for f in frames_raw]
    s = [pp.smooth(x, cfg.smooth_sigma) for x in g]
    w = [pp.tissue_weight(x, cfg.tissue_lo, cfg.tissue_hi) for x in s]
    return g, s, w


def register_all(fs, ws, cfg, pairs, speckle_area, centers=None, radius=None, radii=None):
    pad = pad_shape_for(fs[0].shape)
    specs = {}
    def spec(k):
        if k not in specs:
            specs[k] = FrameSpectra(fs[k], ws[k], pad)
        return specs[k]
    out = []
    for n, (i, j) in enumerate(pairs):
        c = None if centers is None else centers[n]
        rad = radius if radii is None else radii[n]
        r = register_pair(i, j, spec(i), spec(j), pad, cfg.search_x, cfg.search_y,
                          speckle_area, cfg.min_overlap_frac, cfg.min_ncc,
                          cfg.peak_exclusion, center=c, radius=rad)
        out.append(r)
    return out


def detect_blind(ws, frac=0.25):
    """Frames with (almost) no tissue echo after probe-fixed removal = probe
    lift-off / air gap.  They carry no position information: excluded from
    registration and compositing; their pose comes from the motion prior."""
    mass = np.array([float(w.sum()) for w in ws])
    return mass < frac * max(np.median(mass), 1e-6)


def register_guided(fs, ws, cfg, N, speckle_area, t=None, blind=None):
    """Guided multi-lag registration with explicit gap bridging.

    * short lags (<= 3) between sighted frames: full search
    * frames around every blind run (lift-off) and every long time gap are
      bridged by full-search pairs (3 before x 3 after)
    * a robust pre-solve (GNC, image + motion only) of those pairs predicts
      the remaining long-lag pairs, which are searched only within
      R(L) = guide_r0 + guide_r1 * L px of the prediction; pairs spanning a
      blind run or a time gap > cfg.gap_s are always full-search."""
    import copy
    tt = np.arange(N) / 12.0 if t is None else np.asarray(t)
    blind = np.zeros(N, bool) if blind is None else blind
    good = np.where(~blind)[0]
    rank = -np.ones(N, int); rank[good] = np.arange(len(good))
    short = [L for L in cfg.lags if L <= 3] or [1]
    full_pairs = set(); prov = {}
    # short lags counted in *sighted* frames, so a blind run is skipped over
    for L in short:
        for a in range(len(good) - L):
            pr_ = (int(good[a]), int(good[a + L])); full_pairs.add(pr_); prov[pr_] = ("short", L)
    # bridges around blind runs / time gaps
    for a in range(len(good) - 1):
        i, j = good[a], good[a + 1]
        if (j - i) > 1 or (tt[j] - tt[i]) > cfg.gap_s:
            for u in range(3):
                for v in range(3):
                    if a - u >= 0 and a + 1 + v < len(good):
                        pr_ = (int(good[a - u]), int(good[a + 1 + v])); full_pairs.add(pr_)
                        prov.setdefault(pr_, ("bridge", 1 + u + v))
    full_pairs = sorted(full_pairs)
    first = register_all(fs, ws, cfg, full_pairs, speckle_area)
    c = copy.deepcopy(cfg); c.use_landmark = False; c.verbose = False          # GNC / VCE switches inherited from cfg
    pre = solve(build_graph(N, tt, first, None, c), c)
    P = np.stack([pre["p"], pre["q"]], 1)
    pairs, centers, radii, fpairs = [], [], [], []
    done = set(full_pairs)
    for L in cfg.lags:
        if L in short:
            continue
        for a in range(len(good) - L):
            i, j = int(good[a]), int(good[a + L])
            if (i, j) in done:
                continue
            if (j - i) != L or (tt[j] - tt[i]) > cfg.gap_s * L:
                fpairs.append((i, j))                       # spans a blind run / gap
            else:
                pairs.append((i, j)); centers.append(tuple(P[j] - P[i]))
                radii.append(cfg.guide_r0 + cfg.guide_r1 * L)
    rest = register_all(fs, ws, cfg, pairs, speckle_area, centers, radii=radii)
    rest2 = register_all(fs, ws, cfg, fpairs, speckle_area) if fpairs else []
    # provenance: role and intended (sighted-frame) lag of every pair, so that later selection does not depend
    # on the raw frame-index difference (which differs from the lag across blind runs and for gap bridges)
    for r in first:
        r.role, r.lag = prov[(r.i, r.j)]
    for r in rest:
        r.role, r.lag = "long", int(rank[r.j] - rank[r.i])
    for r in rest2:
        r.role, r.lag = "long_gap", int(rank[r.j] - rank[r.i])
    return first + rest + rest2


def initial_probe_fixed(g, pct=20):
    """Trajectory-free first guess of the probe-fixed layer: structure that is
    bright in (almost) every frame at the same pixel is probe-fixed; moving
    anatomy is bright only transiently -> low percentile over time."""
    S = np.stack(g)
    B0 = np.percentile(S, pct, axis=0).astype(np.float32)
    B0 = np.minimum(B0, 0.8 * np.median(S, axis=0))
    return pp.smooth(np.maximum(B0, 0), 1.0)


def sequential_pairs(N, lags):
    return [(i, i + L) for L in lags for i in range(N - L)]


def select_pairs(pairs, lags):
    """Pairs that belong to the lag set `lags`, selected by provenance: guided pairs by their intended
    (sighted-frame) lag, gap bridges always (they are the only constraints across a lift-off / time gap),
    un-guided sequential pairs by their frame-index difference."""
    lagset = set(lags)
    out = []
    for r in pairs:
        role = getattr(r, "role", "")
        if role == "bridge":
            out.append(r)
        elif role in ("short", "long", "long_gap"):
            if r.lag in lagset:
                out.append(r)
        elif (r.j - r.i) in lagset:
            out.append(r)
    return out


def lag_group(L):
    """Lag-stratified variance components: registration error grows with the
    baseline (less overlap, more non-rigid / out-of-plane mismatch), so each
    lag band gets its own VCE-estimated noise scale."""
    if L <= 1:
        return "image_L1"
    if L <= 3:
        return "image_L2-3"
    if L <= 8:
        return "image_L4-8"
    return "image_L9+"


def build_graph(N, t, pair_results, coords, cfg, loop_results=(), blind=None):
    G = Graph(N)
    G.add_gauge(0)
    G.add_motion_prior(t, cfg.motion_sigma_acc)
    if cfg.use_image:
        for r in pair_results:
            if r.ok:
                G.add_relative_2d(r.i, r.j, r.d, r.cov, lag_group(r.j - r.i), rho=r.rho)
    for r in loop_results:
        if r.ok:
            G.add_relative_2d(r.i, r.j, r.d, r.cov, "loop", rho=r.rho)
    if cfg.use_landmark and coords is not None:
        gx = np.array([c.get("gm_x_mm", np.nan) for c in coords], float)
        gy = np.array([c.get("gm_y_mm", np.nan) for c in coords], float)
        qs = np.array([c.get("quality_score", 1.0) for c in coords], float)
        for i in range(N - 1):
            j = i + 1
            if blind is not None and (blind[i] or blind[j]):
                continue
            if np.isfinite(gx[i]) and np.isfinite(gx[j]) and qs[i] > 0.05 and qs[j] > 0.05:
                G.add_relative_1d(i, j, -(gx[j] - gx[i]) / cfg.px_mm_x, cfg.landmark_sigma0, "x", "landmark")
                G.add_relative_1d(i, j, -(gy[j] - gy[i]) / cfg.px_mm_y, cfg.landmark_sigma0, "y", "landmark")
    return G


def loop_closure_pairs(p, q, cfg, quality):
    N = len(p); pairs, centers = [], []
    for i in range(0, N, cfg.lc_stride):
        for j in range(i + cfg.lc_min_lag, N, cfg.lc_stride):
            dx, dy = p[j] - p[i], q[j] - q[i]
            if abs(dx) < cfg.lc_max_pred_dx and abs(dy) < cfg.search_y - 5 \
               and quality[i] > 0.05 and quality[j] > 0.05:
                pairs.append((i, j)); centers.append((dx, dy))
    return pairs, centers


def solve_stack(N, t, pairs, loop, coords, cfg, blind=None):
    """Graph build + robust solve from registrations; the lag set and the cue switches in `cfg` are applied here."""
    pr = select_pairs(pairs, cfg.lags)
    lp = loop if cfg.use_loop_closure else []
    G = build_graph(N, t, pr, coords, cfg, lp, blind=blind)
    return solve(G, cfg, log=lambda *a: _log(cfg, *a))


def stitch_sweep(frames_raw, coords=None, cfg: StitchConfig = None, timestamps=None):
    """Reconstruct one sweep. frames_raw: list of frames (gray or BGR); coords: per-frame dicts with 'ts' (s) and,
    optionally, the network GM position 'gm_x_mm', 'gm_y_mm' and 'quality_score'. Returns a dict with the trajectory
    p, q (px, top-left corner of each frame on the canvas), its standard deviations sd_p, sd_q, the blind mask, the
    probe-fixed layer, the composite and diagnostics."""
    cfg = cfg or StitchConfig()
    T0 = time.time()
    N = len(frames_raw)
    g, fs, ws = condition(frames_raw, cfg)
    if timestamps is not None:
        t = np.asarray(timestamps, float)
    elif coords is not None and all("ts" in c for c in coords):
        t = np.array([c["ts"] for c in coords], float)
    else:
        t = np.arange(N) / 12.0
    t = t - t[0]
    quality = (np.array([c.get("quality_score", 1.0) for c in coords], float)
               if coords is not None else np.ones(N))
    spk = cfg.speckle_area_px or estimate_speckle_area(fs, ws)
    _log(cfg, f"[ioustitch] N={N} frames, speckle cell ~{spk:.1f} px^2")

    pairs = sequential_pairs(N, cfg.lags)
    B = np.zeros_like(fs[0]); fs_cur, ws_cur = fs, ws
    if cfg.layer_separation:
        B = initial_probe_fixed(g)
        spw = pp.static_pattern_weight(g, B)
        fs_cur = [pp.smooth(np.maximum(x - B, 0), cfg.smooth_sigma) for x in g]
        ws_cur = [w * spw for w in ws]
    res = None; pr = None; pr_stage0 = None
    n_layer = (cfg.layer_iters if cfg.layer_separation else 0)
    for it in range(n_layer + 1):
        t1 = time.time()
        # lift-off frames are identified once, on the trajectory-free artefact
        # estimate (most reliable for no-echo detection), and kept thereafter
        if it == 0:
            blind = detect_blind(ws_cur)
            _log(cfg, f"  [blind] {int(blind.sum())} lift-off / no-echo frames")
        if cfg.guided:
            pr = register_guided(fs_cur, ws_cur, cfg, N, spk, t, blind)
        else:
            pr = register_all(fs_cur, ws_cur, cfg, pairs, spk)
            for r_ in pr:
                r_.role, r_.lag = "seq", int(r_.j - r_.i)
        nok = sum(r.ok for r in pr)
        if it == 0:
            pr_stage0 = pr
        _log(cfg, f"  [reg it{it}] {nok}/{len(pr)} pairs ok ({time.time()-t1:.1f}s)")
        G = build_graph(N, t, pr, coords, cfg, blind=blind)
        res = solve(G, cfg, log=lambda *a: _log(cfg, *a))
        _log(cfg, f"  [solve it{it}] span x={np.ptp(res['p']):.1f}px y={np.ptp(res['q']):.1f}px  "
                  f"inliers={ {k: round(v, 2) for k, v in res['inlier_rate'].items()} }")
        if it < n_layer:
            B = pp.estimate_probe_fixed_layer(g, ws, res["p"], res["q"])
            spw = pp.static_pattern_weight(g, B)
            fs_cur = [pp.smooth(np.maximum(x - B, 0), cfg.smooth_sigma) for x in g]
            ws_cur = [w * spw for w in ws]

    pairs_stage0 = pr_stage0
    loop = []
    if cfg.use_loop_closure:
        lp, centers = loop_closure_pairs(res["p"], res["q"], cfg, np.where(blind, 0.0, quality))
        if lp:
            loop = register_all(fs_cur, ws_cur, cfg, lp, spk, centers, cfg.lc_window)
            for r_ in loop:
                r_.role, r_.lag = "loop", int(r_.j - r_.i)
            _log(cfg, f"  [loop] {sum(r.ok for r in loop)}/{len(lp)} re-visit pairs registered")
    res = solve_stack(N, t, pr, loop, coords, cfg, blind)

    frames_clean = [np.maximum(x - B, 0) for x in g] if cfg.layer_separation else g
    qcomp = np.where(blind, 0.0, np.maximum(quality, 1e-3))
    comp = composite(frames_clean, res["p"], res["q"], qcomp, res["sd_p"], ws, cfg)
    res.update(dict(pairs=pr, pairs_stage0=pairs_stage0, loop=loop, probe_fixed_layer=B, t=t, quality=quality,
                    composite=comp, blind=blind, frames_gray=g, frames_clean=frames_clean,
                    tissue=ws, reg_frames=fs_cur, reg_weights=ws_cur,
                    runtime_s=time.time() - T0, speckle_area=spk))
    _log(cfg, f"[ioustitch] done in {res['runtime_s']:.1f}s  sweep = "
              f"{np.ptp(res['p']) * cfg.px_mm_x:.2f} mm")
    return res
