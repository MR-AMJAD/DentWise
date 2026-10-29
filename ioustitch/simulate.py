"""
Generic simulator of intra-oral ultrasound sweeps with ground truth (generic benchmark, Supplementary Section S0).

The clinical data provide no trajectory ground truth; this simulator provides sweeps with known trajectories and
the following nuisances:

  * fully-developed speckle from a tissue-fixed complex scatterer field
    (Rayleigh envelope), partial frame-to-frame decorrelation
    (elevational motion / pressure) mixed in with a fresh field
  * layered periodontal anatomy: coupling surface, gingiva, enamel/root
    surface with acoustic shadow, alveolar bone crest, interproximal gaps
  * PROBE-FIXED artefacts: near-field ring-down line, reverberation bands,
    deep bright blobs (they do NOT move with the tissue)
  * irregular frame times, dwell, speed changes, lift-off gaps with
    un-observed motion, back-and-forth re-visits
  * a simulated probe-orientation (yaw) signal that follows the dental-arch
    curvature, with noise, quantisation, bias drift and dropouts; it is used
    only by the complementary-filter baseline (baselines.cf_imu)
  * a GM "landmark detector" that switches between teeth, has outliers,
    and returns 0.0 when nothing is detected

All geometric conventions follow registration.py:  frame i shows canvas
pixel X at u = X - p_i.
"""
import numpy as np
import cv2


def _gauss_field(shape, sx, sy, rng):
    z = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    re = cv2.GaussianBlur(z.real.astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)
    im = cv2.GaussianBlur(z.imag.astype(np.float32), (0, 0), sigmaX=sx, sigmaY=sy)
    return re + 1j * im


def make_tissue(width, height, rng, oy=0.0, tooth_pitch=(260, 340)):
    """Echogenicity map E(X, Y) in [0,1] + landmark positions."""
    X = np.arange(width)[None, :].astype(np.float32)
    Y = np.arange(height)[:, None].astype(np.float32)
    Y = Y - float(oy)                    # anatomy defined in frame-row units
    E = np.full((height, width), 0.003, np.float32)
    gum_top = 70 + 3 * np.sin(X / 90.0 + rng.uniform(0, 6))                 # gingival surface
    # teeth: sequence of crowns with interproximal dips
    tooth_y = np.full(width, 200.0, np.float32)
    gm_pts, x = [], rng.uniform(-80, 0)
    while x < width + 200:
        pitch = rng.uniform(*tooth_pitch)
        c = x + pitch / 2
        xs = np.arange(width)
        prof = 95 + 40 * ((xs - c) / (pitch / 2)) ** 2 + rng.uniform(-6, 6)   # convex crown/root
        inside = np.abs(xs - c) < pitch / 2
        tooth_y = np.where(inside, np.minimum(tooth_y, prof), tooth_y)
        gm_pts.append(c + rng.uniform(-0.2, 0.2) * pitch)                     # GM landmark
        x += pitch
    tooth_y = cv2.GaussianBlur(tooth_y[None, :], (0, 0), 3)[0]
    bone_y = tooth_y + 55 + 8 * np.sin(X[0] / 50.0)
    ty = tooth_y[None, :]; by = bone_y[None, :]
    gum = (Y > gum_top) & (Y < ty)
    E = np.where(gum, 0.06 + 0.03 * np.sin(X / 23.0) * np.cos(Y / 17.0), E)
    E = np.where(np.abs(Y - ty) < 3.0, 0.95, E)                               # enamel/root line
    E = np.where((Y > ty + 3) & (Y < by), 0.008, E)                            # acoustic shadow
    E = np.where(np.abs(Y - by) < 2.5, 0.40, E)                               # bone crest
    E = np.where(Y > by + 2.5, 0.02, E)
    # a few focal bright reflectors (calculus-like)
    for _ in range(int(width / 60)):
        cx, cy = rng.uniform(0, width), rng.uniform(80, 150)
        E += 0.3 * np.exp(-((X - cx) ** 2 / 18 + (Y - cy) ** 2 / 6)).astype(np.float32)
    return np.clip(E, 0, 1).astype(np.float32), np.array(gm_pts), tooth_y


def make_probe_fixed(H, W, rng, strength=1.0):
    B = np.zeros((H, W), np.float32)
    B[62:68, :] += 0.55 * strength                                            # ring-down / coupling line
    for r in (150, 205, 232):                                                 # reverberation bands
        B[r:r + 2, :] += 0.12 * strength * rng.uniform(0.5, 1.2)
    for _ in range(4):                                                        # deep bright blobs
        cx, cy = rng.uniform(20, W - 20), rng.uniform(200, 245)
        yy, xx = np.mgrid[:H, :W]
        B += 0.35 * strength * np.exp(-((xx - cx) ** 2 / 60 + (yy - cy) ** 2 / 8)).astype(np.float32)
    return cv2.GaussianBlur(B, (0, 0), 1.0)


def frame_affine(rot_deg, scale, X0, Y0, W, H):
    """Rendering geometry of one frame: frame pixel u shows canvas A2 (u - c) + c + (X0, Y0) (small probe tilt
    about the frame centre + axial compression). Returns the 2x3 matrix for cv2.warpAffine(WARP_INVERSE_MAP) and
    a function mapping canvas points (2, n) to frame pixel coordinates (the inverse transform)."""
    a_ = np.radians(rot_deg); cs, sn = np.cos(a_), np.sin(a_)
    A2 = np.array([[cs, -sn * scale], [sn, cs * scale]], np.float32)
    cvec = np.array([W / 2, H / 2], np.float32)
    tvec = cvec + np.array([X0, Y0], np.float32) - A2 @ cvec
    M = np.hstack([A2, tvec[:, None]]).astype(np.float32)

    def to_frame(Xc):
        Xc = np.asarray(Xc, float)
        return np.linalg.solve(A2.astype(float), Xc - (cvec + np.array([X0, Y0]))[:, None]) + cvec[:, None]
    return M, to_frame


def make_trajectory(rng, scenario, n_target=90):
    """Returns t, p, q (px) for n frames."""
    t = [0.0]
    while len(t) < n_target:
        dt = float(np.clip(rng.normal(0.085, 0.012), 0.05, 0.14))
        if scenario in ("jerky", "mixed") and rng.random() < 0.03:
            dt = rng.uniform(0.4, 1.2)                                        # gated gap / lift-off
        t.append(t[-1] + dt)
    t = np.array(t)
    T = t[-1]
    # speed profile (px/s)
    knots = np.linspace(0, T, 7)
    v = rng.uniform(15, 70, size=knots.size)                             # 0.4-1.9 mm/s
    if scenario in ("dwell", "mixed", "jerky"):
        v[rng.integers(1, knots.size - 1)] = 0.0                              # dwell
    vt = np.interp(t, knots, v)
    if scenario in ("reversal", "mixed"):
        a, b = sorted(rng.uniform(0.3, 0.75, 2) * T)
        b = max(b, a + 1.2)
        vt = np.where((t > a) & (t < b), -0.8 * vt, vt)                       # back-and-forth revisit
    dt = np.diff(t, prepend=t[0])
    p = np.cumsum(vt * dt)
    # vertical: pressure drift + small tremor
    q = 8 * np.sin(2 * np.pi * t / rng.uniform(3, 7) + rng.uniform(0, 6)) + \
        np.cumsum(rng.normal(0, 0.4, t.size))
    return t, p - p[0], q - q[0]


def simulate_sweep(seed=0, scenario="mixed", n_frames=90, H=256, W=256,
                   artifact_strength=0.6, decorrelation=0.30, liftoff_rate=0.03,
                   elevation_mod=0.35, max_rot_deg=0.8, max_scale=0.015,
                   px_mm=None, px_mm_y=None):
    # pixel spacings of the stored frame (mm/px): identical to the reconstruction's StitchConfig, so the GM detector's
    # millimetre coordinates convert back to the same pixels (lateral 7.0/256, axial 7.344/256 mm)
    from .config import StitchConfig
    px_mm = StitchConfig.px_mm_x if px_mm is None else px_mm
    px_mm_y = StitchConfig.px_mm_y if px_mm_y is None else px_mm_y
    rng = np.random.default_rng(seed)
    t, p, q = make_trajectory(rng, scenario, n_frames)
    marg = 60
    TW = int(np.ceil(p.max() - p.min())) + W + 2 * marg
    TH = int(np.ceil(q.max() - q.min())) + H + 2 * marg
    ox, oy = marg - p.min(), marg - q.min()                                   # canvas offset
    E, gm_pts, tooth_y = make_tissue(TW, TH, rng, oy=oy)
    S = _gauss_field((TH, TW), 1.6, 0.9, rng)                                 # tissue-fixed scatterers
    Bfix = make_probe_fixed(H, W, rng, artifact_strength)
    # out-of-plane (elevational) appearance change: two smooth tissue-fixed
    # modulation fields mixed with a slowly rotating phase
    Ga = cv2.GaussianBlur(rng.standard_normal((TH, TW)).astype(np.float32), (0, 0), 6)
    Gb = cv2.GaussianBlur(rng.standard_normal((TH, TW)).astype(np.float32), (0, 0), 6)
    Ga /= Ga.std() + 1e-6; Gb /= Gb.std() + 1e-6
    th = np.cumsum(rng.normal(0, 0.08, t.size))
    rot = cv2.GaussianBlur(rng.normal(0, max_rot_deg, (1, t.size)).astype(np.float32), (0, 0), 3)[0] * 2
    scl = 1 + cv2.GaussianBlur(rng.normal(0, max_scale, (1, t.size)).astype(np.float32), (0, 0), 3)[0] * 2

    # arch curvature field (deg/mm) along canvas x  -> yaw
    kx = np.linspace(0, TW, 6)
    kap = rng.uniform(0.3, 4.0, kx.size) * rng.choice([-1, 1])
    xs = np.arange(TW, dtype=float)
    Ycum = np.concatenate([[0.0], np.cumsum(0.5 * (np.interp(xs[1:], kx, kap) +
                                                   np.interp(xs[:-1], kx, kap)))]) * px_mm
    yaw_gt = np.interp(p + ox, xs, Ycum) - np.interp(ox, xs, Ycum)          # deg
    bias = np.cumsum(rng.normal(0, 0.004, t.size)) + 0.01 * t
    yaw_meas = -150 + yaw_gt + bias + rng.normal(0, 0.05, t.size)
    yaw_meas = np.round(yaw_meas / (180 / 32768)) * (180 / 32768)
    imu_valid = rng.random(t.size) > 0.03

    frames, coords = [], []
    gain = 1.0
    # lift-off episodes of 1-4 consecutive frames (air gap)
    lift = np.zeros(t.size, bool)
    k = 2
    while k < t.size - 2:
        if rng.random() < liftoff_rate:
            L = int(rng.integers(1, 5)); lift[k:k + L] = True; k += L + 3
        k += 1
    last_gm = None
    gm_true = []
    for i in range(t.size):
        X0, Y0 = p[i] + ox, q[i] + oy
        # frame(u) = canvas(R S (u - c) + c + X0): small probe tilt + axial compression
        M, to_frame = frame_affine(rot[i], scl[i], X0, Y0, W, H)
        crop = lambda A: cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        mod = np.cos(th[i]) * crop(Ga) + np.sin(th[i]) * crop(Gb)
        e = crop(E) * np.clip(1 + elevation_mod * mod, 0.2, 3.0)
        s = crop(S.real.astype(np.float32)) + 1j * crop(S.imag.astype(np.float32))
        fresh = _gauss_field((H, W), 1.6, 0.9, rng)
        a = np.sqrt(1 - decorrelation)
        env = np.abs(a * s + np.sqrt(decorrelation) * fresh)
        env = env / 0.6
        gain = float(np.clip(gain * np.exp(rng.normal(0, 0.02)), 0.8, 1.25))
        depth_att = np.exp(-np.arange(H) / 400.0)[:, None]
        lin = gain * e * env * depth_att
        DR = 30.0                                                             # dB display dynamic range
        img = np.clip((20 * np.log10(np.maximum(lin, 1e-6)) + DR) / DR, 0, 1)
        if lift[i]:
            img = np.zeros_like(img)          # air gap: no tissue echo at all
        img = img + Bfix
        img = img + np.abs(rng.normal(0, 0.015, (H, W)))
        img = np.clip(img, 0, 1)
        frames.append(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

        # GM landmark detector. True GM positions in this frame follow the SAME geometry as the rendered image:
        # frame pixel u shows canvas A2 (u - c) + c + (X0, Y0), so u = A2^-1 (X - c - (X0, Y0)) + c.
        # The anatomy is defined in rows Y - oy, i.e. the GM of tooth k lies at canvas (gm_k, tooth_y - 18 + oy).
        gxc = np.clip(gm_pts, 0, TW - 1).astype(int)
        u, v_gm = to_frame(np.stack([gm_pts, tooth_y[gxc] - 18 + oy], 0))
        cand = np.where((u > 0) & (u < W))[0]
        gm_true.append((u[cand[np.argmin(np.abs(u[cand] - W / 2))]], v_gm[cand[np.argmin(np.abs(u[cand] - W / 2))]])
                       if cand.size else (np.nan, np.nan))
        vis = np.where((u > 12) & (u < W - 12))[0]
        if lift[i] or vis.size == 0 or rng.random() < 0.05:
            # not detected: the smoothed display cache holds the last value
            gmx, gmy = (last_gm if last_gm is not None else (W / 2 * px_mm, 80 * px_mm_y))
            gmx += rng.normal(0, 0.05); gmy += rng.normal(0, 0.05)
            qual = 0.05 if lift[i] else 0.2
        else:
            k = vis[np.argmin(np.abs(u[vis] - W / 2))]                        # detector switches teeth
            ux = u[k] + rng.normal(0, 1.5)
            uy = v_gm[k] + rng.normal(0, 1.5)
            if rng.random() < 0.05:
                ux = rng.uniform(10, W - 10)                                  # wrong structure
            gmx, gmy = ux * px_mm, uy * px_mm_y              # lateral / axial pixel spacing (as StitchConfig)
            last_gm = (gmx, gmy)
            qual = float(np.clip(rng.normal(0.4, 0.03), 0.2, 0.5))
        coords.append(dict(frame_n=i, ts=float(t[i]), gm_x_mm=float(gmx), gm_y_mm=float(gmy),
                           quality_score=qual, imu_yaw=float(yaw_meas[i]),
                           imu_valid=bool(imu_valid[i]), imu_gyro_z=0.0))
    gt = dict(p=p, q=q, t=t, lift=lift, tissue=E, origin=(ox, oy), yaw_gt=yaw_gt, gm_true=np.array(gm_true),
              gm_canvas=np.stack([gm_pts, tooth_y[np.clip(gm_pts, 0, TW - 1).astype(int)] - 18 + oy], 0),
              rot=rot, scale=scl,
              probe_fixed=Bfix)
    return frames, coords, gt
