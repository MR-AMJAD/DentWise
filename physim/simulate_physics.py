"""
Physics-based simulator of apico-coronal intra-oral ultrasound sweeps at 40 MHz, with ground truth.

Image formation (linear convolution model; Gao et al. / COLE-type, PSF from PyMUST-SIMUS):
  * Two-way point-spread function of a 40-MHz, 256-element linear array (pitch 27.3 um, line-by-line focused
    transmission, dynamic receive focusing) computed with PyMUST/SIMUS at 11 depths (psf.py); bandwidth (45 %)
    and transmit/receive F-numbers (7/6) calibrated to the clinical speckle size; spatially variant convolution
    by linear interpolation between depth nodes.
  * Tissue: 2-D apico-coronal section of the buccal/lingual side of a tooth, extruded in elevation:
    coupling stand-off (gel), gingiva / alveolar mucosa (diffuse scattering, fibre texture), buccal bone
    plate ending at the alveolar bone crest (ABC), periodontal ligament, root (cementum/dentin), crown
    (enamel) from the cemento-enamel junction (CEJ), gingival margin (GM).
  * Interfaces are coherent (specular) reflectors with the normal-incidence reflection coefficients of the
    adjacent media (acoustic impedances, MRayl: gel 1.50, soft tissue 1.63, cortical bone 7.4, dentin 8.0,
    enamel 16.5), EXCEPT the gel / keratinised-epithelium surface, which uses a phenomenological effective
    coefficient of 0.35 (the impedance value would be ~0.04) that also sets the transmission loss below it;
    angle-dependent return (Gaussian in incidence angle, 20 deg) plus a diffuse part.
  * Approximation: each slice is convolved once on the canvas and warped to every probe pose, so tilt and
    compression deform the PSF with the tissue and attenuation / incidence angles are not recomputed per pose
    (PSF-based approximation, not per-frame acoustic propagation).
  * Two-way attenuation (soft tissue 1.0 dB/cm/MHz, hard tissue 30 dB/mm at 40 MHz) and transmission
    losses (1 - R^2) along each A-line -> acoustic shadow below bone and tooth.
  * Elevation: 21 tissue-fixed scatterer slices (40 um apart) weighted by the depth-dependent elevation
    sensitivity from SIMUS-3D; the anatomy is three-dimensional (interface depth varies with elevation, rough
    part of each reflection independent between slices), so out-of-plane probe motion decorrelates speckle and
    interfaces physically (rate calibrated to the clinical frame-to-frame decorrelation).
  * Probe-fixed structure (moves with the probe, not with the tissue): lens / matching-layer ring-down,
    reverberation of an attached stand-off at fixed depths, small gas bubbles in the coupling gel adhering
    to the lens (bright point + shadow column, multiplicative), one weak element (column gain).
  * Lift-off: air under the lens -> no tissue echo, total reflection -> strong probe-fixed ring-down.
  * Electronic noise, time-gain compensation, envelope detection, log compression (40 dB) with a grey map
    (gamma 1.6; levels calibrated to the clinical grey-level histogram), 8-bit display,
    4 x 4 area averaging to the 256 x 256 stored frame (27.3 x 28.7 um pixels, as in the clinical data).
Motion: same trajectory generator as the generic simulator (speed changes, dwell, reversal, lift-off
gaps, pressure drift) + in-plane tilt following the tissue surface + elevational wander.
Outputs follow ioustitch.simulate.simulate_sweep: frames (BGR uint8), coords (per-frame dicts with the
GM 'landmark detector' and the simulated probe-orientation fields), gt (p, q, t, lift, probe_fixed, landmarks).
"""
import numpy as np, cv2, os, sys
from scipy.signal import fftconvolve
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder that contains ioustitch/
from ioustitch.simulate import make_trajectory
F = 4                                          # fine-grid factor (fine pixel 6.84 x 7.17 um)
PXL, PXA = 7.0 / 256, 7.344 / 256              # stored-frame pixel (mm): lateral, axial
Z = dict(gel=1.50, soft=1.63, bone=7.4, dentin=8.0, enamel=16.5)


def refl(z1, z2):
    return (Z[z2] - Z[z1]) / (Z[z2] + Z[z1])


_PSF = {}
PSF_FILE = os.environ.get('PHYSIM_PSF', 'psf_40MHz_cal4.npz')   # cal3 PSF + elevation sensitivity tabulated to +-1.0 mm
SIM_VERSION = '2026-10-11'      # bump when the image formation, anatomy, motion or detector model changes


def cache_version():
    """Fingerprint of everything that generates a cached sequence: model version, PSF file content, and the source
    of this simulator and of the shared trajectory generator (ioustitch/simulate.py: make_trajectory). The job
    parameters (scenario settings, frame count, seed) are fingerprinted separately per sequence (gen_cache.cache_key)."""
    import hashlib
    def sha(p):
        return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]
    import ioustitch.simulate as _gs
    return (f'{SIM_VERSION}|{PSF_FILE}:{sha(os.path.join(HERE, PSF_FILE))}|simulate_physics.py:{sha(os.path.abspath(__file__))}'
            f'|simulate.py:{sha(_gs.__file__)}')


def load_psf(name=None):
    name = name or PSF_FILE
    if name not in _PSF:
        d = np.load(os.path.join(HERE, name))
        P = dict(psf=d['psf'], depths=d['depths'] * 1e3, eoffs=d['eoffs'] * 1e3, elev=d['elev'])
        P['peak'] = np.abs(P['psf']).max()
        _PSF[name] = P
    return _PSF[name]


def smooth_curve(n, rng, amp, scale):
    """Smooth random curve of length n (fine pixels), std ~ amp, correlation length ~ scale (fine px)."""
    c = cv2.GaussianBlur(rng.standard_normal((1, n)).astype(np.float32), (0, 0), scale)[0]
    return amp * c / (c.std() + 1e-9)


def make_anatomy(TWf, THf, r_face, rng, x_mid):
    """Depths (fine px, canvas rows) of the tissue surfaces for every canvas column, plus landmark columns.
    x runs from apical (left) to coronal (right)."""
    x = np.arange(TWf)
    mm = lambda v: v / (PXL / F)                      # mm -> fine lateral px
    mz = lambda v: v / (PXA / F)                      # mm -> fine axial px
    # landmarks (lateral): CEJ near the middle of the swept region, GM coronal, ABC apical
    xC = x_mid + mm(rng.uniform(-1.0, 1.0))
    xG = xC + mm(rng.uniform(0.6, 2.0))               # healthy: GM coronal to CEJ
    xA = xC - mm(rng.uniform(1.4, 2.6))               # ABC 1.4-2.6 mm apical to CEJ
    # tooth surface: inclined line + convex crown (height of contour) + small waviness
    phi = np.radians(rng.choice([-1, 1]) * rng.uniform(4, 14))
    zC = r_face + mz(rng.uniform(2.6, 3.4))
    zt = zC + np.tan(phi) * (x - xC) * (PXL / PXA)
    bulge = mz(rng.uniform(0.15, 0.4)) * np.exp(-((x - (xC + mm(1.8))) / mm(1.6)) ** 2)
    zt = zt - bulge + smooth_curve(TWf, rng, mz(0.02), mm(0.6))
    # soft tissue thickness over the tooth: thick apically, tapering to the free margin at GM
    h_api = mz(rng.uniform(1.5, 2.2))
    h_gm = mz(rng.uniform(0.35, 0.6))
    s = np.clip((xG - x) / mm(3.0), 0, 1)
    h = h_gm + (h_api - h_gm) * s ** 0.7 + smooth_curve(TWf, rng, mz(0.04), mm(0.8))
    zs = np.where(x <= xG, zt - h, zt)                # outer tissue surface (gel below the lens above it)
    zs = np.maximum(zs, r_face + mz(0.6))
    # sulcus: tiny notch at the margin
    # bone plate: from apical to ABC, outer surface between soft-tissue surface and root, tapering at the crest
    pdl = mz(0.15)
    th_b = mz(rng.uniform(0.5, 0.9)) * np.clip((xA - x) / mm(1.2), 0, 1) ** 0.6
    zb = zt - pdl - th_b                              # outer bone surface (== zt - pdl at the crest)
    bone = x < xA
    return dict(zs=zs, zt=zt, zb=zb, bone=bone, xG=xG, xC=xC, xA=xA, pdl=pdl, phi=phi)


ALPHA_SOFT = 1.0          # dB/cm/MHz, gingiva / mucosa (one-way)


def build_reflectivity(A, TWf, THf, r_face, rng, K, fibre, yk=None, iface_scale=1.0, elev_anat=0.3, iface_coh=0.7):
    """Complex reflectivity of K elevation slices (tissue-fixed): diffuse scattering (independent between slices)
    + interfaces. The anatomy is three-dimensional: in slice k (elevation y_k) every interface lies at
    z_i(x) + y_k G_i(x), with G_i a smooth random elevational slope (std elev_anat), and its reflection has a
    coherent part (fraction iface_coh, identical in all slices) and a rough part (independent between slices).
    Two-way attenuation and transmission losses along each A-line are included."""
    x = np.arange(TWf)[None, :]
    zz = np.arange(THf)[:, None].astype(np.float32)
    zs, zt, zb, bone = A['zs'][None], A['zt'][None], A['zb'][None], A['bone'][None]
    crown = np.arange(TWf)[None] >= A['xC']
    soft = (zz >= zs) & (zz < np.where(bone, zb, zt))
    pdl = bone & (zz >= zt - A['pdl']) & (zz < zt)
    hard = ((zz >= zb) & bone & ~pdl) | (zz >= zt)
    # attenuation (Np per fine axial px, one-way)
    dz_mm = PXA / F
    a_soft = ALPHA_SOFT * 40 / 10 / 8.686 * dz_mm     # soft tissue at 40 MHz
    a_hard = 30.0 / 8.686 * dz_mm                     # 30 dB/mm
    a_gel = 0.002 * 40 ** 2 / 10 / 8.686 * dz_mm
    alpha = np.where(hard, a_hard, np.where(soft | pdl, a_soft, a_gel)).astype(np.float32)
    alpha[:r_face] = 0
    att = np.exp(-2 * np.cumsum(alpha, axis=0))       # two-way amplitude
    # interfaces (row positions per column) and reflection coefficients
    R_gs = refl('gel', 'soft'); R_ge = refl('gel', 'enamel'); R_se = refl('soft', 'enamel')
    R_sb = refl('soft', 'bone'); R_sd = refl('soft', 'dentin')
    cols = np.arange(TWf)
    ifaces = []
    xg = cols <= A['xG']
    ifaces.append((A['zs'], np.where(xg, 0.35, 0.0) + 0.0, xg))                       # gel / keratinised epithelium
    ifaces.append((A['zt'], np.where(crown[0], np.where(xg, R_se, R_ge), R_sd), ~A['bone'] | crown[0]))  # tooth
    ifaces.append((A['zb'], np.full(TWf, R_sb), A['bone']))                            # bone plate
    # transmission losses below each interface (two-way amplitude: 1 - R^2)
    trans = np.ones((THf, TWf), np.float32)
    for zi, Ri, m in ifaces:
        below = (zz > zi[None]) & m[None]
        trans *= np.where(below, 1 - Ri[None] ** 2, 1.0)
    att *= trans
    # diffuse backscatter level (amplitude): soft tissue with fibre texture, PDL weaker, epithelium brighter
    lev = np.zeros((THf, TWf), np.float32)
    lev[soft] = 1.0
    epi = soft & (zz < zs + 0.12 / dz_mm)
    lev[epi] = 1.5
    lev[pdl] = 0.5
    lev[hard] = 0.25                                  # hard-tissue scattering (fully attenuated anyway)
    gel_lvl = 0.01
    lev[(zz < zs) & (zz >= r_face)] = gel_lvl
    lev *= fibre
    lev *= att
    # interfaces: per-slice depth (3-D anatomy) and rough/coherent reflection (sub-pixel row split)
    spec = None
    k2 = 4 * np.pi * 40e6 / 1540 * dz_mm * 1e-3      # 2 k0 per fine axial px
    yk = np.zeros(K) if yk is None else np.asarray(yk)
    IF = []
    for zi, Ri, m in ifaces:
        slope = np.gradient(zi) * (PXA / PXL)         # dz/dx in physical units
        th = np.arctan(slope)
        Aang = 0.15 + 0.85 * np.exp(-(np.degrees(th) / 20.0) ** 2)
        rough = np.exp(0.45 * smooth_curve(TWf, rng, 1.0, 25)) * (1 + 0.2 * rng.standard_normal(TWf))
        amp = iface_scale * Ri * Aang * m * rough * att[np.clip(zi.astype(int) - 1, 0, THf - 1), cols]
        phi = 0.6 * rng.standard_normal(TWf)
        G = smooth_curve(TWf, rng, elev_anat, 120)                     # elevational slope dz/dy
        IF.append((zi, amp, phi, G, m))
    slices = []
    for k in range(K):
        z = (rng.standard_normal((THf, TWf)) + 1j * rng.standard_normal((THf, TWf))).astype(np.complex64)
        sl = lev * z
        for zi, amp, phi, G, m in IF:
            zk = zi + yk[k] * G / (PXA / F)                       # depth of this interface in slice k
            nk = cv2.GaussianBlur(rng.standard_normal((2, TWf)).astype(np.float32), (0, 0), 1.5)
            nk = (nk[0] + 1j * nk[1]) / max(np.abs(nk[0] + 1j * nk[1]).std() * 1.0, 1e-6) / np.sqrt(2)
            a = amp * (iface_coh + (1 - iface_coh) * nk) * np.exp(-1j * (k2 * zk + phi))
            r0 = np.floor(zk).astype(int); fr = zk - r0
            ok = (r0 >= 0) & (r0 < THf - 1) & m
            sl[r0[ok], cols[ok]] += (a * (1 - fr))[ok]
            sl[r0[ok] + 1, cols[ok]] += (a * fr)[ok]
        slices.append(sl)
    return slices, spec, dict(soft=soft, hard=hard)


def depth_conv(field, r_face, P):
    """Spatially variant convolution with the SIMUS PSF (linear interpolation between depth nodes)."""
    THf = field.shape[0]
    dep = (np.arange(THf) - r_face) * PXA / F          # mm from the lens
    nodes = P['depths']
    out = np.zeros_like(field)
    for j, d in enumerate(nodes):
        lo = nodes[j - 1] if j > 0 else -1e9
        hi = nodes[j + 1] if j < len(nodes) - 1 else 1e9
        w = np.where(dep <= d, np.clip((dep - lo) / (d - lo), 0, 1), np.clip((hi - dep) / (hi - d), 0, 1))
        if j == 0: w = np.where(dep <= d, 1.0, w)
        if j == len(nodes) - 1: w = np.where(dep >= d, 1.0, w)
        rows = np.nonzero(w > 0)[0]
        if rows.size == 0:
            continue
        a, b = max(rows.min() - 60, 0), min(rows.max() + 60, THf)
        k = (P['psf'][j][:, ::-1] / P['peak']).astype(np.complex64)
        c = fftconvolve(field[a:b], k, mode='same')
        out[a:b] += (w[a:b, None] * c).astype(np.complex64)
    return out


def elev_weights(P, dep_mm, yk, e):
    """Two-way elevational sensitivity at each depth for slice offsets yk - e (mm). The SIMUS-3D profile is
    tabulated over P['eoffs']; offsets outside that grid raise an error instead of being extrapolated
    (the tabulated grid must cover max|yk| + max|e|)."""
    E = P['elev'] / P['elev'][:, np.argmin(np.abs(P['eoffs']))][:, None]   # relative to the on-axis value per depth
    lo, hi = P['eoffs'][0], P['eoffs'][-1]
    out = np.zeros((len(yk), dep_mm.size), np.float32)
    for k, y in enumerate(yk):
        if not (lo - 1e-9 <= y - e <= hi + 1e-9):
            raise ValueError(f'elevation offset {y - e:.3f} mm outside the tabulated sensitivity ({lo:.2f}..{hi:.2f} mm)')
        prof = np.array([np.interp(y - e, P['eoffs'], E[j]) for j in range(E.shape[0])])
        out[k] = np.interp(dep_mm, P['depths'], prof)
    return out


def probe_fixed_iq(H, W, rng, strength, P, lift=False):
    """Probe-fixed base-band field on the fine frame grid and multiplicative column gain (shadows, weak element)."""
    Hf, Wf = H * F, W * F
    f = np.zeros((Hf, Wf), np.complex64)
    xs = np.arange(Wf)
    sens = 1 + cv2.GaussianBlur(rng.standard_normal((1, Wf)).astype(np.float32), (0, 0), 40)[0] * 3
    sens = np.clip(sens, 0.6, 1.4)
    dz_mm = PXA / F
    k2 = 4 * np.pi * 40e6 / 1540 * dz_mm * 1e-3
    # lens / matching-layer ring-down: decaying echoes at multiples of the lens round trip
    dL = rng.uniform(0.16, 0.22)
    for m in range(1, 6):
        r = int(m * dL / dz_mm)
        if r < Hf:
            f[r, :] += (0.25 * 0.55 ** m * sens * (1.0 if lift else 0.03)) * np.exp(-1j * k2 * r)
    # reverberation of an attached stand-off at fixed deep depths (partial lateral extent)
    for _ in range(int(rng.integers(2, 4))):
        r = int(rng.uniform(5.0, 6.8) / dz_mm)
        c0, wd = rng.uniform(0.15, 0.85) * Wf, rng.uniform(0.03, 0.12) * Wf
        prof = np.exp(-((xs - c0) / wd) ** 2) * rng.uniform(0.008, 0.025) * (2.5 if lift else 1.0)
        f[r, :] += prof * sens * np.exp(-1j * (k2 * r + 0.3 * rng.standard_normal(Wf)))
    # gas bubbles adhering to the lens: bright point reflectors + shadow column below
    gain = np.ones(Wf, np.float32)
    for _ in range(int(rng.integers(0, 3))):
        c0 = int(rng.uniform(0.05, 0.95) * Wf); r = int(rng.uniform(0.15, 0.6) / dz_mm)
        wd = int(rng.uniform(4, 10))
        f[r, max(c0 - wd, 0):c0 + wd] += rng.uniform(0.15, 0.4)
        gain *= 1 - 0.55 * np.exp(-((xs - c0) / (1.6 * wd)) ** 2)
    # weak element
    c0 = int(rng.uniform(0.1, 0.9) * Wf)
    gain *= 1 - 0.4 * np.exp(-((xs - c0) / 6.0) ** 2)
    f = depth_conv_frame(f * strength, P)
    return f, gain


def fine_affine(ang, sc, X0, Y0, Wf):
    """Geometry of one frame on the fine grid: frame fine pixel (u, v) shows canvas A2 ((u, v) - c) + c + (X0, Y0)
    with c = (Wf/2, 0) (tilt about the centre of the lens, axial compression). Returns the warpAffine matrix
    (WARP_INVERSE_MAP) and the inverse mapping canvas fine coordinates (2, n) -> frame fine coordinates."""
    a_ = np.radians(ang); cs, sn = np.cos(a_), np.sin(a_)
    A2 = np.array([[cs, -sn * sc], [sn, cs * sc]], np.float32)
    cvec = np.array([Wf / 2, 0], np.float32)
    tvec = cvec + np.array([X0, Y0], np.float32) - A2 @ cvec
    Mx = np.hstack([A2, tvec[:, None]]).astype(np.float32)

    def to_frame(Xc):
        Xc = np.asarray(Xc, float)
        return np.linalg.solve(A2.astype(float), Xc - (cvec + np.array([X0, Y0]))[:, None]) + cvec[:, None]
    return Mx, to_frame


def fine_to_stored(uf):
    """Fine-grid pixel coordinate -> stored-frame pixel coordinate (4 x 4 area averaging: stored pixel k covers
    fine pixels F k .. F k + F - 1, centre F k + (F - 1) / 2)."""
    return (np.asarray(uf, float) - (F - 1) / 2) / F


def depth_conv_frame(field, P):
    return depth_conv(field, 0, P)


def simulate_sweep_phys(seed=0, scenario='mixed', n_frames=90, H=256, W=256, artifact_strength=1.0,
                        elev_rate=0.05, liftoff_rate=0.03, max_rot_deg=0.8, K=21, dy=0.04, max_scale=0.01, elev_clip=0.25,
                        DR=40.0, noise_db=-58.0, speckle_db=-27.0, Lref=19.0, iface_db=-2.0, gamma=1.6, elev_anat=1.25, iface_coh=0.2,
                        return_debug=False):
    rng = np.random.default_rng(seed + 1000)
    P = load_psf()
    t, p, q = make_trajectory(rng, scenario, n_frames)
    p = np.round(p * F) / F; q = np.round(q * F) / F          # positions on the fine grid
    marg = 16
    TW = int(np.ceil(p.max() - p.min())) + W + 2 * marg
    TH = int(np.ceil(q.max() - q.min())) + H + 2 * marg
    ox, oy = marg - p.min(), marg - q.min()
    TWf, THf = TW * F, TH * F
    r_face = int(round(oy * F))                                 # lens row at q = 0
    x_mid = (ox + (p.max() + p.min()) / 2 + W / 2) * F
    A = make_anatomy(TWf, THf, r_face, rng, x_mid)
    fib = cv2.GaussianBlur(rng.standard_normal((THf, TWf)).astype(np.float32), (0, 0), sigmaX=10, sigmaY=2.5)
    fibre = np.exp(0.55 * fib / (fib.std() + 1e-9)).astype(np.float32)          # log-normal fibre texture
    yk = (np.arange(K) - (K - 1) / 2) * dy
    slices, spec, M = build_reflectivity(A, TWf, THf, r_face, rng, K, fibre, yk=yk,
                                         iface_scale=10 ** (iface_db / 20) / np.sqrt(15) / (0.02 * 10 ** (speckle_db / 20) * np.sqrt(dy / 0.04)),
                                         elev_anat=elev_anat, iface_coh=iface_coh)
    sl_scale = 0.02 * 10 ** (speckle_db / 20) * np.sqrt(dy / 0.04)                         # scatterer amplitude per slice (sets the
    # specular-to-speckle ratio: bone/enamel surface echoes ~30-40 dB above gingival speckle)
    IQs = [depth_conv(s * sl_scale, r_face, P) for s in slices]
    del slices
    # probe-fixed field and lift-off variant
    Bf, colgain = probe_fixed_iq(H, W, np.random.default_rng(seed + 7), artifact_strength, P)
    # lift-off: the coupling gel stays on the lens and loses contact with the tissue -> total reflection at the
    # gel/air boundary (depth = current gel thickness) and its reverberations between lens and boundary
    dzf = PXA / F
    k2 = 4 * np.pi * 40e6 / 1540 * dzf * 1e-3

    def gel_air(g_rows, rr):
        f = np.zeros((H * F, W * F), np.complex64)
        tilt = rr.normal(0, 0.01) * np.arange(W * F)
        for m, a in ((1, 1.0), (2, 0.30), (3, 0.09), (4, 0.027)):
            rows = (m * (g_rows + tilt)).astype(int)
            ok = (rows > 0) & (rows < H * F)
            amp = a * 3.0 * (1 + 0.1 * rr.standard_normal(W * F))
            f[rows[ok], np.arange(W * F)[ok]] += (amp * np.exp(-1j * k2 * rows))[ok]
        return depth_conv_frame(f, P)
    # motion: elevation wander, in-plane tilt following the tissue surface, lift-off episodes
    e = np.cumsum(rng.normal(0, elev_rate, t.size)) + cv2.GaussianBlur(
        rng.normal(0, 1, (1, t.size)).astype(np.float32), (0, 0), 6)[0] * 0.15
    e = np.clip(e - e.mean(), -elev_clip, elev_clip)
    scl = 1 + cv2.GaussianBlur(rng.normal(0, max_scale, (1, t.size)).astype(np.float32), (0, 0), 3)[0] * 2   # pressure
    surf = cv2.GaussianBlur(A['zs'][None].astype(np.float32), (0, 0), 120)[0]
    slope_deg = np.degrees(np.arctan(np.gradient(surf) * (PXA / PXL)))
    rot = np.array([0.3 * np.interp((p[i] + ox + W / 2) * F, np.arange(TWf), slope_deg) for i in range(t.size)])
    rot = rot - rot.mean()
    rot = np.clip(rot, -1.5, 1.5) + cv2.GaussianBlur(rng.normal(0, max_rot_deg, (1, t.size)).astype(np.float32),
                                                      (0, 0), 3)[0] * 2
    lift = np.zeros(t.size, bool)
    k = 2
    while k < t.size - 2:
        if rng.random() < liftoff_rate:
            L = int(rng.integers(1, 5)); lift[k:k + L] = True; k += L + 3
        k += 1
    dep_mm = np.arange(H * F) * PXA / F
    tgc = np.exp(2 * (0.5 * 40 / 10 / 8.686) * dep_mm)[:, None].astype(np.float32)   # compensates 0.5 dB/cm/MHz
    noise_amp = 10 ** (noise_db / 20)
    # display reference: fixed so that strong specular echoes saturate (set once per simulator, not per frame)
    frames, coords, disp_B = [], [], []
    gain = 1.0
    last_gm = None
    lm_frame = []
    Wf, Hf = W * F, H * F

    def crop(Afield, X0, Y0, ang, sc=1.0):
        # frame fine pixel (u, v) <- canvas (R S (u - c) + c + (X0, Y0)): tilt about the lens centre, axial scale
        Mx, _ = fine_affine(ang, sc, X0, Y0, Wf)
        re = cv2.warpAffine(np.ascontiguousarray(Afield.real), Mx, (Wf, Hf), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        im = cv2.warpAffine(np.ascontiguousarray(Afield.imag), Mx, (Wf, Hf), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return re + 1j * im

    def to_display(iq):
        env = np.abs(iq) * tgc
        L = 20 * np.log10(np.maximum(env, 1e-9))
        img = np.clip((L - (Lref - DR)) / DR, 0, 1) ** gamma          # grey map (post-processing curve)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img

    for i in range(t.size):
        X0, Y0 = (p[i] + ox) * F, (q[i] + oy) * F - 0.0
        Y0 = Y0 - r_face + r_face                              # lens row in canvas = (q + oy) F
        nrm = np.random.default_rng(seed * 7919 + i)
        if lift[i]:
            xc = int(np.clip((p[i] + ox + W / 2) * F, 0, TWf - 1))
            g_rows = max(A['zs'][xc] - (q[i] + oy) * F, 20.0) * nrm.uniform(0.85, 1.0)
            iq = gel_air(g_rows, nrm) + Bf
        else:
            Wk = elev_weights(P, dep_mm, yk, e[i])
            iq = np.zeros((Hf, Wf), np.complex64)
            for kk in range(K):
                iq += Wk[kk][:, None] * crop(IQs[kk], X0, Y0, rot[i], scl[i])
            iq = iq * colgain[None, :] + Bf
        gain = float(np.clip(gain * np.exp(nrm.normal(0, 0.02)), 0.8, 1.25))
        iq = gain * iq + noise_amp * (nrm.standard_normal((Hf, Wf)) + 1j * nrm.standard_normal((Hf, Wf))).astype(np.complex64)
        img = to_display(iq)
        frames.append(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
        # true landmark positions in this frame, with the same geometry as the rendered image (stored-frame px)
        _, to_frame = fine_affine(rot[i], scl[i], X0, Y0, Wf)
        lmc = np.array([[A['xG'], A['xC'], A['xA']],
                        [A['zs'][int(np.clip(A['xG'], 0, TWf - 1))], A['zt'][int(np.clip(A['xC'], 0, TWf - 1))],
                         A['zt'][int(np.clip(A['xA'], 0, TWf - 1))] - A['pdl']]], float)
        uvf = to_frame(lmc); uvs = fine_to_stored(uvf)
        lm_frame.append(uvs.T)                                    # rows GM, CEJ, ABC; columns u, v
        # GM landmark detector (as in the clinical system): true GM when in view, edge-clipped or a false
        # instance otherwise, 5 % outliers, holds the last value when nothing is detected
        u = uvs[0, 0]; px_mm = PXL
        uy = uvs[1, 0]
        xf = uvs[0, 2]                                            # bone-crest edge (the false instance)
        if lift[i] or nrm.random() < 0.05:
            gmx, gmy = (last_gm if last_gm is not None else (W / 2 * px_mm, 80 * PXA))
            gmx += nrm.normal(0, 0.05); gmy += nrm.normal(0, 0.05); qual = 0.05 if lift[i] else 0.2
        else:
            if 12 < u < W - 12:
                ux = u + nrm.normal(0, 1.5)
            elif nrm.random() < 0.6:
                ux = float(np.clip(u, 6, W - 6)) + nrm.normal(0, 1.0)       # clipped at the frame edge
            else:                                                            # another instance (bone crest edge)
                ux = float(np.clip(xf, 6, W - 6)) + nrm.normal(0, 1.5)
            if nrm.random() < 0.05:
                ux = nrm.uniform(10, W - 10)
            gmx, gmy = ux * px_mm, (uy + nrm.normal(0, 1.5)) * PXA
            last_gm = (gmx, gmy); qual = float(np.clip(nrm.normal(0.4, 0.03), 0.2, 0.5))
        yaw = -150 + 0.002 * t[i] + nrm.normal(0, 0.05)                      # translation produces no yaw
        coords.append(dict(frame_n=i, ts=float(t[i]), gm_x_mm=float(gmx), gm_y_mm=float(gmy), quality_score=qual,
                           imu_yaw=float(np.round(yaw / (180 / 32768)) * (180 / 32768)), imu_valid=bool(nrm.random() > 0.03),
                           imu_gyro_z=0.0))
    # probe-fixed component as it appears in the display (for E_B): probe-fixed field + noise, no tissue
    nb = np.random.default_rng(seed + 99)
    Bdisp = np.median([to_display(Bf + noise_amp * (nb.standard_normal((Hf, Wf)) + 1j * nb.standard_normal((Hf, Wf))))
                       for _ in range(3)], axis=0).astype(np.float32)
    gt = dict(p=p, q=q, t=t, lift=lift, origin=(ox, oy), yaw_gt=np.zeros(t.size), probe_fixed=Bdisp,
              elev=e, rot=rot, scale=scl, lm=dict(GM=A['xG'] / F, CEJ=A['xC'] / F, ABC=A['xA'] / F),
              lm_frame=np.array(lm_frame), tissue=None)
    if return_debug:
        gt['anatomy'] = A
    return frames, coords, gt
