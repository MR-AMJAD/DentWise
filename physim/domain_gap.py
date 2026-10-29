"""Image statistics of clinical, generic-simulator and physics-simulator frames (domain gap).

Per sweep:  display statistics (fraction of pixels > 8, median grey of those, saturated fraction),
speckle size (FWHM of the normalised autocovariance in tissue patches, lateral and axial, um),
lateral-texture ratio rho, probe-fixed layer strength E_B (trajectory-free B0 estimate, as in the paper),
and appearance decorrelation: NCC between frames i and i+k (k = 1..12) after compensating the displacement
(clinical: reconstructed trajectory; simulation: ground truth), binned by lateral displacement (Fig. S1(d)).

    python physim/domain_gap.py generic            -> results/domain_gap/gap_generic.csv
    python physim/domain_gap.py physics            -> results/domain_gap/gap_physics.csv  (needs the physics cache)
    python physim/domain_gap.py clinical           -> results/clinical/gap_clinical.csv   (needs the clinical sweeps:
        IOUS_CLINICAL_ROOT = folder with the sweep folders, IOUS_STITCHED = output folder of the reconstruction,
        optional IOUS_TAGS = text file with one sweep tag per line to restrict the analysis to a subset)
"""
import sys, os, glob, pickle, numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ioustitch import StitchConfig
from ioustitch.pipeline import initial_probe_fixed, condition
from ioustitch import preprocess as pp
from ioustitch.run_sweep import find_sweeps, sweep_tag, load_sweep
cfg = StitchConfig()
PXL, PXA = 7.0 / 256, 7.344 / 256
BINS = [0, 0.25, 0.5, 1.0, 2.0, 3.0]


def acf_fwhm(imgs, ws):
    acc = None; n = 0
    for im, w in zip(imgs[::3], ws[::3]):
        for y in range(0, 256 - 24, 12):
            for x in range(0, 256 - 24, 12):
                P = im[y:y + 24, x:x + 24]; W_ = w[y:y + 24, x:x + 24]
                if W_.mean() < 0.5 or P.max() >= 0.98 or P.mean() < 0.04:
                    continue
                P = P - P.mean()
                f = np.fft.fft2(P, s=(48, 48)); a = np.fft.ifft2(f * np.conj(f)).real
                a = np.fft.fftshift(a) / max(a.max(), 1e-12)
                acc = a if acc is None else acc + a; n += 1
    if n == 0:
        return np.nan, np.nan, 0
    a = acc / n; c = 24
    def fw(prof, px):
        h = prof[c:]; k = np.argmax(h < 0.5)
        if k == 0: return np.nan
        x0 = (k - 1) + (h[k - 1] - 0.5) / (h[k - 1] - h[k])
        return 2 * x0 * px * 1e3
    return fw(a[c, :], PXL), fw(a[:, c], PXA), n


def sweep_stats(frames_gray_u8, p, q):
    fr = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x in frames_gray_u8]
    g, fs, ws = condition(fr, cfg)
    B = initial_probe_fixed(g); spw = pp.static_pattern_weight(g, B)
    eb = float((B ** 2).sum() / np.mean([x ** 2 for x in g], 0).sum())
    f2 = [pp.smooth(np.maximum(x - B, 0), cfg.smooth_sigma) for x in g]; w2 = [w * spw for w in ws]
    rho = []
    for s, w in zip(f2, w2):
        gx = np.abs(np.diff(s, axis=1))[:-1, :]; gz = np.abs(np.diff(s, axis=0))[:, :-1]; ww = w[:-1, :-1]
        rho.append(float((ww * gx).sum() / max((ww * gz).sum(), 1e-9)))
    u8 = np.stack(frames_gray_u8)
    nz = (u8 > 8).mean((1, 2)); sat = (u8 >= 250).mean((1, 2))
    med = np.array([np.median(x[x > 8]) if (x > 8).any() else 0 for x in u8])
    fl, fa, npatch = acf_fwhm(g, ws)
    # decorrelation vs displacement
    dec = []
    N = len(g)
    for i in range(N):
        for k in range(1, 13):
            j = i + k
            if j >= N: break
            dx = int(round(p[j] - p[i])); dy = int(round(q[j] - q[i]))
            if abs(dx) > 200 or abs(dy) > 40: continue
            # frame j pixel u shows canvas X = u + p_j ; overlap: frame i u_i = u_j + dx
            a = f2[i][max(dy, 0):256 + min(dy, 0), max(dx, 0):256 + min(dx, 0)]
            b = f2[j][max(-dy, 0):256 + min(-dy, 0), max(-dx, 0):256 + min(-dx, 0)]
            wa = w2[i][max(dy, 0):256 + min(dy, 0), max(dx, 0):256 + min(dx, 0)]
            wb = w2[j][max(-dy, 0):256 + min(-dy, 0), max(-dx, 0):256 + min(-dx, 0)]
            best = -1
            for ex in (-2, -1, 0, 1, 2):                    # small search around the predicted shift (trajectory error)
                for ey in (-1, 0, 1):
                    dx2, dy2 = dx + ex, dy + ey
                    a = f2[i][max(dy2, 0):256 + min(dy2, 0), max(dx2, 0):256 + min(dx2, 0)]
                    b = f2[j][max(-dy2, 0):256 + min(-dy2, 0), max(-dx2, 0):256 + min(-dx2, 0)]
                    wa = w2[i][max(dy2, 0):256 + min(dy2, 0), max(dx2, 0):256 + min(dx2, 0)]
                    wb = w2[j][max(-dy2, 0):256 + min(-dy2, 0), max(-dx2, 0):256 + min(-dx2, 0)]
                    w = wa * wb
                    if w.sum() < 500: continue
                    ma = (w * a).sum() / w.sum(); mb = (w * b).sum() / w.sum()
                    cov = (w * (a - ma) * (b - mb)).sum(); va = (w * (a - ma) ** 2).sum(); vb = (w * (b - mb) ** 2).sum()
                    best = max(best, cov / np.sqrt(max(va * vb, 1e-12)))
            if best > -1:
                dec.append((abs(p[j] - p[i]) * PXL, best))
    dec = np.array(dec) if dec else np.zeros((0, 2))
    out = dict(n=N, frac_nz=float(np.median(nz)), med_grey=float(np.median(med)), sat=float(np.median(sat)),
               speckle_lat_um=fl, speckle_ax_um=fa, n_patch=npatch, rho=float(np.median(rho)), EB=100 * eb)
    for a_, b_ in zip(BINS[:-1], BINS[1:]):
        m = (dec[:, 0] >= a_) & (dec[:, 0] < b_)
        out[f'ncc_{a_}-{b_}'] = float(np.median(dec[m, 1])) if m.sum() >= 5 else np.nan
    return out


if __name__ == '__main__':
    which = sys.argv[1]; rows = []
    if which == 'clinical':
        TR = os.environ['IOUS_STITCHED']
        TAGS = set(open(os.environ['IOUS_TAGS']).read().split()) if os.environ.get('IOUS_TAGS') else None
        for d, files, cp in find_sweeps(os.environ['IOUS_CLINICAL_ROOT']):
            tag = sweep_tag(d); tp = os.path.join(TR, tag, 'trajectory.csv')
            if TAGS is not None and tag not in TAGS: continue
            if not os.path.isfile(tp): continue
            fr, co = load_sweep(files, cp); tr = pd.read_csv(tp); n = min(len(fr), len(tr))
            u8 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in fr[:n]]
            r = sweep_stats(u8, tr.p_px.values[:n], tr.q_px.values[:n]); r['tag'] = tag; rows.append(r)
            print(tag, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    elif which == 'generic':
        from ioustitch.simulate import simulate_sweep
        from ioustitch.benchmark import SCEN
        for sc in SCEN:
            for seed in range(5):
                fr, co, gt = simulate_sweep(seed=seed, n_frames=80, **SCEN[sc])
                keep = ~gt['lift']
                u8 = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x, k in zip(fr, keep) if k]
                r = sweep_stats(u8, gt['p'][keep], gt['q'][keep]); r['tag'] = f'{sc}_s{seed}'; rows.append(r)
                print(r['tag'], {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    elif which == 'physics':
        for f in sorted(glob.glob(os.path.join(HERE, 'cache', '*.pkl'))):
            D = pickle.load(open(f, 'rb')); gt = D['gt']; keep = ~gt['lift']
            u8 = [x for x, k in zip(D['frames'], keep) if k]
            r = sweep_stats(u8, gt['p'][keep], gt['q'][keep]); r['tag'] = os.path.basename(f)[:-4]; rows.append(r)
            print(r['tag'], {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    od = os.path.join(os.path.dirname(HERE), 'results', 'clinical' if which == 'clinical' else 'domain_gap')
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(od, f'gap_{which}.csv'), index=False)
