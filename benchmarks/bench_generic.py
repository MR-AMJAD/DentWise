"""Generic-simulation benchmark (Table S1, S3, S4, Section S5, Fig. 4(a), (c)).

Protocol: ioustitch.benchmark.reconstruct_all (end-to-end ablations, controlled chaining, baselines), 25 sequences
(5 scenarios x 5 seeds, 100 frames). Lift-off detection is evaluated on the pipeline's own labels.
Each job writes <tag>.csv (one row per method), <tag>_val.json (photometric consistency, lift-off detection, layer
estimate), <tag>_frames.csv (per-frame error and reported sigma) and <tag>_traj.npz, all atomically, and then a
completion marker <tag>.done.json holding the code/configuration version. A job is skipped only if its marker
matches the current version and all outputs exist; --force recomputes everything.

    python benchmarks/bench_generic.py [--out results/generic] [--procs 2] [--force] [--only nominal_s0]
Runtime: about 4-5 min per sequence on one core; --only runs single sequences (no aggregate) as a quick check
against the shipped per-sequence outputs. MANIFEST.json records the version, the hash of every file that enters it,
the software versions and all 25 job markers (the aggregate is refused if any marker is missing or from another
version).
"""
import sys, os, json, time, argparse, numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from multiprocessing import Pool
from ioustitch import StitchConfig
from ioustitch.simulate import simulate_sweep
from ioustitch.benchmark import SCEN, GENERIC_ABLATIONS, reconstruct_all
from ioustitch.pipeline import condition, initial_probe_fixed, detect_blind
from ioustitch.composite import composite
from ioustitch.evaluate import trajectory_metrics, panorama_fidelity
from ioustitch import preprocess as pp, runmeta

N_FRAMES = 100
OUT = None; FORCE = False; VERSION = None; ABL = list(GENERIC_ABLATIONS)


def pc_lags(frames_s, weights, p, q, lags):
    H, W = frames_s[0].shape; vals = []
    for i in range(len(frames_s)):
        for L in lags:
            j = i + L
            if j >= len(frames_s): continue
            dx, dy = p[j] - p[i], q[j] - q[i]
            if abs(dx) > W * 0.75 or abs(dy) > H * 0.5: continue
            M = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
            fi = cv2.warpAffine(frames_s[i], M, (W, H), flags=cv2.INTER_LINEAR)
            wi = cv2.warpAffine(weights[i], M, (W, H), flags=cv2.INTER_LINEAR)
            w = wi * weights[j]
            if w.sum() < 500: continue
            a = fi; b = frames_s[j]
            ma = (a * w).sum() / w.sum(); mb = (b * w).sum() / w.sum()
            vals.append((w * (a - ma) * (b - mb)).sum() / (np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()) + 1e-9))
    return float(np.mean(vals)) if vals else np.nan


def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    return float(np.mean([(p_ < neg).mean() + 0.5 * (p_ == neg).mean() for p_ in pos]))


def outputs(tag):
    return [f'{OUT}/{tag}.csv', f'{OUT}/{tag}_val.json', f'{OUT}/{tag}_frames.csv', f'{OUT}/{tag}_traj.npz']


def job(a):
    scen, seed = a
    tag = f'{scen}_s{seed}'; outs = outputs(tag); marker = f'{OUT}/{tag}.done.json'
    if not FORCE and runmeta.is_done(marker, VERSION, outs):
        print(f'{tag:12s} up to date (marker {VERSION})', flush=True)
        return tag
    runmeta.begin(marker)                       # invalidate before any output is rewritten
    T0 = time.time()
    print(f'{tag:12s} started', flush=True)
    frames, coords, gt = simulate_sweep(seed=seed, n_frames=N_FRAMES, **SCEN[scen])
    cfg = StitchConfig(); cfg.verbose = False
    px, py = cfg.px_mm_x, cfg.px_mm_y
    g, fs, ws = condition(frames, cfg)
    quality = np.array([c['quality_score'] for c in coords])
    comp_gt = composite(g, gt['p'], gt['q'], quality, None, ws, cfg)
    full, T = reconstruct_all(frames, coords, cfg, ABL)
    L_ = gt['lift']; bl_ = full['blind']
    rows = []
    for name, r in T.items():
        m = trajectory_metrics(r['p'], r['q'], gt['p'], gt['q'], gt['t'], px, py, r['sd_p'])
        pa = np.asarray(r['p']) - np.mean(r['p']) + np.mean(gt['p']); qa = np.asarray(r['q']) - np.mean(r['q']) + np.mean(gt['q'])
        m['pfid'] = panorama_fidelity(composite(g, pa, qa, quality, None, ws, cfg), comp_gt)
        m.update(method=name, scenario=scen, seed=seed, runtime_s=r['runtime_s'], n_lift=int(L_.sum()))
        if r['blind'] is not None:
            m.update(blind_tp=int((r['blind'] & L_).sum()), blind_fp=int((r['blind'] & ~L_).sum()))
        if name == 'proposed':
            m.update(n_pairs=len(full['pairs']), n_pairs_ok=int(sum(x.ok for x in full['pairs'])),
                     n_bridge_ok=int(sum(x.ok and x.role == 'bridge' for x in full['pairs'])),
                     **{k + '_sd': float(np.sqrt(v)) for k, v in full['scales'].items()
                        if not k.startswith(('gauge', 'motion'))})
        rows.append(m)
    # photometric consistency (frames separated with the proposed layer estimate, as in vivo), scheduled lags 1-10
    # and auxiliary lags {5, 7, 9, 10} that are not in the scheduled lag set (they can still occur as gap bridges)
    fs_s = [cv2.GaussianBlur(x, (0, 0), cfg.smooth_sigma) for x in full['frames_clean']]
    val = dict(scenario=scen, seed=seed, n_lift=int(L_.sum()), version=VERSION)
    TT = dict(gt=(gt['p'], gt['q']), proposed=(T['proposed']['p'], T['proposed']['q']),
              chain_raw=(T['seq_reg']['p'], T['seq_reg']['q']), chain_sep=(T['chain_sep']['p'], T['chain_sep']['q']),
              gm_sg=(T['gm_sg']['p'], T['gm_sg']['q']), cf=(T['cf_imu']['p'], T['cf_imu']['q']),
              zero=(np.zeros(len(frames)), np.zeros(len(frames))))
    for k, (p, q) in TT.items():
        val[f'pc_{k}'] = pc_lags(fs_s, full['tissue'], np.asarray(p, float), np.asarray(q, float), range(1, 11))
        val[f'pch_{k}'] = pc_lags(fs_s, full['tissue'], np.asarray(p, float), np.asarray(q, float), (5, 7, 9, 10))
    # detector input exactly as in stitch_sweep (iteration 0); labels must equal the pipeline's
    B0 = initial_probe_fixed(g); spw = pp.static_pattern_weight(g, B0)
    assert np.array_equal(detect_blind([w * spw for w in ws]), bl_), 'detector input differs from the pipeline'
    mass = np.array([float((w * spw).sum()) for w in ws]); mass /= np.median(mass)
    val.update(blind_tp=int((bl_ & L_).sum()), blind_fp=int((bl_ & ~L_).sum()), blind_fn=int((~bl_ & L_).sum()))
    if L_.any():
        val.update(auc=auc(mass[L_], mass[~L_]), max_mass_lift=float(mass[L_].max()), min_mass_sighted=float(mass[~L_].min()))
    Bh = full['probe_fixed_layer']; Bt = gt['probe_fixed']
    ref = np.mean([x ** 2 for x in g], 0).sum()
    val.update(B_corr=float(np.corrcoef(Bh.ravel(), Bt.ravel())[0, 1]), B0_corr=float(np.corrcoef(B0.ravel(), Bt.ravel())[0, 1]),
               B_leak=float((np.maximum(Bh - Bt, 0) ** 2).sum() / max((Bh ** 2).sum(), 1e-9)),
               EB_true=float((Bt ** 2).sum() / ref), EB_est=float((Bh ** 2).sum() / ref), EB0=float((B0 ** 2).sum() / ref))
    ex = (full['p'] - full['p'].mean()) - (gt['p'] - gt['p'].mean())
    runmeta.atomic_csv(pd.DataFrame(rows), outs[0])
    runmeta.atomic_json(val, outs[1])
    runmeta.atomic_csv(pd.DataFrame(dict(scenario=scen, seed=seed, frame=np.arange(len(frames)), err_px=ex, sd_px=full['sd_p'],
                                         blind=bl_, lift=L_)), outs[2])
    runmeta.atomic_npz(outs[3], t=gt['t'], gt_p=gt['p'], gt_q=gt['q'], lift=L_, blind=bl_, sd_p=full['sd_p'],
                       **{f'{k}_p': np.asarray(v['p']) for k, v in T.items()}, **{f'{k}_q': np.asarray(v['q']) for k, v in T.items()})
    runmeta.mark_done(marker, VERSION, outs, runtime_s=time.time() - T0)
    pr = rows[0]; cs = [x for x in rows if x['method'] == 'chain_sep'][0]
    print(f"{tag:12s} proposed {pr['ate_x_um']:6.0f} | chain_sep {cs['ate_x_um']:6.0f} | blind {val['blind_tp']}/{val['n_lift']}"
          f" fp {val['blind_fp']}  [{time.time() - T0:.0f}s]", flush=True)
    return tag


def init(out, force, version, abl=None):
    global OUT, FORCE, VERSION, ABL
    OUT, FORCE, VERSION = out, force, version
    ABL = list(abl) if abl is not None else list(GENERIC_ABLATIONS)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'generic'))
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default='', help='comma list of tags, e.g. nominal_s0 (quick check; no aggregate)')
    ap.add_argument('--seeds', default='0-4', help="seed range (the paper uses 0-4)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = (a.seeds.split('-') + [None])[:2]
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(x) for x in a.seeds.split(',')]
    abl = GENERIC_ABLATIONS
    cfgv = dict(n_frames=N_FRAMES, scen=SCEN, abl=abl)
    version = runmeta.code_version([os.path.abspath(__file__)], cfgv)
    init(a.out, a.force, version, abl)
    jobs = [(sc, s) for sc in SCEN for s in seeds]
    if a.only:
        sel = set(a.only.split(','))
        print(f'version {version}', flush=True)
        for j in [j for j in jobs if f'{j[0]}_s{j[1]}' in sel]:
            job(j)
        sys.exit(0)
    print(f'version {version}: {len(jobs)} sequences, {a.procs} processes', flush=True)
    with Pool(a.procs, initializer=init, initargs=(a.out, a.force, version, abl)) as P:
        done = list(P.imap_unordered(job, jobs))
    runmeta.check_markers([f'{a.out}/{sc}_s{s}.done.json' for sc, s in jobs], version)    # before publishing
    df = pd.concat([pd.read_csv(f'{a.out}/{sc}_s{s}.csv') for sc, s in jobs], ignore_index=True)
    assert not df.duplicated(['scenario', 'seed', 'method']).any()
    runmeta.atomic_csv(df, f'{a.out}/bench_all.csv')
    runmeta.atomic_csv(pd.DataFrame([json.load(open(f'{a.out}/{sc}_s{s}_val.json')) for sc, s in jobs]), f'{a.out}/val_all.csv')
    runmeta.atomic_csv(pd.concat([pd.read_csv(f'{a.out}/{sc}_s{s}_frames.csv') for sc, s in jobs]), f'{a.out}/frames_all.csv')
    runmeta.write_manifest(f'{a.out}/MANIFEST.json', version, [f'{a.out}/{sc}_s{s}.done.json' for sc, s in jobs],
                           extra_files=[os.path.abspath(__file__)], config=cfgv,
                           driver='benchmarks/bench_generic.py')
    print(df.groupby('method').ate_x_um.median().round(0).sort_values())
