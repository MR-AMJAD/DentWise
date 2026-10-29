"""Benchmark on the physics-based sequences (same protocol as the generic benchmark: ioustitch.benchmark.reconstruct_all).

Two acquisition conditions:
  ideal_gate - frames without tissue contact (lift-off) are removed using the simulator's ground-truth labels
               before reconstruction (an ideal/oracle gate standing in for the live quality gate of the clinical
               system; the clinical gate itself is not simulated); lift-off then appears as a time gap with
               unobserved motion.
  ungated    - all frames are kept; lift-off frames show the gel/air reverberation; the pipeline's own lift-off
               detector is tested here.
Before any reconstruction every cached sequence is validated (gen_cache.validate: simulator fingerprint including
the simulator / trajectory-generator source and PSF, job parameters, GM recalibration version); the run stops if
any sequence is stale. Outputs (in --out, default results/physics/): <tag>_<cond>.csv and _traj.npz per job, written
atomically, then a completion marker <tag>_<cond>.done.json; the marker is deleted before a job recomputes. A job is
skipped only if the marker matches and all outputs exist (--force recomputes). The aggregate phys_bench_all.csv
contains exactly this invocation's jobs; MANIFEST.json records versions, file hashes and every job marker.
Runtime: about 2-4 min per sequence and condition on one core.

    python physim/phys_bench.py [ideal_gate,ungated] [--cache physim/cache] [--out results/physics] [--procs 2] [--force] [--only jerky_s1]
"""
import sys, os, time, pickle, argparse, hashlib, numpy as np, pandas as pd, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # folder that contains ioustitch/
sys.path.insert(0, HERE)
from multiprocessing import Pool
from ioustitch import StitchConfig
from ioustitch.evaluate import trajectory_metrics
from ioustitch.benchmark import PHYSICS_ABLATIONS, reconstruct_all
from ioustitch import runmeta

OUT = None; FORCE = False; VERSION = None; ABL = list(PHYSICS_ABLATIONS)


def job_version(D):
    """Same for every job of a run: code version | simulator fingerprint | scenario settings | GM recalibration.
    (Each sequence's own job parameters are checked by gen_cache.validate before the run.)"""
    import gen_cache, json
    scen = hashlib.sha256(json.dumps(dict(s=gen_cache.SCEN, n=gen_cache.N_FRAMES), sort_keys=True).encode()).hexdigest()[:12]
    return f"{VERSION}|{D.get('cache_version')}|scen:{scen}|{D.get('recal_version')}"


def run(args):
    path, cond = args
    tag = os.path.basename(path)[:-4]
    outs = [f'{OUT}/{tag}_{cond}.csv', f'{OUT}/{tag}_{cond}_traj.npz']; marker = f'{OUT}/{tag}_{cond}.done.json'
    D = pickle.load(open(path, 'rb'))
    version = job_version(D)
    if not FORCE and runmeta.is_done(marker, version, outs):
        print(f'{tag:12s} {cond:10s} up to date', flush=True)
        return outs[0]
    runmeta.begin(marker)                       # invalidate before any output is rewritten
    t0 = time.time()
    print(f'{tag:12s} {cond:10s} started', flush=True)
    gt = D['gt']; lift = gt['lift']
    keep = ~lift if cond == 'ideal_gate' else np.ones(lift.size, bool)
    frames = [cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) for x, k in zip(D['frames'], keep) if k]
    coords = [c for c, k in zip(D['coords'], keep) if k]
    gp, gq, gtt, gl = gt['p'][keep], gt['q'][keep], gt['t'][keep], lift[keep]
    cfg = StitchConfig(); cfg.verbose = False
    px, py = cfg.px_mm_x, cfg.px_mm_y
    full, T = reconstruct_all(frames, coords, cfg, ABL)
    rows = []
    for name, r in T.items():
        m = trajectory_metrics(r['p'], r['q'], gp, gq, gtt, px, py, r['sd_p'])
        m.update(method=name, scenario=D['scenario'], seed=int(D['seed']),
                 cond=cond, runtime_s=r['runtime_s'], n_frames=len(frames), n_lift=int(gl.sum()),
                 travel_mm=float(np.ptp(gp) * px))
        if r['blind'] is not None:
            m.update(blind_tp=int((r['blind'] & gl).sum()), blind_fp=int((r['blind'] & ~gl).sum()))
        rows.append(m)
    runmeta.atomic_csv(pd.DataFrame(rows), outs[0])
    runmeta.atomic_npz(outs[1], t=gtt, gt_p=gp, gt_q=gq, p=full['p'], q=full['q'], sd_p=full['sd_p'], blind=full['blind'], lift=gl,
                       **{f'{k}_p': np.asarray(v['p']) for k, v in T.items()}, **{f'{k}_q': np.asarray(v['q']) for k, v in T.items()})
    runmeta.mark_done(marker, version, outs, runtime_s=time.time() - t0, cache_key=D.get('cache_key'))
    pr = rows[0]; cs = [r for r in rows if r['method'] == 'chain_sep'][0]
    print(f"{tag:12s} {cond:10s} proposed {pr['ate_x_um']:6.0f} um | chain_sep {cs['ate_x_um']:6.0f} | lift {pr['n_lift']} "
          f"blind_tp {pr['blind_tp']} fp {pr['blind_fp']}  [{time.time() - t0:.0f}s]", flush=True)
    return outs[0]


def aggregate(outputs):
    """Concatenate exactly the per-sequence outputs of this invocation and check key uniqueness."""
    df = pd.concat([pd.read_csv(f) for f in sorted(set(outputs))], ignore_index=True)
    key = ['cond', 'scenario', 'seed', 'method']
    dup = df.duplicated(key, keep=False)
    if dup.any():
        raise RuntimeError(f'duplicate benchmark rows: {df[dup][key].values.tolist()[:5]}')
    return df


def init(out, force, version, abl=None):
    global OUT, FORCE, VERSION, ABL
    OUT, FORCE, VERSION = out, force, version
    ABL = list(abl) if abl is not None else list(PHYSICS_ABLATIONS)


def validate_cache(paths):
    import gen_cache
    bad = [(os.path.basename(p), r) for p in paths for ok, r in [gen_cache.validate(p)] if not ok]
    if bad:
        raise SystemExit('stale or unvalidated physics cache - regenerate with gen_cache.py (+ recal_gm.py):\n  ' +
                         '\n  '.join(f'{n}: {r}' for n, r in bad))


if __name__ == '__main__':
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument('conds', nargs='?', default='ideal_gate,ungated')
    ap.add_argument('--cache', default=os.path.join(HERE, 'cache'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(HERE), 'results', 'physics'))
    ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--only', default='', help='comma list of sequence tags, e.g. jerky_s1 (quick check; no aggregate)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    abl = PHYSICS_ABLATIONS
    cfgv = dict(abl=abl)
    version = runmeta.code_version([os.path.abspath(__file__)], cfgv)
    paths = sorted(glob.glob(os.path.join(os.path.abspath(a.cache), '*.pkl')))
    if a.only:
        paths = [p for p in paths if os.path.basename(p)[:-4] in set(a.only.split(','))]
    validate_cache(paths)
    jobs = [(p, c) for c in a.conds.split(',') for p in paths]
    print(f'version {version}: {len(jobs)} jobs, {a.procs} processes, cache validated', flush=True)
    init(a.out, a.force, version, abl)
    if a.only:
        for j in jobs:
            run(j)
        sys.exit(0)
    with Pool(a.procs, initializer=init, initargs=(a.out, a.force, version, abl)) as P:
        outputs = list(P.imap_unordered(run, jobs))
    markers = [f"{a.out}/{os.path.basename(p)[:-4]}_{c}.done.json" for p, c in jobs]
    jver = job_version(pickle.load(open(paths[0], 'rb')))          # same for every job (validated cache)
    runmeta.check_markers(markers, jver)                           # before publishing the aggregate
    df = aggregate(outputs)
    runmeta.atomic_csv(df, f'{a.out}/phys_bench_all.csv')
    runmeta.write_manifest(f'{a.out}/MANIFEST.json', jver, markers,
                           extra_files=[os.path.abspath(__file__)], config=dict(**cfgv, conds=a.conds),
                           driver='physim/phys_bench.py', code_version=version,
                           cache=os.path.basename(os.path.abspath(a.cache)))
    print(df.groupby(['cond', 'method']).ate_x_um.median().unstack(0).round(0))
