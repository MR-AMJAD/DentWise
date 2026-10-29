"""Generate and cache the physics-based benchmark sequences (5 scenarios x seeds, 80 frames).

Every pickle records two fingerprints:
  cache_version  simulator model version + PSF file hash + source hashes of simulate_physics.py and of the shared
                 trajectory generator ioustitch/simulate.py (simulate_physics.cache_version())
  cache_key      the job parameters: scenario name and settings, number of frames, seed (cache_key())
A cached sequence is reused only if both match; --force regenerates. validate() is called by phys_bench.py before
any reconstruction, together with the GM-recalibration version (recal_gm.RECAL_VERSION).

    python gen_cache.py [--seeds 0-4] [--cache cache] [--procs 2] [--force]
The paper uses seeds 0-4 (25 sequences).
"""
import sys, os, json, hashlib, numpy as np, pickle, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simulate_physics as sp
from multiprocessing import Pool
SCEN = dict(nominal=dict(scenario='nominal'), jerky=dict(scenario='jerky', liftoff_rate=0.06),
            reversal=dict(scenario='reversal'), mixed=dict(scenario='mixed', liftoff_rate=0.04),
            hard=dict(scenario='mixed', artifact_strength=3.0, elev_rate=0.08, liftoff_rate=0.06, max_rot_deg=1.5))
N_FRAMES = 80
CACHE = os.path.join(HERE, 'cache')
FORCE = False


def cache_key(sc, seed):
    return hashlib.sha256(json.dumps(dict(scenario=sc, settings=SCEN[sc], n_frames=N_FRAMES, seed=int(seed)),
                                     sort_keys=True).encode()).hexdigest()[:16]


def validate(path, need_recal=True):
    """(ok, reason) for one cached sequence against the current simulator, job parameters and GM recalibration."""
    try:
        D = pickle.load(open(path, 'rb'))
    except Exception as e:
        return False, f'unreadable ({e})'
    if D.get('cache_version') != sp.cache_version():
        return False, f"simulator fingerprint {D.get('cache_version')} != {sp.cache_version()}"
    sc, seed = D.get('scenario'), D.get('seed')
    if sc not in SCEN or D.get('cache_key') != cache_key(sc, seed):
        return False, 'job parameters (scenario settings / frames / seed) differ from gen_cache.py'
    if need_recal:
        import recal_gm
        if D.get('recal_version') != recal_gm.RECAL_VERSION:
            return False, f"GM recalibration {D.get('recal_version')} != {recal_gm.RECAL_VERSION} (run recal_gm.py)"
    return True, ''


def job(a):
    sc, seed = a
    out = os.path.join(CACHE, f'{sc}_s{seed}.pkl')
    if not FORCE and os.path.exists(out) and validate(out, need_recal=False)[0]:
        return out
    fr, co, gt = sp.simulate_sweep_phys(seed=seed, n_frames=N_FRAMES, **SCEN[sc])
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(dict(frames=np.stack([x[..., 0] for x in fr]), coords=co, gt=gt, scenario=sc, seed=seed,
                         cache_version=sp.cache_version(), cache_key=cache_key(sc, seed)), f)
    os.replace(tmp, out)
    return out


def init(force, cache):
    global FORCE, CACHE
    FORCE, CACHE = force, cache


def seeds_arg(s):
    a, b = (s.split('-') + [None])[:2]
    return list(range(int(a), int(b) + 1)) if b else [int(x) for x in s.split(',')]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--force', action='store_true'); ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--seeds', default='0-4'); ap.add_argument('--cache', default=CACHE)
    a = ap.parse_args()
    cache = os.path.abspath(a.cache); os.makedirs(cache, exist_ok=True)
    jobs = [(sc, s) for sc in SCEN for s in seeds_arg(a.seeds)]
    with Pool(a.procs, initializer=init, initargs=(a.force, cache)) as P:
        for o in P.imap_unordered(job, jobs):
            print(o, flush=True)
