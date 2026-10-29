"""GM 'landmark detector' recalibrated to the clinical live signal: frame-to-frame jumps > 1 mm are rare in the
clinical coordinates (median 0 %, IQR 0-0.6 % of frames); when the GM leaves the field of view the detector
reports it at the frame edge (as in the clinical per-frame detections, x = 250 px); 5.5 % of frames hold the
previous value (missed detection, as in the clinical coordinates); 0.5 % outliers; lift-off holds the last value.
The random stream of each sequence is seeded from its stored scenario/seed identifier (not from the file path),
so the result does not depend on the operating system's path separator. Idempotent (starts from coords_v0)."""
import os, glob, pickle, zlib, numpy as np
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
PXL, PXA = 7.0 / 256, 7.344 / 256
W = 256
RECAL_VERSION = 'gm-recal-2'


def stable_tag(D, f):
    if 'scenario' in D and 'seed' in D:
        return f"{D['scenario']}_s{int(D['seed'])}.pkl", int(D['seed'])
    name = Path(f).name                                   # native path handling (Windows and POSIX)
    return name, int(name.rsplit('_s', 1)[1][:-4])


def recal(f):
    D = pickle.load(open(f, 'rb'))
    if 'coords_v0' not in D:
        D['coords_v0'] = [dict(c) for c in D['coords']]
    gt = D['gt']; tag, seed = stable_tag(D, f)
    rng = np.random.default_rng(10_000 + seed + zlib.crc32(tag.encode()) % 1000)
    last = None; new = []
    for i, c0 in enumerate(D['coords_v0']):
        c = dict(c0)
        u = float(gt['lm_frame'][i][0, 0])        # true GM column in this frame (same geometry as the rendering)
        if gt['lift'][i] or rng.random() < 0.055:
            gx, gy = last if last is not None else (W / 2 * PXL, c0['gm_y_mm']); q = 0.05 if gt['lift'][i] else 0.2
        else:
            ux = (u if 12 < u < W - 12 else float(np.clip(u, 6, W - 6))) + rng.normal(0, 1.5 if 12 < u < W - 12 else 1.0)
            if rng.random() < 0.005:
                ux = rng.uniform(10, W - 10)
            gx, gy = ux * PXL, c0['gm_y_mm']; q = float(np.clip(rng.normal(0.4, 0.03), 0.2, 0.5)); last = (gx, gy)
        c.update(gm_x_mm=float(gx), gm_y_mm=float(gy), quality_score=q)
        new.append(c)
    D['coords'] = new; D['recal_version'] = RECAL_VERSION; D['recal_tag'] = tag
    tmp = f + '.tmp'; pickle.dump(D, open(tmp, 'wb')); os.replace(tmp, f)
    gx = np.array([c['gm_x_mm'] for c in new])
    return tag, float((np.abs(np.diff(gx)) > 1).mean())


if __name__ == '__main__':
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'cache')      # python recal_gm.py [cache_dir]
    for f in sorted(glob.glob(os.path.join(cache, '*.pkl'))):
        tag, j = recal(f); print(tag, 'jumps>1mm %.3f' % j)
