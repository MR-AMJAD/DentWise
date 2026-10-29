"""Content fingerprints of a physics cache folder, so a regenerated cache can be compared with the one used for the
published results without shipping the 130-MB pickles.
    python cache_manifest.py [cache] [--check MANIFEST.json]
Per sequence: sha256 of the frames, of the ground truth (p, q, t, lift) and of the (recalibrated) per-frame coordinates (GM, orientation),
plus the stored simulator fingerprint, job key and recalibration version."""
import sys, os, glob, json, pickle, hashlib, argparse, numpy as np


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def fingerprint(path):
    D = pickle.load(open(path, 'rb'))
    gt = D['gt']
    coords = json.dumps([{k: (round(float(v), 9) if isinstance(v, (float, np.floating)) else v) for k, v in sorted(c.items())}
                         for c in D['coords']], sort_keys=True, default=str).encode()
    return dict(frames=sha(np.ascontiguousarray(D['frames']).tobytes()),
                gt=sha(b''.join(np.ascontiguousarray(np.asarray(gt[k], float)).tobytes() for k in ('p', 'q', 't', 'lift'))),
                coords=sha(coords), cache_version=D.get('cache_version'), cache_key=D.get('cache_key'),
                recal_version=D.get('recal_version'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cache', nargs='?', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache'))
    ap.add_argument('--check', default='')
    a = ap.parse_args()
    M = {os.path.basename(p)[:-4]: fingerprint(p) for p in sorted(glob.glob(os.path.join(a.cache, '*.pkl')))}
    if a.check:
        R = json.load(open(a.check))
        diff = {k: [f for f in ('frames', 'gt', 'coords') if M.get(k, {}).get(f) != R[k][f]] for k in R}
        diff = {k: v for k, v in diff.items() if v}
        print('identical to', a.check, 'for', len(R) - len(diff), 'of', len(R), 'sequences', '' if not diff else diff)
    else:
        out = os.path.join(a.cache, 'MANIFEST.json')
        json.dump(M, open(out, 'w'), indent=1); print('wrote', out, len(M), 'sequences')
