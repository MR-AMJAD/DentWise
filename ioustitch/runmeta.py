"""Run manifests for the benchmark drivers.

* A job is skipped only if its completion marker exists, records the same code/configuration version, and all its
  outputs exist.
* `begin(marker)` deletes the marker BEFORE a job (re)computes, so a forced or version-triggered rerun that is
  interrupted part-way can never leave an old marker next to partially rewritten outputs.
* Outputs are written atomically (temporary file + rename) and the marker is written last.
* `check_markers` is called before an aggregate is published, and `write_manifest` (written after the aggregate)
  records the version, the hash of every file that enters it, the configuration, the software versions and every
  job's marker; both refuse if any job marker is missing or has another version, so an aggregate is never published
  from outputs of different code versions."""
import hashlib, json, os, platform, sys, time

PKG = os.path.dirname(os.path.abspath(__file__))
CORE = ['config.py', 'preprocess.py', 'registration.py', 'graph.py', 'pipeline.py', 'composite.py', 'evaluate.py',
        'baselines.py', 'simulate.py', 'benchmark.py']


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _files(extra_files):
    return [os.path.join(PKG, c) for c in CORE if os.path.isfile(os.path.join(PKG, c))] + list(extra_files)


def file_hashes(extra_files=()):
    """{relative name: sha256[:16]} of every file that enters the version (package files as ioustitch/<name>)."""
    out = {}
    for f in _files(extra_files):
        key = ('ioustitch/' + os.path.basename(f)) if os.path.dirname(os.path.abspath(f)) == PKG else os.path.basename(f)
        out[key] = file_hash(f)[:16]
    return out


def code_version(extra_files=(), extra=None):
    """Hash of the reconstruction package, any extra files (simulator, PSF, driver) and a JSON-able config."""
    h = hashlib.sha256()
    for f in _files(extra_files):
        h.update(os.path.basename(f).encode()); h.update(file_hash(f).encode())
    h.update(json.dumps(extra or {}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def is_done(marker, version, outputs):
    if not os.path.isfile(marker) or not all(os.path.isfile(o) for o in outputs):
        return False
    try:
        return json.load(open(marker)).get('version') == version
    except (OSError, ValueError):
        return False


def begin(marker):
    """Invalidate a job before (re)computing it: remove its completion marker first."""
    for f in (marker, marker + '.tmp'):
        if os.path.exists(f):
            os.remove(f)


def mark_done(marker, version, outputs, **info):
    tmp = marker + '.tmp'
    json.dump(dict(version=version, outputs=[os.path.basename(o) for o in outputs], finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                   **info), open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, marker)


def software():
    import numpy, scipy
    try:
        import cv2; cvv = cv2.__version__
    except Exception:
        cvv = None
    return dict(python=sys.version.split()[0], numpy=numpy.__version__, scipy=scipy.__version__, opencv=cvv,
                platform=platform.platform())


def check_markers(markers, version):
    """Every job marker must exist and carry `version`; call BEFORE publishing an aggregate."""
    jobs = {}
    for m in markers:
        if not os.path.isfile(m):
            raise RuntimeError(f'missing job marker {m}: aggregate refused')
        d = json.load(open(m))
        if d.get('version') != version:
            raise RuntimeError(f'job marker {m} has version {d.get("version")}, expected {version}: aggregate refused')
        jobs[os.path.basename(m)] = dict(finished=d.get('finished'), runtime_s=d.get('runtime_s'))
    return jobs


def write_manifest(path, version, markers, extra_files=(), config=None, **info):
    """Write MANIFEST.json for an aggregate; every job marker must exist and carry `version`."""
    jobs = check_markers(markers, version)
    atomic_json(dict(version=version, written=time.strftime('%Y-%m-%d %H:%M:%S'), n_jobs=len(jobs),
                     files=file_hashes(extra_files), config=config or {}, software=software(), jobs=jobs, **info), path)


def atomic_csv(df, path):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False); os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + '.tmp'
    json.dump(obj, open(tmp, 'w'), indent=1, default=str); os.replace(tmp, path)


def atomic_npz(path, **arrays):
    import numpy as np
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays); os.replace(tmp, path)
