"""Core correctness and regression tests:  python -m ioustitch.tests.test_core  (or: python -m pytest ioustitch)"""
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift
from ..registration import FrameSpectra, pad_shape_for, register_pair
from ..graph import Graph, solve
from ..config import StitchConfig


def test_registration_convention():
    rng = np.random.default_rng(0)
    T = np.abs(gaussian_filter(rng.standard_normal((400, 900)), 1.5)); H = W = 256
    crop = lambda p, q: ndshift(T, (-q, -p), order=3)[:H, :W]      # frame(u) = T(u + p)
    pad = pad_shape_for((H, W)); w = np.ones((H, W))
    fi = crop(40, 30)
    for dx, dy in [(10, 3), (-7.3, 2.6), (120.4, -15.2)]:
        r = register_pair(0, 1, FrameSpectra(fi, w, pad), FrameSpectra(crop(40 + dx, 30 + dy), w, pad),
                          pad, 200, 40, 6.0)
        assert r.ok and np.allclose(r.d, [dx, dy], atol=0.1), (dx, dy, r.d)


def test_graph_recovers_chain_with_outlier():
    N = 30; true = np.cumsum(np.full(N, 3.0)); true -= true[0]
    G = Graph(N); G.add_gauge(0); G.add_motion_prior(np.arange(N) * 0.1, 4000)
    for L in (1, 2, 3):
        for i in range(N - L):
            d = np.array([true[i + L] - true[i], 0.0])
            if (i, L) == (10, 1):
                d[0] += 40                                            # gross outlier
            G.add_relative_2d(i, i + L, d, np.eye(2) * 0.25, "image_L1")
    cfg = StitchConfig(); cfg.verbose = False
    res = solve(G, cfg)
    assert np.max(np.abs(res["p"] - true)) < 0.5, np.max(np.abs(res["p"] - true))


def _physim_dir():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(here, "..", "..", "physim"),):
        if os.path.isfile(os.path.join(c, "simulate_physics.py")):
            return os.path.abspath(c)
    raise FileNotFoundError("physics simulator not found")


def test_final_solve_keeps_gap_bridges():
    """Regression: every registered pair, including gap bridges and sighted-lag
    pairs across lift-off, must reach the final solve."""
    from ..simulate import simulate_sweep
    from ..pipeline import stitch_sweep, select_pairs
    fr, co, gt = simulate_sweep(seed=0, n_frames=40, scenario="mixed", liftoff_rate=0.12)
    cfg = StitchConfig(); cfg.verbose = False
    r = stitch_sweep(fr, co, cfg)
    kept = select_pairs(r["pairs"], cfg.lags)
    assert len(kept) == len(r["pairs"]), (len(kept), len(r["pairs"]))
    raw_not_in_lagset = [x for x in r["pairs"] if x.ok and (x.j - x.i) not in set(cfg.lags)]
    assert raw_not_in_lagset, "test sequence should contain pairs whose raw index difference is not a lag"
    assert all(x.role for x in r["pairs"]), "every pair must carry its provenance"
    # a lag-1 ablation keeps short lag-1 pairs and bridges, nothing else
    one = select_pairs(r["pairs"], (1,))
    assert all(x.role == "bridge" or x.lag == 1 for x in one)


def _peak(img):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    w = np.maximum(img - 0.5 * img.max(), 0)
    return float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())


def test_generic_landmark_geometry_matches_rendering():
    """Regression: landmark coordinates use the same affine transform as the rendered frame."""
    import cv2
    from ..simulate import frame_affine
    H = W = 256; X, Y = 400.0, 180.0
    yy, xx = np.mgrid[:600, :900]
    canvas = np.exp(-((xx - X) ** 2 + (yy - Y) ** 2) / (2 * 1.5 ** 2)).astype(np.float32)
    for rot, sc in [(0.0, 1.0), (1.5, 1.03), (-1.2, 0.97)]:
        M, to_frame = frame_affine(rot, sc, 250.3, 70.6, W, H)
        img = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        u, v = to_frame(np.array([[X], [Y]]))[:, 0]
        uu, vv = _peak(img)
        assert abs(uu - u) < 0.1 and abs(vv - v) < 0.1, (rot, sc, (u, v), (uu, vv))


def test_physics_landmark_geometry_matches_rendering():
    """Regression: the physics simulator's landmark mapping matches its crop geometry, including the
    4 x 4 area averaging to the stored frame."""
    import sys, cv2
    sys.path.insert(0, _physim_dir())
    import simulate_physics as sp
    F = sp.F; Wf = Hf = 256 * F
    Xf, Yf = 1800.0, 420.0
    yy, xx = np.mgrid[:1400, :3000]
    canvas = np.exp(-((xx - Xf) ** 2 + (yy - Yf) ** 2) / (2 * 3.0 ** 2)).astype(np.float32)
    for rot, sc in [(0.0, 1.0), (1.5, 1.02)]:
        Mx, to_frame = sp.fine_affine(rot, sc, 1100.0, 150.0, Wf)
        img = cv2.warpAffine(canvas, Mx, (Wf, Hf), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        u, v = sp.fine_to_stored(to_frame(np.array([[Xf], [Yf]])))[:, 0]
        uu, vv = _peak(small)
        assert abs(uu - u) < 0.15 and abs(vv - v) < 0.15, (rot, sc, (u, v), (uu, vv))


def test_benchmark_aggregation_rejects_duplicates():
    """Regression: aggregation reads only the requested outputs and refuses duplicate keys."""
    import sys, os, tempfile, pandas as pd
    sys.path.insert(0, _physim_dir())
    import phys_bench as pb
    d = tempfile.mkdtemp()
    row = dict(cond="ideal_gate", scenario="nominal", seed=0, method="proposed", ate_x_um=1.0)
    a = os.path.join(d, "a.csv"); pd.DataFrame([row]).to_csv(a, index=False)
    b = os.path.join(d, "b.csv"); pd.DataFrame([row]).to_csv(b, index=False)
    assert len(pb.aggregate([a, a])) == 1                     # the same file listed twice is read once
    try:
        pb.aggregate([a, b]); raise AssertionError("duplicates were not rejected")
    except RuntimeError:
        pass


def test_vce_switch_reaches_guidance_presolve():
    """Regression: cfg.vce=False must also hold in the guidance pre-solve."""
    from .. import pipeline
    from ..simulate import simulate_sweep
    seen = []
    orig = pipeline.solve
    def spy(G, cfg, *a, **k):
        seen.append(bool(cfg.vce)); return orig(G, cfg, *a, **k)
    fr, co, gt = simulate_sweep(seed=1, n_frames=30, scenario="nominal")
    cfg = StitchConfig(); cfg.verbose = False; cfg.vce = False
    g, fs, ws = pipeline.condition(fr, cfg)
    pipeline.solve = spy
    try:
        pipeline.register_guided(fs, ws, cfg, len(fr), 12.0)
    finally:
        pipeline.solve = orig
    assert seen and not any(seen), seen


def test_generic_gm_millimetres_use_axial_spacing():
    """Regression: the GM detector's mm coordinates convert back to the true frame pixels
    with the reconstruction's lateral AND axial spacings (unbiased, SD ~1.5 px)."""
    from ..simulate import simulate_sweep
    c = StitchConfig(); d = []
    for seed in range(3):
        fr, co, gt = simulate_sweep(seed=seed, n_frames=60, scenario="nominal")
        for i, ci in enumerate(co):
            tu, tv = gt["gm_true"][i]
            ux, uy = ci["gm_x_mm"] / c.px_mm_x, ci["gm_y_mm"] / c.px_mm_y
            if np.isfinite(tu) and abs(ux - tu) < 6 and ci["quality_score"] > 0.3:
                d.append((ux - tu, uy - tv))
    d = np.array(d)
    assert len(d) > 50 and np.all(np.abs(np.median(d, 0)) < 0.5) and np.all(d.std(0) < 2.0), (np.median(d, 0), d.std(0))


def test_recal_seed_independent_of_path_separator():
    """Regression: the physics GM recalibration seed comes from the stored identifier."""
    import sys
    sys.path.insert(0, _physim_dir())
    import recal_gm as R
    D = dict(scenario="hard", seed=0)
    assert R.stable_tag(D, r".\cache\hard_s0.pkl") == R.stable_tag(D, "/x/cache/hard_s0.pkl") == ("hard_s0.pkl", 0)


def test_run_manifest_skips_only_complete_current_outputs():
    """Regression: a job is complete only with a matching version marker and all outputs."""
    import os, tempfile
    from .. import runmeta
    d = tempfile.mkdtemp(); outs = [os.path.join(d, "a.csv"), os.path.join(d, "b.json")]; mk = os.path.join(d, "a.done.json")
    open(outs[0], "w").write("x\n")
    assert not runmeta.is_done(mk, "v1", outs)                # missing output, no marker
    open(outs[1], "w").write("{}")
    assert not runmeta.is_done(mk, "v1", outs)                # no marker (interrupted job)
    runmeta.mark_done(mk, "v1", outs)
    assert runmeta.is_done(mk, "v1", outs)
    assert not runmeta.is_done(mk, "v2", outs)                # code/configuration changed


def test_interrupted_rerun_invalidates_marker():
    """Regression: a forced / version-triggered rerun deletes the marker before rewriting, and
    an aggregate manifest is refused when a job marker is missing or from another version."""
    import os, tempfile
    from .. import runmeta
    d = tempfile.mkdtemp(); outs = [os.path.join(d, "a.csv")]; mk = os.path.join(d, "a.done.json")
    open(outs[0], "w").write("x\n"); runmeta.mark_done(mk, "v1", outs)
    runmeta.begin(mk)                                         # rerun starts ... and is interrupted here
    assert not runmeta.is_done(mk, "v1", outs)
    runmeta.mark_done(mk, "v1", outs)
    m2 = os.path.join(d, "b.done.json"); runmeta.mark_done(m2, "v0", outs)
    for markers in ([mk, m2], [mk, os.path.join(d, "missing.done.json")]):
        try:
            runmeta.write_manifest(os.path.join(d, "MANIFEST.json"), "v1", markers); raise AssertionError("not refused")
        except RuntimeError:
            pass
    runmeta.write_manifest(os.path.join(d, "MANIFEST.json"), "v1", [mk])
    assert os.path.isfile(os.path.join(d, "MANIFEST.json"))


def test_physics_cache_fingerprint_covers_source_and_job():
    """Regression: the cache fingerprint covers simulator and trajectory-generator source and
    the PSF; the job parameters (scenario settings, frames, seed) are checked per sequence."""
    import sys, os, pickle, tempfile
    sys.path.insert(0, _physim_dir())
    import simulate_physics as sp, gen_cache as gc, recal_gm
    v = sp.cache_version()
    assert "simulate_physics.py:" in v and "simulate.py:" in v and sp.PSF_FILE in v
    assert gc.cache_key("jerky", 1) != gc.cache_key("jerky", 2)
    old = dict(gc.SCEN["jerky"]); gc.SCEN["jerky"]["liftoff_rate"] = 0.07
    k2 = gc.cache_key("jerky", 1); gc.SCEN["jerky"].clear(); gc.SCEN["jerky"].update(old)
    assert k2 != gc.cache_key("jerky", 1)
    f = os.path.join(tempfile.mkdtemp(), "jerky_s1.pkl")
    D = dict(scenario="jerky", seed=1, cache_version=v, cache_key=gc.cache_key("jerky", 1), recal_version=recal_gm.RECAL_VERSION)
    pickle.dump(D, open(f, "wb")); assert gc.validate(f)[0]
    for k, bad in (("cache_version", v + "x"), ("cache_key", gc.cache_key("jerky", 2)), ("recal_version", "old")):
        pickle.dump(dict(D, **{k: bad}), open(f, "wb")); assert not gc.validate(f)[0], k


if __name__ == "__main__":
    test_registration_convention(); test_graph_recovers_chain_with_outlier()
    test_generic_landmark_geometry_matches_rendering(); test_physics_landmark_geometry_matches_rendering()
    test_benchmark_aggregation_rejects_duplicates(); test_final_solve_keeps_gap_bridges()
    test_vce_switch_reaches_guidance_presolve(); test_generic_gm_millimetres_use_axial_spacing()
    test_recal_seed_independent_of_path_separator(); test_run_manifest_skips_only_complete_current_outputs()
    test_interrupted_rerun_invalidates_marker(); test_physics_cache_fingerprint_covers_source_and_job()
    print("all tests passed")
