"""
Simulation benchmark protocol (shared by the generic and the physics-based benchmark).

Defines the scenarios, the ablations and ONE function, `reconstruct_all`, that produces every trajectory compared in
the paper, so the two benchmark drivers cannot drift apart:

  proposed        full method (stitch_sweep)
  ablations       every ablation is run END TO END: stitch_sweep with one component switched off, so registration,
                  guidance (including the guidance pre-solve), layer estimate and revisit candidates are those of the
                  ablated method.
                    -layer     no probe-fixed layer separation (C1)
                    -gnc       plain least squares instead of GNC (also in the guidance pre-solve)
                    -vce       fixed a-priori weights instead of VCE (also in the guidance pre-solve)
                    -multilag  scheduled lag set reduced to lag 1; gap bridges and revisit factors are KEPT, so this
                               removes the scheduled longer-lag factors, not every non-consecutive pair
                    -loop      no revisit closure
                    -landmark  no GM landmark factors
  chain_sep       "Chaining*": pairwise chaining with the same front-end procedure (baselines.chain_same_frontend)
  seq_reg         pairwise chaining of the unseparated frames (no gap handling)
  gm_sg           GM tracking (previous offline method)
  cf_imu          complementary filter of the previous live system: GM position fused with the simulated
                  probe-orientation signal of the simulator (simulation only)

Drivers: benchmarks/bench_generic.py (generic simulator), physim/phys_bench.py (physics-based simulator).
"""
import copy, time
import numpy as np

from .config import StitchConfig
from .pipeline import stitch_sweep, condition, register_all
from .registration import estimate_speckle_area
from . import baselines as bl

SCEN = {
    "nominal":  dict(scenario="nominal"),
    "jerky":    dict(scenario="jerky", liftoff_rate=0.06),
    "reversal": dict(scenario="reversal"),
    "mixed":    dict(scenario="mixed", liftoff_rate=0.04),
    "hard":     dict(scenario="mixed", artifact_strength=1.2, decorrelation=0.45, liftoff_rate=0.06,
                     elevation_mod=0.5, max_rot_deg=1.5, max_scale=0.03),
}

E2E_ABLATIONS = {
    "-layer": dict(layer_separation=False),
    "-gnc": dict(gnc=False),
    "-vce": dict(vce=False),
    "-multilag": dict(lags=(1,)),
    "-loop": dict(use_loop_closure=False),
    "-landmark": dict(use_landmark=False),
}
GENERIC_ABLATIONS = list(E2E_ABLATIONS)
PHYSICS_ABLATIONS = ["-layer", "-gnc", "-vce", "-multilag", "-loop"]


def reconstruct_all(frames, coords, cfg=None, ablations=GENERIC_ABLATIONS, timestamps=None):
    """Every trajectory of the benchmark for one sequence.
    Returns (full, out): `full` is the stitch_sweep result of the proposed method; `out` maps method name ->
    dict(p, q, sd_p or None, runtime_s, blind or None)."""
    cfg = copy.deepcopy(cfg or StitchConfig())
    px, py = cfg.px_mm_x, cfg.px_mm_y
    ts = np.array([c['ts'] for c in coords]) if timestamps is None else np.asarray(timestamps, float)
    out = {}
    t0 = time.time(); full = stitch_sweep(frames, coords, cfg)
    out['proposed'] = dict(p=full['p'], q=full['q'], sd_p=full['sd_p'], runtime_s=time.time() - t0, blind=full['blind'])
    for name in ablations:
        c = copy.deepcopy(cfg)
        for k, v in E2E_ABLATIONS[name].items():
            setattr(c, k, v)
        t1 = time.time(); r = stitch_sweep(frames, coords, c)
        out[name] = dict(p=r['p'], q=r['q'], sd_p=r['sd_p'], runtime_s=time.time() - t1, blind=r['blind'])
    t1 = time.time(); p, q, b = bl.chain_same_frontend(frames, cfg, timestamps=ts)
    out['chain_sep'] = dict(p=p, q=q, sd_p=None, runtime_s=time.time() - t1, blind=b)
    N = len(frames)
    g, fs, ws = condition(frames, cfg)
    seq = register_all(fs, ws, cfg, [(i, i + 1) for i in range(N - 1)], estimate_speckle_area(fs, ws))
    p, q = bl.seq_reg(seq, N); out['seq_reg'] = dict(p=p, q=q, sd_p=None, runtime_s=np.nan, blind=None)
    p, q = bl.gm_sg(coords, px, py); out['gm_sg'] = dict(p=p, q=q, sd_p=None, runtime_s=np.nan, blind=None)
    p, q = bl.cf_imu(coords, px, py); out['cf_imu'] = dict(p=p, q=q, sd_p=None, runtime_s=np.nan, blind=None)
    return full, out
