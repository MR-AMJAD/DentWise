"""Sensitivity of the reconstruction to the strength of the probe-fixed layer (mixed scenario; Fig. 4(b)).
    python benchmarks/artifact_sweep.py <seeds, e.g. 0 or 0,2,4> [strengths, e.g. 0.27,0.33]      ARTIFACT_OUT=folder
The paper uses seeds 0-4 and the strengths 0, 0.05, 0.1, 0.2, 0.27, 0.33, 0.4 and 0.8 (eight levels x five seeds), e.g.
    for s in 0 1 2 3 4: python benchmarks/artifact_sweep.py s ;  python benchmarks/artifact_sweep.py s 0.27,0.33
Each call writes results/artifact/artifact_sweep_<seeds>[_s<strengths>].csv; the grouping of seeds does not change
the results."""
import sys, os, copy, time, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.environ.get('ARTIFACT_OUT', os.path.join(ROOT, 'results', 'artifact'))
sys.path.insert(0, ROOT)
from ioustitch import StitchConfig, stitch_sweep
from ioustitch.simulate import simulate_sweep
from ioustitch.pipeline import register_all, condition
from ioustitch.registration import estimate_speckle_area
from ioustitch.evaluate import trajectory_metrics
from ioustitch import baselines as bl
from ioustitch import preprocess as pp
seeds = [int(s) for s in sys.argv[1].split(',')]
rows = []
STR = tuple(float(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (0.0, 0.05, 0.1, 0.2, 0.4, 0.8)
from ioustitch.pipeline import initial_probe_fixed
for a in STR:
    for s in seeds:
        fr, co, gt = simulate_sweep(seed=s, n_frames=100, scenario='mixed', liftoff_rate=0.04, artifact_strength=a)
        g = [pp.to_gray01(x) for x in fr]
        ratio = float((gt['probe_fixed'] ** 2).sum() / np.mean([x ** 2 for x in g], 0).sum())
        eb0 = float((initial_probe_fixed(g) ** 2).sum() / np.mean([x ** 2 for x in g], 0).sum())
        cfg = StitchConfig(); cfg.verbose = False
        def rec(name, p, q, sd=None):
            m = trajectory_metrics(p, q, gt['p'], gt['q'], gt['t'], cfg.px_mm_x, cfg.px_mm_y, sd)
            rows.append(dict(strength=a, seed=s, energy_ratio=ratio, eb0=eb0, method=name, **m))
        r = stitch_sweep(fr, co, cfg); rec('proposed', r['p'], r['q'], r['sd_p'])
        c = copy.deepcopy(cfg); c.layer_separation = False
        r2 = stitch_sweep(fr, co, c); rec('-layer', r2['p'], r2['q'], r2['sd_p'])
        g2, fs, ws = condition(fr, cfg)
        seq = register_all(fs, ws, cfg, [(i, i + 1) for i in range(len(fr) - 1)], estimate_speckle_area(fs, ws))
        rec('seq_reg', *bl.seq_reg(seq, len(fr)))
        pcs, qcs, _ = bl.chain_same_frontend(fr, cfg, timestamps=np.array([c['ts'] for c in co])); rec('chain_sep', pcs, qcs)
        rec('gm_sg', *bl.gm_sg(co, cfg.px_mm_x, cfg.px_mm_y))
        os.makedirs(OUTD, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUTD, f'artifact_sweep_{sys.argv[1].replace(",", "")}' + (f'_s{sys.argv[2].replace(",", "_")}' if len(sys.argv) > 2 else '') + '.csv'), index=False)
        print(a, s, round(ratio, 4), {rw['method']: round(rw['ate_x_um']) for rw in rows[-5:]}, flush=True)
