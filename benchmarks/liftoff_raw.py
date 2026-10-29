"""Lift-off detection without layer separation (the detector of the '-layer' variant: raw tissue mass) on the
25 generic sequences, for comparison with the pipeline's labels (Table S4, column 'Detected, no separation').
    python benchmarks/liftoff_raw.py [seeds, default 0,1,2,3,4] [output folder, default results/generic]"""
import sys, os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ioustitch import StitchConfig
from ioustitch.simulate import simulate_sweep
from ioustitch.benchmark import SCEN
from ioustitch.pipeline import condition, detect_blind
SEEDS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '0,1,2,3,4').split(',')]
OUTF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'results', 'generic')
rows = []
for sc in SCEN:
    for s in SEEDS:
        fr, co, gt = simulate_sweep(seed=s, n_frames=100, **SCEN[sc]); cfg = StitchConfig()
        g, fs, ws = condition(fr, cfg); b = detect_blind(ws); L = gt['lift']
        m = np.array([w.sum() for w in ws]); m = m / np.median(m)
        rows.append(dict(scenario=sc, seed=s, lift=int(L.sum()), tp_raw=int((b & L).sum()), fp_raw=int((b & ~L).sum()),
                         min_mass_lift_raw=float(m[L].min()) if L.any() else np.nan))
os.makedirs(OUTF, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(OUTF, 'liftoff_raw.csv'), index=False)
print(pd.DataFrame(rows).groupby('scenario')[['lift', 'tp_raw', 'fp_raw']].sum(), pd.DataFrame(rows).min_mass_lift_raw.min())
