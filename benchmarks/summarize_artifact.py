"""Sensitivity to the strength of the probe-fixed layer (Section III-A, Fig. 4(b)) and recovery of the layer energy
by the trajectory-free estimate B0.

    python benchmarks/summarize_artifact.py [results/artifact]

Writes artifact_summary.csv (median lateral ATE per method and strength level, um) and artifact_summary.json.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'artifact')
a = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(D, 'artifact_sweep_*.csv')))], ignore_index=True)
assert not a.duplicated(['strength', 'seed', 'method']).any()
EB = a.groupby('strength').energy_ratio.median() * 100
T = a.groupby(['strength', 'method']).ate_x_um.median().unstack('method')
T.insert(0, 'EB_pct', EB)
T.to_csv(os.path.join(D, 'artifact_summary.csv'))
print('median lateral ATE (um) by strength level (E_B: median true probe-fixed energy, %)')
print(T.round(1).to_string())
p = a[a.method == 'proposed']
q = p[p.energy_ratio >= 0.02]
out = dict(n_sequences_per_method=int(a.groupby('method').size().iloc[0]), levels=int(a.strength.nunique()),
           seeds=sorted(int(s) for s in a.seed.unique()), EB_pct=[float(v) for v in EB],
           median_ate_um={m: [float(v) for v in T[m]] for m in T.columns if m != 'EB_pct'},
           B0_energy_recovered_EB_ge_2pct=dict(n=len(q), min=float((q.eb0 / q.energy_ratio).min()),
                                               max=float((q.eb0 / q.energy_ratio).max())))
json.dump(out, open(os.path.join(D, 'artifact_summary.json'), 'w'), indent=1)
print(json.dumps(out['B0_energy_recovered_EB_ge_2pct']))
