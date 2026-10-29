"""Table S5 and the text numbers of the physics-based benchmark:  python physim/summarize_phys.py [results folder]
Writes table_s5_<cond>.csv and summary_phys.json into the results folder."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'results', 'physics')
d = pd.read_csv(os.path.join(OUT, 'phys_bench_all.csv'))
SC = ['nominal', 'jerky', 'reversal', 'mixed', 'hard']
ORDER = ['proposed', '-layer', '-loop', '-gnc', '-vce', '-multilag', 'chain_sep', 'seq_reg', 'cf_imu', 'gm_sg']
BASE = ['chain_sep', 'seq_reg', 'cf_imu', 'gm_sg']          # Holm family: the four baselines within each condition
out = {}
for cond in ('ideal_gate', 'ungated'):
    x = d[d.cond == cond]; pr = x[x.method == 'proposed'].set_index(['scenario', 'seed']).ate_x_um
    rows = []; P = {}
    for m in ORDER:
        y = x[x.method == m].set_index(['scenario', 'seed']).ate_x_um
        k = pr.index.intersection(y.index)
        P[m] = float(wilcoxon(pr[k], y[k]).pvalue) if m != 'proposed' and (pr[k] != y[k]).any() else np.nan
        rows.append(dict(method=m, **{s: float(x[(x.method == m) & (x.scenario == s)].ate_x_um.median()) for s in SC},
                         all=float(y.median()), q1=float(y.quantile(.25)), q3=float(y.quantile(.75)),
                         over500=int((y > 500).sum()), p=P[m], p90=float(y.quantile(.9)), mean=float(y.mean()), max=float(y.max()),
                         runtime=float(x[x.method == m].runtime_s.median())))
    # Holm family: BASE
    ps = sorted(BASE, key=lambda m: P[m]); run = 0; holm = {}
    for r, m in enumerate(ps):
        run = max(run, min(1, P[m] * (len(ps) - r))); holm[m] = run
    for r in rows:
        r['p_holm'] = holm.get(r['method'], np.nan)
    T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, f'table_s5_{cond}.csv'), index=False)
    print(cond); print(T.round(3).to_string())
    p = x[x.method == 'proposed']
    out[cond] = dict(sle=float(p.sle_pct.median()), cover95_median=float(p.cover95.median()), cover95_mean=float(p.cover95.mean()),
                     n_lift=int(p.n_lift.sum()), blind_tp=int(p.blind_tp.sum()), blind_fp=int(p.blind_fp.sum()),
                     failures=[f'{a} s{b}' for a, b in p[p.ate_x_um > 500][['scenario', 'seed']].values])
json.dump(out, open(os.path.join(OUT, 'summary_phys.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
