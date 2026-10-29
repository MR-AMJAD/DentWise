"""Summaries of the generic benchmark: Table S1, Table S3, Table S4, Section S5 and the numbers of Section III-A.

    python benchmarks/summarize_generic.py [results/generic]

Reads bench_all.csv, val_all.csv, frames_all.csv, liftoff_raw.csv and the per-sequence *_traj.npz files written by
bench_generic.py / liftoff_raw.py and writes table_s1.csv, table_s3.csv, table_s4.csv, val_fig.csv and summary.json
into the same folder.
"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ioustitch import StitchConfig
from ioustitch.evaluate import trajectory_metrics

D = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'generic')).rstrip('/') + '/'
b = pd.read_csv(D + 'bench_all.csv'); V = pd.read_csv(D + 'val_all.csv'); F = pd.read_csv(D + 'frames_all.csv')
SC = ['nominal', 'jerky', 'reversal', 'mixed', 'hard']
ORDER = ['proposed', '-layer', '-loop', '-gnc', '-vce', '-multilag', '-landmark', 'chain_sep', 'seq_reg', 'cf_imu', 'gm_sg']
LABEL = {'proposed': 'Proposed', '-layer': '- layer separation (C1)', '-loop': '- revisit closure (C3)',
         '-gnc': '- GNC (plain least squares)', '-vce': '- VCE (fixed weights)', '-multilag': '- multi-lag (scheduled lag 1)',
         '-landmark': '- GM landmark factors', 'chain_sep': 'Chaining* (same front end)',
         'seq_reg': 'Pairwise chaining, raw frames', 'cf_imu': 'Complementary filter', 'gm_sg': 'GM tracking'}
b = b[b.method.isin(ORDER)]
cfg = StitchConfig()
out = {}

# ---------------- Table S1: paired Wilcoxon tests vs the proposed method, Holm over the 10 comparisons ----------------
base = b[b.method == 'proposed'].set_index(['scenario', 'seed']).ate_x_um
P = {}
for m in ORDER[1:]:
    o = b[b.method == m].set_index(['scenario', 'seed']).ate_x_um
    k = base.index.intersection(o.index)
    P[m] = float(wilcoxon(base[k], o[k]).pvalue) if (base[k] != o[k]).any() else 1.0
holm, run = {}, 0.0
for r, m in enumerate(sorted(P, key=P.get)):
    run = max(run, min(1.0, P[m] * (len(P) - r))); holm[m] = run
rows = []
for m in ORDER:
    d = b[b.method == m]
    rows.append(dict(method=m, label=LABEL[m], **{s: d[d.scenario == s].ate_x_um.median() for s in SC},
                     median=d.ate_x_um.median(), q1=d.ate_x_um.quantile(.25), q3=d.ate_x_um.quantile(.75),
                     mean=d.ate_x_um.mean(), sd=d.ate_x_um.std(), failures=int((d.ate_x_um > 100).sum()),
                     sle_pct=d.sle_pct.median(), pfid=d.pfid.median(), ate_y=d.ate_y_um.median(),
                     rpe1s=d.rpe1s_x_um.median(), p=P.get(m, np.nan), p_holm=holm.get(m, np.nan)))
T1 = pd.DataFrame(rows); T1.to_csv(D + 'table_s1.csv', index=False)
print('Table S1 (lateral ATE, um)')
print(T1[['label', 'median', 'mean', 'sd', 'failures', 'sle_pct', 'pfid', 'p_holm']].round(3).to_string(index=False))

# ---------------- Table S3: photometric consistency of each trajectory minus that of the true trajectory -------------
name = dict(proposed='proposed', chain_sep='chain_sep', chain_raw='seq_reg', gm_sg='gm_sg', cf='cf_imu')
for k, m in name.items():
    V[f'ate_{k}'] = [b[(b.scenario == s) & (b.seed == e) & (b.method == m)].ate_x_um.values[0] for s, e in zip(V.scenario, V.seed)]
z = []
for s, e in zip(V.scenario, V.seed):
    tr = np.load(f'{D}{s}_s{e}_traj.npz'); n = len(tr['t'])
    z.append(trajectory_metrics(np.zeros(n), np.zeros(n), tr['gt_p'], tr['gt_q'], tr['t'], cfg.px_mm_x, cfg.px_mm_y)['ate_x_um'])
V['ate_zero'] = z
V.to_csv(D + 'val_fig.csv', index=False)
K = ['proposed', 'chain_sep', 'chain_raw', 'gm_sg', 'cf', 'zero']
s3 = {}
for k in K:
    s3[k] = {s: float((V[V.scenario == s][f'pc_{k}'] - V[V.scenario == s]['pc_gt']).median()) for s in SC}
    s3[k]['all'] = float((V[f'pc_{k}'] - V['pc_gt']).median())
    s3[k]['auxiliary_lags'] = float((V[f'pch_{k}'] - V['pch_gt']).median())
    s3[k]['median_ate_um'] = float(V[f'ate_{k}'].median())
pd.DataFrame(s3).T.to_csv(D + 'table_s3.csv')
x, y = [], []
for k in K:
    x += list(V[f'pc_{k}']); y += list(np.log(V[f'ate_{k}']))
rho = spearmanr(x, y)
out['table_s3'] = dict(rows=s3, pc_true_median={s: float(V[V.scenario == s].pc_gt.median()) for s in SC},
                       spearman_pc_logate=float(rho.statistic), spearman_p=float(rho.pvalue), n_pairs=len(x))
print('\nTable S3 (PC minus PC of the true trajectory)'); print(pd.DataFrame(s3).T.round(3).to_string())
print('Spearman PC vs log ATE: %.2f over %d pairs' % (rho.statistic, len(x)))

# ---------------- Table S4: lift-off detection ----------------
raw = pd.read_csv(D + 'liftoff_raw.csv') if os.path.isfile(D + 'liftoff_raw.csv') else None
s4 = {}
for s in SC + ['all']:
    v = V if s == 'all' else V[V.scenario == s]
    s4[s] = dict(lift_off=int(v.n_lift.sum()), detected_no_separation=int(raw.tp_raw.sum() if s == 'all' else raw[raw.scenario == s].tp_raw.sum())
                 if raw is not None else None, detected_pipeline=int(v.blind_tp.sum()), false_alarms=int(v.blind_fp.sum()),
                 max_rel_mass_lift=float(v.max_mass_lift.max()), min_rel_mass_sighted=float(v.min_mass_sighted.min()))
s4['all']['sighted_frames'] = int(len(F) - F.lift.sum())
pd.DataFrame(s4).T.to_csv(D + 'table_s4.csv')
out['table_s4'] = s4
print('\nTable S4'); print(pd.DataFrame(s4).T.to_string())
hard = V[V.scenario == 'hard']
out['missed_lift_off'] = [dict(seed=int(r.seed), missed=int(r.blind_fn), EB_true_pct=round(100 * r.EB_true, 1)) for r in hard.itertuples() if r.blind_fn]
out['EB_true_pct_range'] = [100 * float(V.EB_true.min()), 100 * float(V.EB_true.max())]
out['EB_true_pct_range_hard'] = [100 * float(hard.EB_true.min()), 100 * float(hard.EB_true.max())]
out['layer_estimate'] = dict(B0_corr=[float(V.B0_corr.min()), float(V.B0_corr.median()), float(V.B0_corr.max())],
                             B0_energy_recovered=[float((V.EB0 / V.EB_true).min()), float((V.EB0 / V.EB_true).max())],
                             Bhat_corr=[float(V.B_corr.min()), float(V.B_corr.median()), float(V.B_corr.max())],
                             Bhat_energy_recovered_median=float((V.EB_est / V.EB_true).median()),
                             EB0_pct_median=100 * float(V.EB0.median()))

# ---------------- Section S5: coverage of the reported uncertainty ----------------
pr = b[b.method == 'proposed'].set_index(['scenario', 'seed'])
F['ok_seq'] = [pr.loc[(s, k)].ate_x_um <= 100 for s, k in zip(F.scenario, F.seed)]
sighted = F[~F.lift.astype(bool)]
px_um = cfg.px_mm_x * 1e3
cal = {}
for nm, d in (('success', sighted[sighted.ok_seq]), ('failed', sighted[~sighted.ok_seq])):
    zz = np.abs(d.err_px / d.sd_px)
    cal[nm] = dict(n_frames=len(d), within05=float((zz <= 0.5).mean()), within1=float((zz <= 1).mean()),
                   within196=float((zz <= 1.96).mean()), median_sigma_um=float(d.sd_px.median() * px_um),
                   median_abs_err_um=float(np.abs(d.err_px).median() * px_um))
cal['n_success'] = int((pr.ate_x_um <= 100).sum())
cal['failed_ate_um'] = sorted(pr[pr.ate_x_um > 100].ate_x_um.round(0).tolist(), reverse=True)
cal['failed_cover95'] = pr[pr.ate_x_um > 100].cover95.round(2).tolist()
out['section_s5'] = cal

# ---------------- Section III-A ----------------
p_ = b[b.method == 'proposed']
out['text'] = dict(proposed_median_ate_um=float(p_.ate_x_um.median()), proposed_axial_ate_um=float(p_.ate_y_um.median()),
                   proposed_rpe1s_um=float(p_.rpe1s_x_um.median()), proposed_sle_pct=float(p_.sle_pct.median()),
                   proposed_pfid=float(p_.pfid.median()), runtime_median_s=float(p_.runtime_s.median()),
                   vce_sd={c: float(p_[c].median()) for c in p_.columns if c.endswith('_sd') and p_[c].notna().any()})
out['p'] = P; out['p_holm'] = holm
json.dump(out, open(D + 'summary.json', 'w'), indent=1, default=float)
print('\nSection S5', json.dumps(cal, indent=1, default=float))
print('Section III-A', json.dumps(out['text'], indent=1, default=float))
