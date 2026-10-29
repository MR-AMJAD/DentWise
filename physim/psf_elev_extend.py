"""Extend the tabulated elevational sensitivity of the calibrated PSF (psf_40MHz_cal3.npz, +-0.4 mm) to +-1.0 mm,
with the same array and imaging parameters (receive F-number 6, transmit F-number 7, 45 % bandwidth), so that
elev_weights() never extrapolates (slice offsets relative to the probe reach +-0.65 mm). The lateral-axial PSF is
copied unchanged. Writes psf_40MHz_cal4.npz and reports the agreement with cal3 on the common offsets."""
import numpy as np, time, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psf as ps
d = dict(np.load('psf_40MHz_cal3.npz'))
P = ps.array_param(float(d['fn_rx']), float(d['bandwidth']))
fn_tx = float(d['fn_tx'])
eoffs = np.arange(-100, 101) * 10e-6
EL = []; t0 = time.time()
for z0 in d['depths']:
    e, _ = ps.line_psf(z0, eoffs, P, fn_tx=fn_tx, elev=True)
    EL.append(np.abs(e).max(axis=0))
    print(f'z = {z0 * 1e3:.1f} mm  [{time.time() - t0:.0f}s]', flush=True)
EL = np.array(EL)
# agreement on the common grid
common = np.isin(np.round(eoffs * 1e6).astype(int), np.round(d['eoffs'] * 1e6).astype(int))
old = d['elev'] / d['elev'].max(axis=1, keepdims=True); new = EL[:, common] / EL[:, common].max(axis=1, keepdims=True)
print('max abs difference on common offsets (relative to per-depth max):', float(np.abs(old - new).max()))
rel = EL / EL[:, len(eoffs) // 2][:, None]
for k, z0 in enumerate(d['depths']):
    v = {o: float(np.interp(o * 1e-3, eoffs, rel[k])) for o in (0.4, 0.5, 0.65, 0.8, 1.0)}
    print(f'z {z0 * 1e3:.1f} mm  relative sensitivity at', {k_: round(v_, 3) for k_, v_ in v.items()})
d['eoffs'] = eoffs; d['elev'] = EL
np.savez_compressed('psf_40MHz_cal4.npz', **d)
print('saved psf_40MHz_cal4.npz')
