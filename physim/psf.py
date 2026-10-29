"""Two-way point-spread function of a 40-MHz linear array, computed with PyMUST (SIMUS + DAS).

Array (assumed, chosen to match the 7.0-mm frame of the clinical system): 256 elements, pitch 27.3 um,
kerf 4 um, centre frequency 40 MHz, pulse-echo -6 dB fractional bandwidth 60 %, element height 1.0 mm,
elevation focus 3.0 mm. Imaging: line-by-line focused transmission (transmit focus 2.5 mm, transmit
F-number 2), dynamic receive focusing (F-number 1.5), I/Q demodulation.

For each depth z0 the lateral-axial PSF is sampled by moving one point scatterer laterally relative to a fixed
image line (x = 0); the elevational sensitivity is sampled by moving it in elevation (3-D SIMUS).
Output: psf_40MHz.npz with complex PSFs on a 6.84 um (lateral) x 7.17 um (axial) grid.

NOTE: the defaults above (F-numbers 1.5/2, 60 % bandwidth) are NOT the values used by the simulator. The benchmark
uses psf_40MHz_cal3.npz (receive/transmit F-numbers 6/7, 45 % bandwidth; `python psf.py 6 7 psf_40MHz_cal3.npz 45`),
calibrated to clinical image statistics, extended in elevation to +-1.0 mm by psf_elev_extend.py
(psf_40MHz_cal4.npz). These are assumed/calibrated parameters, not a verified model of the clinical probe.
"""
import numpy as _np
if not hasattr(_np, 'concat'):
    _np.concat = _np.concatenate
import numpy as np, pymust, time

DX = 7.0e-3 / 256 / 4          # lateral sample (m): 1/4 of the stored-frame pixel
DZ = 7.3440e-3 / 256 / 4       # axial sample (m)


def array_param(fn_rx=1.5, bw=60):
    P = pymust.utils.Param()
    P.fc = 40e6; P.pitch = 7.0e-3 / 256; P.kerf = 4e-6; P.width = P.pitch - P.kerf
    P.Nelements = 256; P.bandwidth = bw; P.c = 1540; P.height = 1.0e-3; P.focus = 3.0e-3
    P.fs = 4 * P.fc; P.fnumber = fn_rx
    return P


def line_psf(z0, offs, P, zf=2.5e-3, fn_tx=2.0, elev=False):
    """Echo of one point scatterer on the image line x = 0, beamformed by delay-and-sum with dynamic receive
    focusing; returns base-band I/Q samples on an axial window around z0, for each scatterer offset."""
    xs = (np.arange(P.Nelements) - (P.Nelements - 1) / 2) * P.pitch
    P.TXapodization = (np.abs(xs) <= zf / fn_tx / 2).astype(float)
    dly = pymust.txdelay(0.0, zf, P).ravel()
    tc = dly[np.argmin(np.abs(xs))]                         # firing time of the central element
    zg = z0 + np.arange(-40, 41) * DZ                      # axial window +-0.29 mm
    ap = np.abs(xs)[None, :] <= zg[:, None] / (2 * P.fnumber)
    w = ap * (0.5 + 0.5 * np.cos(np.pi * xs[None, :] / np.maximum(zg[:, None] / (2 * P.fnumber), 1e-9)))
    tau = tc + zg[:, None] / P.c + np.sqrt(xs[None, :] ** 2 + zg[:, None] ** 2) / P.c   # two-way delay
    out = []
    for o in offs:
        if elev:
            RF, _ = pymust.simus(np.array([0.0]), np.array([o]), np.array([z0]), np.array([1.0]), dly[None], P)
        else:
            RF, _ = pymust.simus(np.array([o]), np.array([z0]), np.array([1.0]), dly[None], P)
        IQ = pymust.rf2iq(RF, P)
        t = np.arange(IQ.shape[0]) / P.fs
        acc = np.zeros(zg.size, complex)
        for e in np.nonzero(ap.any(0))[0]:
            v = np.interp(tau[:, e], t, IQ[:, e].real, 0, 0) + 1j * np.interp(tau[:, e], t, IQ[:, e].imag, 0, 0)
            acc += w[:, e] * v * np.exp(2j * np.pi * P.fc * tau[:, e])
        out.append(acc * np.exp(-2j * np.pi * P.fc * (tc + 2 * zg / P.c)))
    return np.array(out).T, zg                              # (axial, offset)


if __name__ == '__main__':
    import sys
    fn_rx = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
    fn_tx = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    outname = sys.argv[3] if len(sys.argv) > 3 else 'psf_40MHz.npz'
    bw = float(sys.argv[4]) if len(sys.argv) > 4 else 60
    P = array_param(fn_rx, bw)
    depths = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0]) * 1e-3
    offs = np.arange(-100, 101) * DX                       # +-0.68 mm lateral
    eoffs = np.arange(-40, 41) * 10e-6                     # +-0.4 mm elevation
    PSF, EL = [], []
    t0 = time.time()
    for z0 in depths:
        h, zg = line_psf(z0, offs, P, fn_tx=fn_tx)
        PSF.append(h)
        e, _ = line_psf(z0, eoffs, P, fn_tx=fn_tx, elev=True)
        EL.append(np.abs(e).max(axis=0))
        print(f'z = {z0 * 1e3:.1f} mm  lateral FWHM {np.ptp(offs[np.abs(h).max(0) > np.abs(h).max() / 2]) * 1e6:.0f} um'
              f'  axial FWHM {np.ptp(zg[np.abs(h).max(1) > np.abs(h).max() / 2] - z0) * 1e6:.0f} um'
              f'  elevation FWHM {np.ptp(eoffs[EL[-1] > EL[-1].max() / 2]) * 1e6:.0f} um   [{time.time() - t0:.0f}s]',
              flush=True)
    np.savez_compressed(outname, fn_rx=fn_rx, fn_tx=fn_tx, psf=np.array(PSF), depths=depths, offs=offs, dz=DZ, dx=DX,
                        eoffs=eoffs, elev=np.array(EL), fc=P.fc, bandwidth=P.bandwidth, pitch=P.pitch,
                        height=P.height, efocus=P.focus)
