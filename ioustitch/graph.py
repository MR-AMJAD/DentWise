"""
Robust factor graph for the probe trajectory (C4).

State  z = [ p_0..p_{N-1} | q_0..q_{N-1} ]
    p, q   : lateral / axial canvas position of each frame (px)

Factor groups (each robust group has its own variance component, estimated from the data):
    image_L1, image_L2-3, image_L4-8, image_L9+ : masked-NCC pair registrations by lag band, full 2x2 Laplace covariances
    loop      : revisit registrations (same model as image)
    landmark  : GM landmark displacement between consecutive frames
Non-robust priors: gauge, constant-velocity (acceleration random-walk) motion model.

Estimation:
    * Graduated non-convexity with the Geman-McClure loss (Yang et al., IEEE RA-L 2020): outlier rejection without an
      initial guess (false registrations across lift-off, landmark switches).
    * Median form of Foerstner's variance-component estimation (VCE): per-group noise scale from the residuals and
      the redundancy numbers, instead of hand-set fusion weights.
    * Marginal covariance of every frame position from the inverse information matrix of the final solve, with the
      gauge moved to the sweep centroid.
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROBUST_GROUPS = {"image", "loop", "landmark"}


def is_robust(group):
    return group.split("_")[0] in ROBUST_GROUPS


@dataclass
class Factor:
    idx: np.ndarray        # variable indices (k,)
    A: np.ndarray          # (m, k) coefficients
    b: np.ndarray          # (m,)
    L: np.ndarray          # (m, m) whitening (sqrt information), residual = L (A z_idx - b)
    group: str
    meta: dict = field(default_factory=dict)


class Graph:
    def __init__(self, n_frames: int):
        self.N = n_frames
        self.n = 2 * n_frames
        self.factors: List[Factor] = []

    # ---------------- variable indexing ----------------
    def ip(self, i): return i
    def iq(self, i): return self.N + i

    def add(self, idx, A, b, L, group, **meta):
        self.factors.append(Factor(np.asarray(idx, int), np.atleast_2d(np.asarray(A, float)),
                                   np.atleast_1d(np.asarray(b, float)),
                                   np.atleast_2d(np.asarray(L, float)), group, meta))

    # ---------------- factor constructors ----------------
    def add_relative_2d(self, i, j, d, cov, group="image", **meta):
        """(p_j - p_i, q_j - q_i) = d  with covariance cov (2x2)."""
        info = np.linalg.inv(cov)
        L = np.linalg.cholesky(info).T
        idx = [self.ip(i), self.ip(j), self.iq(i), self.iq(j)]
        A = np.array([[-1, 1, 0, 0], [0, 0, -1, 1]], float)
        self.add(idx, A, d, L, group, i=i, j=j, **meta)

    def add_relative_1d(self, i, j, dx, sigma, axis="x", group="landmark", **meta):
        f = self.ip if axis == "x" else self.iq
        self.add([f(i), f(j)], [[-1, 1]], [dx], [[1.0 / sigma]], group, i=i, j=j, axis=axis, **meta)

    def add_gauge(self, i=0, sigma=1e-3):
        self.add([self.ip(i)], [[1]], [0.0], [[1 / sigma]], "gauge")
        self.add([self.iq(i)], [[1]], [0.0], [[1 / sigma]], "gauge")

    def add_motion_prior(self, t, sigma_acc):
        """Constant-velocity (acceleration random-walk) prior on p and q."""
        for k in range(1, self.N - 1):
            d0 = max(t[k] - t[k - 1], 1e-3); d1 = max(t[k + 1] - t[k], 1e-3)
            s = sigma_acc * np.sqrt(0.5 * (d0 + d1))
            a = [1 / d0, -(1 / d0 + 1 / d1), 1 / d1]
            for f in (self.ip, self.iq):
                self.add([f(k - 1), f(k), f(k + 1)], [a], [0.0], [[1 / s]], "motion")
        # tiny first-order prior so frames with no other constraint stay put
        for k in range(1, self.N):
            dt = max(t[k] - t[k - 1], 1e-3)
            s = 2000.0 * np.sqrt(dt)
            for f in (self.ip, self.iq):
                self.add([f(k - 1), f(k)], [[-1, 1]], [0.0], [[1 / s]], "motion")


# ======================================================================
#  Solver
# ======================================================================
def _assemble(factors, n, scales, weights):
    rows, cols, vals, rhs, fid = [], [], [], [], []
    r0 = 0
    for k, f in enumerate(factors):
        s = scales.get(f.group, 1.0)
        w = np.sqrt(weights[k]) / np.sqrt(s)
        LA = (f.L @ f.A) * w
        Lb = (f.L @ f.b) * w
        m = LA.shape[0]
        for a in range(m):
            rows.extend([r0 + a] * len(f.idx)); cols.extend(f.idx); vals.extend(LA[a])
        rhs.extend(Lb); fid.extend([k] * m)
        r0 += m
    J = sp.csr_matrix((vals, (rows, cols)), shape=(r0, n))
    return J, np.asarray(rhs), np.asarray(fid)


def _solve_ls(J, y, n, ridge=1e-9):
    Nm = (J.T @ J).tocsc() + ridge * sp.identity(n, format="csc")
    z = spla.spsolve(Nm, J.T @ y)
    return z, Nm


def _factor_mahal(factors, z, scales):
    r = np.zeros(len(factors))
    for k, f in enumerate(factors):
        e = f.L @ (f.A @ z[f.idx] - f.b) / np.sqrt(scales.get(f.group, 1.0))
        r[k] = float(np.sqrt(e @ e))
    return r


def solve(graph: Graph, cfg, z0=None, log=print):
    n = graph.n
    lin = list(graph.factors)
    groups = sorted(set(f.group for f in lin))
    scales = {g: 1.0 for g in groups}
    z = np.zeros(n) if z0 is None else z0.copy()
    history = []
    CHI2_MED = {1: 0.4549, 2: 1.3863, 3: 2.366}
    max_outer = max(cfg.vce_iters, 1) if cfg.vce else 1
    weights = None
    for outer in range(max_outer):
        factors = lin
        robust = np.array([is_robust(f.group) for f in factors])
        # ---------------- GNC (Geman-McClure) ----------------
        w = np.ones(len(factors))
        J, y, fid = _assemble(factors, n, scales, w)
        z, Nm = _solve_ls(J, y, n)
        if cfg.gnc and robust.any():
            c2 = cfg.gnc_c ** 2
            r = _factor_mahal(factors, z, scales)
            mu = max(2.0 * np.max(r[robust] ** 2) / c2, 1.0)
            for it in range(cfg.gnc_max_iter):
                w = np.ones(len(factors))
                w[robust] = (mu * c2 / (r[robust] ** 2 + mu * c2)) ** 2
                J, y, fid = _assemble(factors, n, scales, w)
                z, Nm = _solve_ls(J, y, n)
                r = _factor_mahal(factors, z, scales)
                if mu <= 1.0:
                    break
                mu = max(mu / cfg.gnc_factor, 1.0)
        weights = w
        if not cfg.vce or outer == max_outer - 1:
            continue
        # ---------------- robust variance-component estimation ----------------
        # sigma_g^2 <- sigma_g^2 * median(r_k^2) / median(chi2_m) / redundancy_fraction_g
        # (median-based => immune to the outlier/VCE shrinkage spiral)
        Ninv = np.linalg.inv(Nm.toarray())
        r = _factor_mahal(factors, z, scales)
        new = {}
        for g in groups:
            if g in ("gauge", "motion"):
                continue
            ks = np.array([k for k, f in enumerate(factors) if f.group == g])
            if len(ks) < 5:
                continue
            m = factors[ks[0]].A.shape[0]
            rows = np.where(np.isin(fid, ks))[0]
            Jg = J[rows].toarray()
            hk = np.einsum("ij,jk,ik->i", Jg, Ninv, Jg)
            wr = w[fid[rows]]
            red_frac = float(np.clip(np.sum(wr * (1 - hk)) / max(np.sum(wr), 1e-9), 0.05, 1.0))
            s2 = float(np.median(r[ks] ** 2)) / CHI2_MED.get(m, m - 2 / 3) / red_frac
            new[g] = float(np.clip(scales[g] * max(s2, 1e-8) ** 0.8, 1e-6, 1e6))
        change = max([abs(np.log(new[g] / scales[g])) for g in new] + [0.0])
        scales.update(new)
        history.append(dict(outer=outer, scales=dict(scales)))
        if cfg.verbose:
            log("   [VCE %d] " % outer + "  ".join(f"{g}:{np.sqrt(v):.2f}" for g, v in scales.items()
                                                  if g not in ("gauge", "motion")))
        if change < 0.03:
            # converged: one last solve with the final scales happens next pass
            max_outer = outer + 2
    # ---------------- final covariance ----------------
    Ninv = np.linalg.inv(Nm.toarray())
    N = graph.N
    C = Ninv[:2 * N, :2 * N]

    def rel_sd(Cb):
        m = Cb.mean(axis=1)
        v = np.diag(Cb) - 2 * m + Cb.mean()
        return np.sqrt(np.maximum(v, 0))
    sd_p = rel_sd(C[:N, :N]); sd_q = rel_sd(C[N:, N:])
    inlier = {}
    r = _factor_mahal(factors, z, scales)
    for k, f in enumerate(factors):
        if is_robust(f.group):
            inlier.setdefault(f.group, []).append(bool(weights[k] > 0.5))
    return dict(z=z, p=z[:N], q=z[N:2 * N], sd_p=sd_p, sd_q=sd_q,
                cov_pq=C, scales=scales, weights=weights, factors=factors,
                inlier_rate={g: float(np.mean(v)) for g, v in inlier.items()},
                n_factors={g: len(v) for g, v in inlier.items()}, history=history,
                mahal=r)
