"""Step 5.1: detection limits in the whitened frame of the adopted noise model.

Phase 4 computed completeness with a white-noise Delta chi^2 statistic. The
adopted noise model is not white, so the recovery statistic must not be either.
Here the full covariance of Eq. (1) -- GP + per-label jitter + scaled formal
errors, at the posterior median of the adopted run -- is Cholesky-factorised and
both the data and the design matrix are multiplied by L^{-1} before the
least-squares comparison. That is the exact generalised-least-squares statistic
for correlated noise, and it reduces to Phase 4's statistic when the GP
amplitude is zero.

The null is calibrated two ways: parametric draws from the fitted covariance
(primary, since it is the model whose limits we are computing) and
within-instrument shuffles of the residuals (cross-check, non-parametric).
"""
import os, json, time, sys
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
import p3data, p3model, p3run
import figs_final as F

ADOPT = os.environ.get("ADOPT", "v1_serval_in_studentt_1p1c")
OUTF = os.environ.get("OUTF", f"{os.path.dirname(os.path.abspath(__file__))}/out/injrec_gp.npz")
NPH = int(os.environ.get("NPH", 100))

d, m, th = F.rebuild(ADOPT)
t = d["t"]; n = len(t); T = t.max() - t.min()

# --- residuals of the adopted model, and its full RV covariance
off = np.zeros(n)
for k in range(d["n_epoch"]):
    off += th[m.ix[f"g_rv{k}"]] * (d["epoch"] == k)
resid = d["rv"] - m._rv_model(th)
Sig = np.diag(m._diag(th, "rv"))
if m.gp is not None and not m.gp_off:
    Sig = Sig + th[m.ix["A_rv"]] ** 2 * m._kbase(th)
L = cholesky(Sig, lower=True)
Linv = np.linalg.inv(L)

# --- whitened design of the four instrument offsets, projected out analytically
X0 = np.array([(d["gidx"] == k).astype(float) for k in range(d["n_group"])]).T   # n x 4
Xw = Linv @ X0
Q, _ = np.linalg.qr(Xw)                       # orthonormal basis of the offset space


def proj(V):
    """V: m x n in the ORIGINAL frame -> whitened and offset-projected, m x n."""
    Vw = V @ Linv.T
    return Vw - (Vw @ Q) @ Q.T


fr = np.arange(1 / 1000., 0.5, 1 / (8 * T))
nf = len(fr)
C = np.empty((nf, n)); S = np.empty((nf, n))
for a in range(0, nf, 4000):
    f = fr[a:a + 4000][:, None]; ph = 2 * np.pi * f * t[None, :]
    C[a:a + 4000] = proj(np.cos(ph)); S[a:a + 4000] = proj(np.sin(ph))
cc = (C * C).sum(1); ss = (S * S).sum(1); cs = (C * S).sum(1)
det = cc * ss - cs * cs
det[np.abs(det) < 1e-10] = np.inf


def dchi2(Y):
    """Y: m x n in the ORIGINAL frame. Returns nf x m."""
    Yp = proj(Y)
    cy = C @ Yp.T; sy = S @ Yp.T
    A = (ss[:, None] * cy - cs[:, None] * sy) / det[:, None]
    B = (-cs[:, None] * cy + cc[:, None] * sy) / det[:, None]
    return A * cy + B * sy


rng = np.random.default_rng(20260908)
# --- threshold, parametric draws from the fitted covariance (primary)
mx = []
for _ in range(20):
    Y = (L @ rng.standard_normal((n, 100))).T
    mx.append(dchi2(Y).max(0))
mx = np.concatenate(mx)
THR = float(np.percentile(mx, 99))
# --- threshold, within-instrument shuffles (non-parametric cross-check)
mxs = []
for _ in range(10):
    Y = np.tile(resid, (100, 1))
    for k in np.unique(d["gidx"]):
        ii = np.where(d["gidx"] == k)[0]
        for j in range(100):
            Y[j, ii] = rng.permutation(Y[j, ii])
    mxs.append(dchi2(Y).max(0))
mxs = np.concatenate(mxs)
THR_S = float(np.percentile(mxs, 99))
print(f"threshold: parametric {THR:.2f} ({len(mx)} draws), shuffle {THR_S:.2f} "
      f"({len(mxs)} draws)", flush=True)

Pgrid = np.geomspace(2, 1000, 50)
Kgrid = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 10.0])
tol = 1.0 / (2 * T)
comp = np.zeros((len(Kgrid), len(Pgrid)))
t0 = time.time()
for ip, P in enumerate(Pgrid):
    near = np.abs(fr - 1.0 / P) < tol
    phs = rng.uniform(0, 2 * np.pi, NPH)
    base = np.sin(2 * np.pi * t[None, :] / P + phs[:, None])
    for ik, K in enumerate(Kgrid):
        D = dchi2(resid[None, :] + K * base)
        gmax_in = near[D.argmax(0)]
        comp[ik, ip] = float(((D[near].max(0) > THR) & gmax_in).mean())
    if ip % 10 == 0:
        print(f"  P={P:8.2f} d  ({time.time()-t0:.0f}s)", flush=True)

def contour(level):
    out = []
    for ip in range(len(Pgrid)):
        c = comp[:, ip]
        j = np.argmax(c >= level) if (c >= level).any() else None
        if j is None or j == 0:
            out.append(Kgrid[0] if j == 0 else np.nan)
        else:
            out.append(float(np.interp(level, [c[j-1], c[j]], [Kgrid[j-1], Kgrid[j]])))
    return np.array(out)

K95, K50 = contour(0.95), contour(0.50)
np.savez(OUTF, Pgrid=Pgrid, Kgrid=Kgrid, comp=comp, THR=THR, THR_shuffle=THR_S,
         K95=K95, K50=K50, nph=NPH, fr=fr, boot=mx, boot_shuffle=mxs, adopt=ADOPT)
kad = m and th[m.ix["K1"]]
i0 = int(np.argmin(np.abs(Pgrid - th[m.ix["P1"]])))
print(f"median K95 = {np.nanmedian(K95):.2f} m/s over 2-1000 d; "
      f"K95 at P0 = {K95[i0]:.2f}; candidate K = {kad:.2f} "
      f"-> factor {kad/K95[i0]:.2f} above the contour")
