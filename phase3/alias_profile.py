"""Profile-likelihood period scan under the adopted correlated-noise model.

At each trial period the amplitude, phase and the four instrument offsets are
profiled out analytically -- they enter the model linearly -- in the whitened
frame of the adopted covariance, with the noise hyperparameters held at their
posterior median. The result is 2*Delta lnL between any two periods, which is
what a likelihood-only comparison of the candidate against its sidereal-day
alias would see.
"""
import os, json
import numpy as np
from scipy.linalg import cholesky
import figs_final as F

ADOPT = os.environ.get("ADOPT", "v1_serval_in_studentt_1p1c")
OUTJ = "out/alias_profile.json"
res = {}
for pipe, tag in (("serval", ADOPT), ("drs", ADOPT.replace("serval", "drs"))):
    try:
        d, m, th = F.rebuild(tag)
    except KeyError:
        continue
    t = d["t"]; n = len(t)
    Sig = np.diag(m._diag(th, "rv"))
    if m.gp is not None and not m.gp_off:
        Sig = Sig + th[m.ix["A_rv"]] ** 2 * m._kbase(th)
    L = cholesky(Sig, lower=True); Li = np.linalg.inv(L)
    X0 = np.array([(d["gidx"] == k).astype(float) for k in range(d["n_group"])]).T
    Q, _ = np.linalg.qr(Li @ X0)
    def proj(V):
        Vw = V @ Li.T
        return Vw - (Vw @ Q) @ Q.T
    y = proj(d["rv"][None, :])[0]
    T = t.max() - t.min()
    fr = np.arange(1 / 10.0, 1 / 1.05, 1 / (20 * T))
    out = np.empty(len(fr))
    for a in range(0, len(fr), 3000):
        f = fr[a:a + 3000][:, None]; ph = 2 * np.pi * f * t[None, :]
        C = proj(np.cos(ph)); S = proj(np.sin(ph))
        cc = (C * C).sum(1); ss = (S * S).sum(1); cs = (C * S).sum(1)
        cy = C @ y; sy = S @ y
        det = cc * ss - cs * cs; det[np.abs(det) < 1e-12] = np.inf
        A = (ss * cy - cs * sy) / det; B = (-cs * cy + cc * sy) / det
        out[a:a + 3000] = A * cy + B * sy         # = 2 Delta lnL
    P = 1 / fr
    def peak(p0, w=0.02):
        s = np.abs(P - p0) < w * p0
        i = np.argmax(out[s])
        return float(P[s][i]), float(out[s][i])
    p_c, v_c = peak(4.26836)
    p_a, v_a = peak(1.30131)
    gl = float(P[np.argmax(out)])
    res[pipe] = dict(P_cand=p_c, dchi2_cand=v_c, P_alias=p_a, dchi2_alias=v_a,
                     dlogl=0.5 * (v_c - v_a), global_max_P=gl,
                     global_max=float(out.max()), tag=tag)
    print(f"{pipe}: candidate {p_c:.5f} d (2dlnL={v_c:.2f}), alias {p_a:.5f} d "
          f"(2dlnL={v_a:.2f}) -> dlnL={0.5*(v_c-v_a):.2f}; global max at {gl:.5f} d")
json.dump(res, open(OUTJ, "w"), indent=1)
