# v14: semi-amplitude at P0 under the quasi-periodic GP at its maximum-likelihood covariance (qpgp.json).
# Generalised least squares: four label offsets + sine + cosine at P0, whitened by the QP covariance.
import numpy as np, pandas as pd, json
from scipy.linalg import solve_triangular
G = pd.read_csv('frozen104.csv'); R = json.load(open('qpgp.json'))
def Kqp(r, A, lam, P, w): return A*A*np.exp(-r*r/(2*lam*lam) - np.sin(np.pi*r/P)**2/(2*w*w))
for tag, drop in [('104', False), ('103', True)]:
    g = G[np.abs(G.t - 2454922.53) > 0.2].reset_index(drop=True) if drop else G
    t = g.t.values - 2455000; r = t[:, None] - t[None, :]; li = g.li.values
    Lm = (li[:, None] == np.arange(4)[None, :]).astype(float)
    o = R[tag]; K = Kqp(r, o['Ar'], o['lam'], o['Prot'], o['w'])
    K[np.diag_indices_from(K)] += g.erv.values**2 + np.array(o['jr'])[li]**2
    L = np.linalg.cholesky(K); ph = 2*np.pi*t/4.26837
    X = np.column_stack([Lm, np.sin(ph), np.cos(ph)])
    Xw = solve_triangular(L, X, lower=True); yw = solve_triangular(L, g.rv.values, lower=True)
    C = np.linalg.inv(Xw.T@Xw); b = C@Xw.T@yw; A, B = b[-2:]; Kp = np.hypot(A, B)
    gv = np.array([A, B])/Kp; print(tag, 'K = %.2f +- %.2f m/s' % (Kp, np.sqrt(gv@C[-2:, -2:]@gv)))
