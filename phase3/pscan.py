"""
Profile-likelihood period scan under the GP noise model.

Justifies the narrow period windows used for the evidence runs: if no period
outside the window comes close in profile likelihood, restricting the prior and
correcting analytically for the search volume is exact to the precision quoted.
"""
import numpy as np, json
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import p3data, p3model

def gls_scan(t, y, e, epoch, Sig, periods):
    c = cho_factor(Sig, lower=True)
    cols0 = [(epoch == k).astype(float) for k in range(epoch.max() + 1)]
    X0 = np.vstack(cols0).T
    def prof(X):
        Si = cho_solve(c, X)
        A = X.T @ Si
        b = np.linalg.solve(A, Si.T @ y)
        r = y - X @ b
        return -0.5 * float(r @ cho_solve(c, r))
    base = prof(X0)
    out = np.empty(len(periods))
    for i, P in enumerate(periods):
        ph = 2 * np.pi * t / P
        out[i] = prof(np.vstack([np.sin(ph), np.cos(ph)] + cols0).T) - base
    return out

def fit_0p(d, series="rv"):
    """ML fit of the no-planet model (GP + per-epoch offsets) for one series."""
    y = d[series]; e = d["e" + series]
    m = p3model.Joint(d, n_planets=1, gp="m32")
    tau = m.tau
    def nll(p):
        ll, lA, lb, ls = p
        if not (np.log(50) < ll < np.log(5000) and -5 < lA < 4 and
                np.log(.5) < lb < np.log(5) and -5 < ls < 4): return 1e9
        S = np.exp(2*lA) * p3model.k_m32(tau, np.exp(ll)) + np.diag((np.exp(lb)*e)**2 + np.exp(2*ls))
        cols = [(d["epoch"] == k).astype(float) for k in range(d["epoch"].max()+1)]
        X = np.vstack(cols).T
        try: c = cho_factor(S, lower=True)
        except Exception: return 1e9
        Si = cho_solve(c, X); b = np.linalg.solve(X.T @ Si, Si.T @ y); r = y - X @ b
        return 0.5*(float(r @ cho_solve(c, r)) + 2*np.sum(np.log(np.diag(c[0]))))
    best = None
    for l0 in (np.log(100), np.log(500), np.log(2000)):
        r = minimize(nll, [l0, np.log(3), 0.0, np.log(3)], method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-4))
        if best is None or r.fun < best.fun: best = r
    ll, lA, lb, ls = best.x
    S = np.exp(2*lA) * p3model.k_m32(tau, np.exp(ll)) + np.diag((np.exp(lb)*e)**2 + np.exp(2*ls))
    return S, dict(l=float(np.exp(ll)), A=float(np.exp(lA)), beta=float(np.exp(lb)),
                   jit=float(np.exp(ls)), nll=float(best.fun))

def _main():
  res = {}
  for pl in ("drs", "serval"):
      d = p3data.build(pl, clip=4.0)
      S, hp = fit_0p(d)
      T = d["t"].max() - d["t"].min()
      f = np.arange(1/10.0, 1/1.05, 1.0/(20*T))
      dl = gls_scan(d["t"], d["rv"], d["erv"], d["epoch"], S, 1/f)
      P = 1/f
      i = np.argmax(dl)
      inwin = (P >= 4.20) & (P <= 4.34)
      j = np.argmax(np.where(inwin, dl, -np.inf))
      k = np.argmax(np.where(~inwin, dl, -np.inf))
      res[pl] = dict(hyper=hp, best_P=float(P[i]), best_dlnl=float(dl[i]),
                     win_best_P=float(P[j]), win_best_dlnl=float(dl[j]),
                     out_best_P=float(P[k]), out_best_dlnl=float(dl[k]), ngrid=len(P))
      print(f"{pl}: GP0p l={hp['l']:.0f}d A={hp['A']:.2f} beta={hp['beta']:.2f} jit={hp['jit']:.2f}")
      print(f"   global best P={P[i]:.5f} dlnL={dl[i]:.2f} | in-window best P={P[j]:.5f} "
            f"dlnL={dl[j]:.2f} | best OUTSIDE window P={P[k]:.5f} dlnL={dl[k]:.2f}")
      # top distinct peaks outside window
      o = np.argsort(np.where(~inwin, dl, -np.inf))[::-1]
      seen=[]
      for idx in o:
          if all(abs(np.log(P[idx])-np.log(s))>0.01 for s in seen):
              seen.append(P[idx]); print(f"      out-of-window peak P={P[idx]:9.4f} dlnL={dl[idx]:6.2f}")
          if len(seen)>=5: break
  json.dump(res, open("out/pscan.json","w"), indent=1)

if __name__ == "__main__":
    _main()

# ---- second-signal scan: profile likelihood over P2 after removing the 4.268 d
# Keplerian under the same GP noise model. Justifies the P2 prior window.
