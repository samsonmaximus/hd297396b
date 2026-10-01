# v15: harmonic budget of a rotation period at 9*P0 = 38.415 d.
# If P_rot = 9*P0, the planet-like signal would be the ninth harmonic of a strictly periodic
# rotational signal. Fit, in the residuals of the planet fit (white-noise frame, per-label
# offsets and jitters refitted), a sinusoid at every other harmonic k/(9 P0), k = 1..11, and report
# K and its 2-sigma upper limit. Also the variance fraction the quasi-periodic kernel of
# fwd_qp_3050.py puts into harmonic n: 2 I_n(kappa) exp(-kappa), kappa = 1/(4 w^2).
import numpy as np, json
from scipy.special import ive
exec(open('split.py').read().split('subsets={')[0])
out = {}
for name, df in [('104', S), ('103', S[np.abs(S.t - 2454922.53) > 0.2])]:
    t = df.t.values
    X = np.c_[np.cos(2*np.pi*t/P0), np.sin(2*np.pi*t/P0)]
    jj, _, idx, L, labs = jitter_fit(df, design=X); w = 1/(df.e.values**2 + jj[idx]**2)
    XX = np.hstack([L, X]); p = np.linalg.solve(XX.T@(XX*w[:, None]), XX.T@(w*df.y.values))
    R = df.copy(); R['y'] = df.y.values - X@p[-2:]
    jn, _, iR, _, _ = jitter_fit(R); wR = 1/(R.e.values**2 + jn[iR]**2)
    rows = []
    for k in range(1, 12):
        f = k/(9*P0)
        A = np.hstack([L, np.c_[np.cos(2*np.pi*t*f), np.sin(2*np.pi*t*f)]])
        M = A.T@(A*wR[:, None]); c = np.linalg.solve(M, A.T@(wR*R.y.values)); C = np.linalg.inv(M)
        a, b = c[-2:]; K = float(np.hypot(a, b)); sK = float(np.sqrt(np.diag(C)[-2:].mean()))
        # 2-sigma upper limit on K: 95th percentile of |N((a,b), C)| by Monte Carlo
        rng = np.random.default_rng(k); draws = rng.multivariate_normal([a, b], C[-2:, -2:], 20000)
        K95 = float(np.percentile(np.hypot(draws[:, 0], draws[:, 1]), 95))
        dchi = float(periodogram(R, np.array([f]), jit=jn)[0][0]) if k != 9 else float('nan')
        rows.append(dict(k=k, P=float(1/f), K=K, sK=sK, K95=K95, dchi2=dchi))
        print(f'{name} k={k:2d} P={1/f:8.3f} d  K={K:5.2f} +- {sK:4.2f}  K95={K95:5.2f}  dchi2={dchi:5.2f}')
    others = [r for r in rows if r['k'] != 9]
    out[name] = dict(rows=rows, max_K95_others=max(r['K95'] for r in others), max_dchi2_others=max(r['dchi2'] for r in others),
                     K_planet=float(np.hypot(*p[-2:])))
    print(name, 'max K95 over k != 9:', round(out[name]['max_K95_others'], 2), ' max dchi2:', round(out[name]['max_dchi2_others'], 2), ' K at P0:', round(out[name]['K_planet'], 2))
frac = {}
for wq in [0.15, 0.2, 0.25, 0.3, 0.4, 0.8]:
    kap = 1/(4*wq*wq); fn = lambda n: 2*ive(n, kap)            # ive = I_n(kappa) exp(-kappa)
    f9 = float(fn(9)); beyond3 = float(1 - ive(0, kap) - sum(fn(n) for n in (1, 2, 3)))
    frac[str(wq)] = dict(f9=f9, beyond3=beyond3, K9_at_6ms=float(np.sqrt(2*f9)*6.0), rms_for_K9_5p5=float(5.5/np.sqrt(2*f9)) if f9 > 0 else None)
    print(f'w={wq}: variance in 9th harmonic {f9:.2e}, beyond 3rd {beyond3:.3f}; K9 at 6 m/s rms {np.sqrt(2*f9)*6:.2f}; rms needed for K9=5.5: {5.5/np.sqrt(2*f9):.0f} m/s')
out['kernel'] = frac
json.dump(out, open('comb9.json', 'w'), indent=1)
