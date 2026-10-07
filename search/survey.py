import numpy as np, pandas as pd, sys, json, time
FIBER_BJD = 2457174.5   # HARPS fiber upgrade (June 2015); RVBank treats pre/post separately

def load_star(df, rvcol='RV_mlc_nzp', ecol='e_RV_mlc_nzp', nightly=True, clip=5.0):
    d = df[(df.f_RV == 0)].copy()
    d = d[np.isfinite(d[rvcol]) & np.isfinite(d[ecol]) & (d[ecol] > 0) & (d[rvcol] > -9e5)]
    t, y, e = d.BJD.values, d[rvcol].values, d[ecol].values
    if nightly:
        night = np.floor(t).astype(int)
        out = []
        for n in np.unique(night):
            m = night == n; w = 1/e[m]**2
            out.append((np.sum(w*t[m])/w.sum(), np.sum(w*y[m])/w.sum(), max(1/np.sqrt(w.sum()), e[m].min()/np.sqrt(m.sum()))))
        t, y, e = map(np.array, zip(*out))
    # per-epoch median subtraction + MAD clipping
    pre = t < FIBER_BJD
    r = y.copy()
    for m in (pre, ~pre):
        if m.sum(): r[m] -= np.median(y[m])
    mad = 1.4826*np.median(np.abs(r - np.median(r))) if len(r) > 5 else np.inf
    keep = np.abs(r - np.median(r)) < clip*max(mad, 1e-3) + 3*e   # generous: keep unless wildly off
    return t[keep], y[keep], e[keep]

def nuisance(t):
    pre = (t < FIBER_BJD).astype(float); post = 1 - pre
    cols = [c for c in (pre, post) if c.sum() > 0]
    span = t.max() - t.min()
    cols.append((t - t.mean())/max(span, 1))   # linear trend
    return np.column_stack(cols)

def gls_offsets(t, y, e, fmin=None, fmax=1.0, ofac=10, chunk=4000, N=None):
    """GLS with nuisance basis N (offsets + trend) fitted jointly. Returns freq, power p=(chi2_0-chi2)/chi2_0, best-fit info."""
    w = 1/e**2
    if N is None: N = nuisance(t)
    Wn = N*w[:, None]
    A = N.T @ Wn; Ainv = np.linalg.pinv(A)
    beta0 = Ainv @ (Wn.T @ y)
    r = y - N @ beta0
    chi2_0 = np.sum(w*r*r)
    span = t.max()-t.min()
    if fmin is None: fmin = 1/span
    df = 1/(ofac*span)
    freqs = np.arange(fmin, fmax, df)
    p = np.empty(len(freqs))
    tt = t - t.mean()
    for i in range(0, len(freqs), chunk):
        f = freqs[i:i+chunk]
        ph = 2*np.pi*np.outer(f, tt)
        C, S = np.cos(ph), np.sin(ph)            # (F,N)
        # orthogonalize against nuisance: c' = c - N (N'WN)^-1 N'W c
        Cc = C - (C @ Wn @ Ainv) @ N.T
        Sc = S - (S @ Wn @ Ainv) @ N.T
        cc = np.sum(w*Cc*Cc, 1); ss = np.sum(w*Sc*Sc, 1); cs = np.sum(w*Cc*Sc, 1)
        yc = np.sum(w*r*Cc, 1); ys = np.sum(w*r*Sc, 1)
        det = cc*ss - cs*cs
        det[det <= 0] = np.inf
        chi2red = (ss*yc*yc - 2*cs*yc*ys + cc*ys*ys)/det
        p[i:i+chunk] = chi2red/chi2_0
    return freqs, p, chi2_0, r

def fit_sine(t, y, e, f, N=None):
    if N is None: N = nuisance(t)
    X = np.column_stack([N, np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)])
    w = 1/e**2
    beta = np.linalg.lstsq(X*np.sqrt(w)[:, None], y*np.sqrt(w), rcond=None)[0]
    K = np.hypot(beta[-1], beta[-2]); model = X @ beta
    return K, model, beta

def fap(p, n, nfree, M):
    prob = (1-p)**((n-nfree-2)/2.)
    return 1-(1-prob)**M if M*prob < 1e-3 else 1-(1-prob)**M

def analyze(t, y, e, npeaks=3, fmax=1.0):
    res = []; N = nuisance(t); yy = y.copy()
    span = t.max()-t.min()
    for k in range(npeaks):
        freqs, p, chi2_0, r = gls_offsets(t, yy, e, N=N, fmax=fmax)
        i = np.argmax(p); f = freqs[i]
        K, model, beta = fit_sine(t, yy, e, f, N)
        M = span*(freqs[-1]-freqs[0])
        rms_before = np.sqrt(np.average(r**2, weights=1/e**2))
        resid = yy - model
        rms_after = np.sqrt(np.average((resid - np.average(resid, weights=1/e**2))**2, weights=1/e**2))
        res.append(dict(P=1/f, p=p[i], K=K, FAP=fap(p[i], len(t), N.shape[1], M), rms_before=rms_before, rms_after=rms_after))
        yy = resid + (N @ beta[:N.shape[1]])   # prewhiten: remove sinusoid only
    return res

if __name__ == '__main__':
    rv = pd.read_csv('repos/HARPS_RVBank/ver02/HARPS_RVBank_ver02.csv', low_memory=False)
    targets = pd.read_csv('cat/rvbank_targets.csv')
    sel = targets[(targets.n >= 20) & (targets.span > 100)].target.tolist()
    out = []; t0 = time.time()
    for j, name in enumerate(sel):
        d = rv[rv.target == name]
        t, y, e = load_star(d)
        if len(t) < 15 or (t.max()-t.min()) < 100: continue
        try:
            peaks = analyze(t, y, e)
        except Exception as ex:
            print('ERR', name, ex); continue
        row = dict(target=name, ra=d.ra.iloc[0], dec=d.dec.iloc[0], nobs=len(d), nnights=len(t), span=t.max()-t.min(),
                   med_err=np.median(e), rms=np.std(y - np.where(t < FIBER_BJD, np.median(y[t < FIBER_BJD]) if (t < FIBER_BJD).any() else 0, np.median(y[t >= FIBER_BJD]) if (t >= FIBER_BJD).any() else 0)))
        for k, pk in enumerate(peaks):
            for kk, v in pk.items(): row[f'{kk}{k+1}'] = v
        out.append(row)
        if j % 100 == 0:
            print(j, name, f'{time.time()-t0:.0f}s', flush=True)
            pd.DataFrame(out).to_csv('survey_results.csv', index=False)
    pd.DataFrame(out).to_csv('survey_results.csv', index=False)
    print('done', len(out))
