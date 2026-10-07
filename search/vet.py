import numpy as np, pandas as pd, sys
from survey import *

oec = pd.read_csv('cat/oec_planets.csv')
ia = pd.read_csv('cat/confirmed_exoplanets.csv')
ps = pd.read_csv('cat/pscomppars_small.csv')
YEAR = 365.25

def norm(s): return ''.join(ch for ch in str(s).upper() if ch.isalnum())

def sep_arcsec(ra1, de1, ra2, de2):
    d2r = np.pi/180
    return 3600/d2r*np.arccos(np.clip(np.sin(de1*d2r)*np.sin(de2*d2r)+np.cos(de1*d2r)*np.cos(de2*d2r)*np.cos((ra1-ra2)*d2r), -1, 1))

def known_planets(name, ra, dec, radius=90):
    hits = []
    n = norm(name)
    if ra > -999:
        s = sep_arcsec(ra, dec, oec.ra.values, oec.dec.values)
        for _, r in oec[s < radius].iterrows():
            hits.append(('OEC', r.planet, r.period, r.pmass, r.smass, r.lists))
        s = sep_arcsec(ra, dec, ia.ra.values, ia.dec.values)
        for _, r in ia[s < radius].iterrows():
            hits.append(('NEA2020', r.pl_name, r.pl_orbper, r.pl_bmassj, np.nan, 'confirmed'))
    # name matches (OEC has many aliases in xml but we kept system/star names only)
    for _, r in oec[[norm(x).startswith(n) or n.startswith(norm(x)) for x in oec.star.fillna(oec.system)]].iterrows():
        if n and len(n) > 3: hits.append(('OECname', r.planet, r.period, r.pmass, r.smass, r.lists))
    for _, r in ia[[norm(x) == n for x in ia.pl_hostname]].iterrows():
        hits.append(('NEAname', r.pl_name, r.pl_orbper, r.pl_bmassj, np.nan, 'confirmed'))
    for _, r in ps[[norm(x) == n for x in ps.hostname]].iterrows():
        hits.append(('PS2026', r.pl_name, r.pl_orbper, np.nan, r.st_mass, 'confirmed'))
    # dedupe by planet name
    seen = {}; 
    for h in hits: seen.setdefault(h[1], h)
    return list(seen.values())

def period_matches(P, Pk, tol=0.05):
    """True if P matches known period Pk, its 1-yr alias, or 2x/0.5x harmonic."""
    if not np.isfinite(Pk) or Pk <= 0: return False
    cands = [Pk, Pk/2, 2*Pk, Pk/3, 3*Pk]
    for sgn in (1, -1):
        f = 1/Pk + sgn/YEAR
        if f > 0: cands.append(1/f)
        f = 1/Pk + sgn/29.53
        if f > 0: cands.append(1/f)
        f = 1/Pk + sgn  # 1-day alias
        if f > 0: cands.append(1/f)
    return any(abs(P-c)/c < tol for c in cands)

def suspicious_period(P):
    flags = []
    for name, P0, tol in [('1yr', YEAR, 0.06), ('1/2yr', YEAR/2, 0.06), ('1/3yr', YEAR/3, 0.05), ('1/4yr', YEAR/4, 0.04),
                          ('1d', 1.0, 0.03), ('2d', 2.0, 0.02), ('0.5d', 0.5, 0.03), ('moon', 29.53, 0.04), ('moon/2', 14.77, 0.03),
                          ('2yr', 2*YEAR, 0.08), ('3yr', 3*YEAR, 0.08)]:
        if abs(P-P0)/P0 < tol: flags.append(name)
    return flags

def msini_mjup(K, P, M=1.0, ecc=0.0):
    return K/28.4329*(P/YEAR)**(1/3)*M**(2/3)*np.sqrt(1-ecc**2)

def indicator_check(d, t_ref, f, cols=('Halpha', 'NaD1', 'NaD2', 'CRX', 'dLW', 'FWHM_DRS', 'BIS', 'RHKp')):
    """GLS power of each activity indicator at candidate frequency f, its global-peak FAP, and correlation with RV."""
    res = {}
    dd = d[d.f_RV == 0]
    tr, yr, er = load_star(d)
    for c in cols:
        if c not in dd: continue
        x = dd[[c, 'BJD']].copy(); x = x[np.isfinite(x[c]) & (x[c] > -9e4) & (x[c] < 9e4)]
        if len(x) < 15: continue
        # nightly bin, unit errors
        night = np.floor(x.BJD.values).astype(int); t = []; y = []
        for n in np.unique(night):
            m = night == n; t.append(x.BJD.values[m].mean()); y.append(x[c].values[m].mean())
        t = np.array(t); y = np.array(y)
        # clip outliers
        mad = 1.4826*np.median(np.abs(y-np.median(y))) + 1e-12
        m = np.abs(y-np.median(y)) < 5*mad; t, y = t[m], y[m]
        if len(t) < 15: continue
        e = np.full(len(t), np.std(y) if np.std(y) > 0 else 1.0)
        try:
            N = nuisance(t)
            freqs, p, chi2_0, r = gls_offsets(t, y, e, N=N, fmax=1.0)
            span = t.max()-t.min(); M = span*(freqs[-1]-freqs[0])
            i = np.argmin(np.abs(freqs-f))
            # local max within +-1/span of f
            win = np.abs(freqs-f) < 1.5/span
            p_at = p[win].max()
            res[c+'_p'] = p_at
            res[c+'_fap'] = fap(p_at, len(t), N.shape[1], M)
            res[c+'_Pbest'] = 1/freqs[np.argmax(p)]
            res[c+'_fapbest'] = fap(p.max(), len(t), N.shape[1], M)
            # correlation with nightly RV (matched nights)
            ni = np.floor(t).astype(int); nr = np.floor(tr).astype(int)
            common, ia_, ib_ = np.intersect1d(ni, nr, return_indices=True)
            if len(common) > 10:
                rr = yr[ib_] - (nuisance(tr[ib_]) @ np.linalg.lstsq(nuisance(tr[ib_]), yr[ib_], rcond=None)[0])
                res[c+'_r'] = np.corrcoef(rr, y[ia_])[0, 1]
        except Exception as ex:
            res[c+'_err'] = str(ex)[:40]
    return res

def bootstrap_fap(t, y, e, p_obs, ntrial=300, fmax=1.0, seed=1):
    rng = np.random.default_rng(seed); N = nuisance(t)
    # shuffle residuals after nuisance fit, add back nuisance model
    w = 1/e**2; beta = np.linalg.lstsq(N*np.sqrt(w)[:, None], y*np.sqrt(w), rcond=None)[0]
    base = N @ beta; r = y-base
    cnt = 0
    for k in range(ntrial):
        idx = rng.permutation(len(r))
        _, p, _, _ = gls_offsets(t, base + r[idx], e[idx], N=N, fmax=fmax, ofac=5)
        if p.max() >= p_obs: cnt += 1
    return (cnt+1)/(ntrial+1)

def split_consistency(t, y, e, f):
    """Fit K and phase separately to pre- and post-upgrade data."""
    out = {}
    for lab, m in (('pre', t < FIBER_BJD), ('post', t >= FIBER_BJD)):
        if m.sum() < 12 or (t[m].max()-t[m].min()) < 1.5/f: continue
        X = np.column_stack([np.ones(m.sum()), (t[m]-t[m].mean()), np.cos(2*np.pi*f*t[m]), np.sin(2*np.pi*f*t[m])])
        w = 1/e[m]**2
        beta, *_ = np.linalg.lstsq(X*np.sqrt(w)[:, None], y[m]*np.sqrt(w), rcond=None)
        cov = np.linalg.pinv((X*w[:, None]).T @ X)
        K = np.hypot(beta[2], beta[3]); eK = np.sqrt((beta[2]**2*cov[2, 2]+beta[3]**2*cov[3, 3])/K**2) if K > 0 else np.nan
        out[lab] = (m.sum(), K, eK, np.degrees(np.arctan2(beta[3], beta[2])) % 360)
    return out
