"""
Phase 3 joint [RV, dLW] model and log-likelihood for HD 297396.

One likelihood over the stacked vector:
    per-epoch offsets for both series (pre/post HARPS fibre upgrade),
    per-programme jitter,
    error-scaling beta on the formal errors,
    a GP with hyperparameters SHARED between the two series and SEPARATE
    amplitudes (the Lillo-Box et al. 2021 structure), in two kernels:
      "m32" : Matern-3/2, timescale log-uniform 50-5000 d
      "qp"  : quasi-periodic, eta3 = P_rot with the Phase-2 25-40 d prior
    and two couplings:
      "shared_hypers" : independent GPs, shared hyperparameters (reference)
      "shared_latent" : one latent GP, rank-1 cross-covariance (adversarial)
    Gaussian or multivariate Student-t (per series) likelihood.

Everything is exact O(N^3) linear algebra; N ~ 106 nights per series, so the
Cholesky is microseconds and there is no reason to approximate the kernel
(which is what celerite2 would force on both Matern-3/2 and the QP form).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import gammaln

LN2PI = np.log(2 * np.pi)


# --------------------------------------------------------------- keplerian --
def kepler_E(M, e, tol=1e-12, itmax=64):
    E = M + e * np.sin(M)
    for _ in range(itmax):
        d = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= d
        if np.max(np.abs(d)) < tol:
            break
    return E


def rv_signal(t, P, K, phase, e=0.0, w=0.0, tref=0.0):
    """Keplerian RV; e in [0,1), w in radians."""
    if e < 1e-10:
        return K * np.cos(2 * np.pi * ((t - tref) / P - phase))
    e = min(e, 0.95)
    M = 2 * np.pi * ((t - tref) / P - phase)
    E = kepler_E(np.mod(M, 2 * np.pi), e)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return K * (np.cos(nu + w) + e * np.cos(w))


# ------------------------------------------------------------------ kernels --
def k_m32(tau, l):
    x = np.sqrt(3.0) * tau / l
    return (1.0 + x) * np.exp(-x)


def k_qp(tau, eta2, eta3, eta4):
    return np.exp(-0.5 * (tau / eta2) ** 2
                  - 0.5 * (np.sin(np.pi * tau / eta3) / eta4) ** 2)


# ------------------------------------------------------------------- priors --
class U:                       # uniform
    def __init__(s, a, b): s.a, s.b = a, b
    def __call__(s, u): return s.a + (s.b - s.a) * u


class LU:                      # log-uniform
    def __init__(s, a, b): s.la, s.lb = np.log(a), np.log(b)
    def __call__(s, u): return np.exp(s.la + (s.lb - s.la) * u)


class BETA:                    # Kipping (2013) eccentricity prior
    def __init__(s, a=0.867, b=3.03):
        from scipy.stats import beta
        s.d = beta(a, b)
    def __call__(s, u): return float(s.d.ppf(u))


class FIXED:
    def __init__(s, v): s.v = v
    def __call__(s, u): return s.v


# -------------------------------------------------------------------- model --
class Joint:
    def __init__(self, d, n_planets=1, ecc=False, gp="m32",
                 coupling="shared_hypers", like="gauss",
                 toi=None, gp_off=False, e_prior="uniform", p1_prior=(1.05, 10.0),
                 p2_prior=(10.0, 1000.0), prot_prior=(25.0, 40.0), seeing=None):
        self.d = d
        self.n_planets, self.ecc, self.gp = n_planets, ecc, gp
        self.coupling, self.like = coupling, like
        self.e_prior = e_prior
        self.seeing = seeing
        if seeing is not None:
            s = np.asarray(seeing["s"], float)
            self.see_ok = np.isfinite(s)
            self.see_s = np.where(self.see_ok, s, 0.0)
            self.see_mode = seeing["mode"]
            self.see_ref = float(seeing.get("ref", 0.90))    # arcsec, sample median
            self.see_s0 = float(seeing.get("s0", 1.00))      # arcsec, HARPS fibre aperture
            self.see_thr = float(seeing.get("thr", 1.50))    # arcsec, Phase-0 threshold
        self.toi = toi                       # (P, T0) fixed, K free; or None
        self.gp_off = gp_off                 # pin GP amplitudes to zero
        self.tref = float(np.median(d["t"]))
        self.p1_window, self.p2_window = tuple(p1_prior), tuple(p2_prior)
        tt = d["t"]
        self.tau = np.abs(tt[:, None] - tt[None, :])
        self.Erv = [(d["epoch"] == k) for k in range(d["n_epoch"])]
        self.Gid = d["gidx"]
        self.n = d["n"]

        p = []
        if n_planets >= 1:
            p.append(("P1", LU(*p1_prior)))
            p.append(("K1", U(0.0, 30.0)))
            p.append(("ph1", U(0.0, 1.0)))
            if ecc:
                # uniform prior on e (conservative: gives the eccentric model
                # a fair shot) and on omega
                ep = BETA() if e_prior == "kipping" else U(0.0, 0.9)
                p += [("ecc", ep), ("omega", U(0.0, 2 * np.pi))]
        if n_planets >= 2:
            p += [("P2", LU(*p2_prior)), ("K2", U(0.0, 30.0)), ("ph2", U(0.0, 1.0))]
        if toi is not None:
            p.append(("K_toi", U(0.0, 30.0)))
        for k in range(d["n_epoch"]):
            p.append((f"g_rv{k}", U(-30.0, 30.0)))
        for k in range(d["n_epoch"]):
            p.append((f"g_dlw{k}", U(-60.0, 60.0)))
        p += [("beta_rv", LU(0.5, 5.0)), ("beta_dlw", LU(0.5, 5.0))]
        for k in range(d["n_group"]):
            p.append((f"s_rv{k}", U(0.0, 15.0)))
        for k in range(d["n_group"]):
            p.append((f"s_dlw{k}", U(0.0, 40.0)))
        if seeing is not None:
            # Seeing-dependent RV variance. The predictor is DIMM seeing from the
            # ESO archive headers -- instrumental metadata, established in Phase 0
            # independently of any residual of this model.
            if self.see_mode == "power":
                p += [("gam_see", U(0.0, 15.0)), ("q_see", U(0.0, 10.0))]
            elif self.see_mode == "hinge":
                p += [("gam_see", U(0.0, 40.0))]
            elif self.see_mode == "thresh":
                p += [("s_bad", U(0.0, 40.0))]
            else:
                raise ValueError(self.see_mode)
            # The nights with no DIMM value get their own free jitter rather than
            # an imputed seeing: imputing the median would quietly assert they
            # were good nights.
            p += [("s_nodimm", U(0.0, 15.0))]
        if gp is not None and not gp_off:
            if gp == "m32":
                p += [("gp_l", LU(50.0, 5000.0))]
            elif gp == "qp":
                p += [("eta3", U(*prot_prior)), ("eta2_u", U(0.0, 1.0)),
                      ("eta4", LU(0.05, 5.0))]
            p += [("A_rv", U(0.0, 20.0)), ("A_dlw", U(0.0, 60.0))]
        if like == "studentt":
            p.append(("nu", LU(2.0, 50.0)))
        self.pnames = [x[0] for x in p]
        self.priors = [x[1] for x in p]
        self.ndim = len(p)
        self.ix = {n: i for i, n in enumerate(self.pnames)}

    # ---- dynesty prior transform
    def ptform(self, u):
        v = np.empty_like(u)
        for i, pr in enumerate(self.priors):
            v[i] = pr(u[i])
        if self.gp == "qp" and not self.gp_off:
            # eta2 log-uniform between eta3 and 5000 d  (evolution timescale
            # must exceed the rotation period for the kernel to mean anything)
            e3 = v[self.ix["eta3"]]
            v[self.ix["eta2_u"]] = np.exp(np.log(e3) +
                                          (np.log(5000.0) - np.log(e3)) * u[self.ix["eta2_u"]])
        return v

    # ---- model pieces
    def _rv_model(self, th):
        d, x = self.d, self.ix
        m = np.zeros(self.n)
        if self.n_planets >= 1:
            m = rv_signal(d["t"], th[x["P1"]], th[x["K1"]], th[x["ph1"]],
                          th[x["ecc"]] if self.ecc else 0.0,
                          th[x["omega"]] if self.ecc else 0.0, self.tref)
        if self.n_planets >= 2:
            m = m + rv_signal(d["t"], th[x["P2"]], th[x["K2"]], th[x["ph2"]],
                              tref=self.tref)
        if self.toi is not None:
            Pt, T0t = self.toi
            m = m + th[x["K_toi"]] * np.cos(2 * np.pi * (d["t"] - T0t) / Pt)
        for k, msk in enumerate(self.Erv):
            m = m + th[x[f"g_rv{k}"]] * msk
        return m

    def _dlw_model(self, th):
        m = np.zeros(self.n)
        for k, msk in enumerate(self.Erv):
            m = m + th[self.ix[f"g_dlw{k}"]] * msk
        return m

    def _diag(self, th, which):
        d, x = self.d, self.ix
        e = d["erv"] if which == "rv" else d["edlw"]
        b = th[x["beta_rv"]] if which == "rv" else th[x["beta_dlw"]]
        s = np.array([th[x[f"s_{which}{k}"]] for k in range(d["n_group"])])
        var = (b * e) ** 2 + s[self.Gid] ** 2
        if self.seeing is not None and which == "rv":
            add = np.zeros_like(var)
            if self.see_mode == "power":
                g, q = th[x["gam_see"]], th[x["q_see"]]
                add[self.see_ok] = (g * (self.see_s[self.see_ok] / self.see_ref) ** q) ** 2
            elif self.see_mode == "hinge":
                g = th[x["gam_see"]]
                ex = np.clip(self.see_s - self.see_s0, 0.0, None)
                add[self.see_ok] = (g * ex[self.see_ok]) ** 2
            else:
                add[self.see_ok & (self.see_s > self.see_thr)] = th[x["s_bad"]] ** 2
            add[~self.see_ok] = th[x["s_nodimm"]] ** 2
            var = var + add
        return var

    def _kbase(self, th):
        x = self.ix
        if self.gp == "m32":
            return k_m32(self.tau, th[x["gp_l"]])
        return k_qp(self.tau, th[x["eta2_u"]], th[x["eta3"]], th[x["eta4"]])

    # ---- log-likelihood
    def loglike(self, th):
        x = self.ix
        r_rv = self.d["rv"] - self._rv_model(th)
        r_dlw = self.d["dlw"] - self._dlw_model(th)
        d_rv, d_dlw = self._diag(th, "rv"), self._diag(th, "dlw")

        if self.gp is None or self.gp_off:
            Arv = Adlw = 0.0
            Kb = None
        else:
            Arv, Adlw = th[x["A_rv"]], th[x["A_dlw"]]
            Kb = self._kbase(th)

        nu = th[x["nu"]] if self.like == "studentt" else None

        if self.coupling == "shared_latent" and Kb is not None:
            n = self.n
            S = np.empty((2 * n, 2 * n))
            S[:n, :n] = Arv ** 2 * Kb
            S[n:, n:] = Adlw ** 2 * Kb
            S[:n, n:] = Arv * Adlw * Kb
            S[n:, :n] = S[:n, n:].T
            S[np.diag_indices(2 * n)] += np.concatenate([d_rv, d_dlw])
            return _mvn(np.concatenate([r_rv, r_dlw]), S, nu)

        out = 0.0
        for r, dg, A in ((r_rv, d_rv, Arv), (r_dlw, d_dlw, Adlw)):
            S = (A ** 2) * Kb if Kb is not None else np.zeros((self.n, self.n))
            S = S + np.diag(dg)
            out += _mvn(r, S, nu)
        return out


def _mvn(r, S, nu=None):
    """Multivariate normal (nu=None) or multivariate Student-t log-density."""
    try:
        c, low = cho_factor(S, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -1e300
    logdet = 2.0 * np.sum(np.log(np.diag(c)))
    q = float(r @ cho_solve((c, low), r, check_finite=False))
    n = r.size
    if nu is None:
        return -0.5 * (q + logdet + n * LN2PI)
    return (gammaln(0.5 * (nu + n)) - gammaln(0.5 * nu) - 0.5 * n * np.log(nu * np.pi)
            - 0.5 * logdet - 0.5 * (nu + n) * np.log1p(q / nu))
