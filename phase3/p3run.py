"""Nested-sampling driver for the Phase 3 model ladder."""
import os, json, pickle, time
import numpy as np
import dynesty
from dynesty import utils as dyfunc
import p3data, p3model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# Narrow sampling windows (see p3_prior_note): the evidence is computed with a
# narrow log-uniform period prior and corrected analytically to the wide search
# prior, after a profile-likelihood scan shows no competing period mode.
P1_NARROW = (4.20, 4.34)
P1_WIDE   = (1.05, 10.0)
P2_NARROW = (150.0, 260.0)
P2_WIDE   = (10.0, 1000.0)
TOI = (3.0178746, 2460009.249357)


def prior_correction(model):
    """ln(prior mass of the narrow sampling window / prior mass under the wide
    search prior). Uses the model's ACTUAL period window, which differs between
    the candidate and its 1.3013 d sidereal-day alias."""
    c = 0.0
    if model.n_planets >= 1:
        w = getattr(model, "p1_window", P1_NARROW)
        c += np.log(np.log(w[1] / w[0]) / np.log(P1_WIDE[1] / P1_WIDE[0]))
    if model.n_planets >= 2:
        c += np.log(np.log(P2_NARROW[1] / P2_NARROW[0]) / np.log(P2_WIDE[1] / P2_WIDE[0]))
    return c


def run(model, tag, nlive=750, walks=30, seed=42, dlogz=0.1, nproc=2):
    f = os.path.join(OUT, f"ns_{tag}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    t0 = time.time()
    kw = dict(nlive=nlive, sample="rwalk", walks=walks, bound="multi", rstate=np.random.default_rng(seed))
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            s = dynesty.NestedSampler(model.loglike, model.ptform, model.ndim,
                                      pool=pool, queue_size=nproc, **kw)
            s.run_nested(dlogz=dlogz, print_progress=False)
            res = s.results
    else:
        s = dynesty.NestedSampler(model.loglike, model.ptform, model.ndim, **kw)
        s.run_nested(dlogz=dlogz, print_progress=False)
        res = s.results
    w = np.exp(res.logwt - res.logz[-1])
    samples = dyfunc.resample_equal(res.samples, w / w.sum())
    out = dict(tag=tag, names=model.pnames, logz=float(res.logz[-1]),
               logzerr=float(res.logzerr[-1]), pcorr=float(prior_correction(model)),
               samples=samples, ncall=int(np.sum(res.ncall)), niter=int(res.niter),
               wall=time.time() - t0, ndim=model.ndim,
               logl_max=float(res.logl.max()))
    pickle.dump(out, open(f, "wb"))
    return out


def summarize(r):
    q = lambda x: np.percentile(x, [16, 50, 84])
    return {n: [float(v) for v in q(r["samples"][:, i])] for i, n in enumerate(r["names"])}
