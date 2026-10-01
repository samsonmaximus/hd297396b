"""emcee driver: posteriors only (no evidence). Used for injection-recovery
and for the K-stability table, where lnZ is not needed and nested sampling
would cost ~50x more for the same marginal posterior on K."""
import numpy as np, emcee, os
from scipy.optimize import minimize

def logpost(u, model):
    if np.any(u <= 0) or np.any(u >= 1):
        return -np.inf
    th = model.ptform(u)
    ll = model.loglike(th)
    return ll if np.isfinite(ll) else -np.inf


def run_mcmc(model, nwalk=None, nstep=4000, burn=1500, seed=1, thin=10, nstart=400):
    """Sample in the unit hypercube, so the prior is flat by construction."""
    rng = np.random.default_rng(seed)
    nd = model.ndim
    nwalk = nwalk or max(4 * nd, 40)
    # seed walkers near the best of nstart random draws, then a local polish
    U = rng.random((nstart, nd))
    lp = np.array([logpost(u, model) for u in U])
    u0 = U[np.argmax(lp)]
    r = minimize(lambda u: -logpost(np.clip(u, 1e-9, 1-1e-9), model), u0,
                 method="Nelder-Mead", options=dict(maxiter=6000, fatol=1e-3))
    u0 = np.clip(r.x, 1e-6, 1 - 1e-6)
    p0 = np.clip(u0 + 0.02 * rng.standard_normal((nwalk, nd)), 1e-9, 1 - 1e-9)
    moves = [(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
    s = emcee.EnsembleSampler(nwalk, nd, logpost, args=(model,), moves=moves)
    s.run_mcmc(p0, nstep, progress=False)
    chain = s.get_chain(discard=burn, thin=thin, flat=True)
    th = np.array([model.ptform(u) for u in chain])
    return th, s, float(-r.fun)
