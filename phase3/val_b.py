"""
Validation B: injection-recovery on simulated data.

Simulate on the REAL timestamps with the REAL formal errors: a Matern-3/2 GP
draw for RV and an independent draw with the same timescale for dLW (exactly
the model's shared-hyperparameter structure), per-group jitter, per-epoch
offsets, plus a known Keplerian. Then fit with the full Phase 3 model and look
at the pull (K_fit - K_true)/sigma_K over independent realisations.
"""
import numpy as np, json, sys
import p3data, p3model, p3run, p3mcmc

TRUTH = dict(l=400.0, A_rv=3.5, A_dlw=9.0, beta=1.15,
             s_rv=[3.0, 5.0, 3.0, 2.0, 3.5], s_dlw=[5.0, 6.0, 4.0, 8.0, 5.0],
             g_rv=[0.0, 2.5], g_dlw=[0.0, -4.0], P=4.26836, ph=0.37)

def simulate(d, K_true, seed):
    rng = np.random.default_rng(seed)
    n = d["n"]
    Kb = p3model.k_m32(np.abs(d["t"][:, None] - d["t"][None, :]), TRUTH["l"])
    L = np.linalg.cholesky(Kb + 1e-8 * np.eye(n))
    gp_rv = TRUTH["A_rv"] * (L @ rng.standard_normal(n))
    gp_dlw = TRUTH["A_dlw"] * (L @ rng.standard_normal(n))
    s_rv = np.array(TRUTH["s_rv"])[d["gidx"]]
    s_dlw = np.array(TRUTH["s_dlw"])[d["gidx"]]
    wn_rv = rng.normal(0, np.hypot(TRUTH["beta"] * d["erv"], s_rv))
    wn_dlw = rng.normal(0, np.hypot(TRUTH["beta"] * d["edlw"], s_dlw))
    off_rv = np.array(TRUTH["g_rv"])[d["epoch"]]
    off_dlw = np.array(TRUTH["g_dlw"])[d["epoch"]]
    tref = float(np.median(d["t"]))
    kep = p3model.rv_signal(d["t"], TRUTH["P"], K_true, TRUTH["ph"], tref=tref)
    sim = dict(d)
    sim["rv"] = kep + gp_rv + wn_rv + off_rv
    sim["dlw"] = gp_dlw + wn_dlw + off_dlw
    sim["rv"] -= np.median(sim["rv"]); sim["dlw"] -= np.median(sim["dlw"])
    return sim

def one(case):
    K_true, seed = case
    d0 = p3data.build("serval", clip=4.0)
    if True:
        sim = simulate(d0, K_true, seed)
        m = p3model.Joint(sim, n_planets=1, gp="m32", p1_prior=p3run.P1_NARROW)
        th, s, lmax = p3mcmc.run_mcmc(m, nstep=4500, burn=2000, seed=seed + 7)
        i = m.ix
        K = th[:, i["K1"]]; P = th[:, i["P1"]]; L = th[:, i["gp_l"]]; A = th[:, i["A_rv"]]
        q = np.percentile(K, [16, 50, 84])
        sig = 0.5 * (q[2] - q[0])
        row = dict(K_true=K_true, seed=seed, K=[float(x) for x in q],
                   pull=float((q[1] - K_true) / sig) if sig > 0 else None,
                   K95=float(np.percentile(K, 95)),
                   P=float(np.median(P)), gp_l=float(np.median(L)),
                   A_rv=float(np.median(A)), acc=float(np.mean(s.acceptance_fraction)))
        print(f"K_true={K_true:5.2f} seed={seed:3d}  K={q[1]:5.2f} +{q[2]-q[1]:.2f}/-{q[1]-q[0]:.2f}"
              f"  pull={row['pull'] if row['pull'] is None else round(row['pull'],2)}"
              f"  K95={row['K95']:.2f}  gp_l={row['gp_l']:.0f}  A_rv={row['A_rv']:.2f}"
              f"  acc={row['acc']:.2f}", flush=True)
        return row

if __name__ == "__main__":
    from multiprocessing import Pool
    cases = [(5.96, s) for s in range(5)] + [(0.0, 100), (0.0, 101), (2.5, 200), (2.5, 201)]
    with Pool(2) as p:
        rows = p.map(one, cases)
    json.dump(rows, open("out/val_b.json", "w"), indent=1)
    det = [r for r in rows if r["K_true"] > 0]
    pulls = np.array([r["pull"] for r in det])
    print(f"\npull mean={pulls.mean():+.2f}  sd={pulls.std(ddof=1):.2f}  (want ~0 +- 1)")
