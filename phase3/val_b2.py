"""Validation B, re-run with nested sampling after emcee showed non-convergence
in 22 dimensions (acceptance 0.12-0.18, wildly asymmetric marginals)."""
import numpy as np, json, sys
import p3data, p3model, p3run
from val_b import simulate

if __name__ == "__main__":
    d0 = p3data.build("serval", clip=4.0)
    cases = [(5.96, s) for s in range(5)] + [(0.0, 100), (0.0, 101), (2.5, 200), (2.5, 201)]
    rows = []
    fj = "out/val_b2.json"
    import os
    if os.path.exists(fj): rows = json.load(open(fj))
    done = {(r["K_true"], r["seed"]) for r in rows}
    for K_true, seed in cases:
        if (K_true, seed) in done: continue
        sim = simulate(d0, K_true, seed)
        m = p3model.Joint(sim, n_planets=1, gp="m32", p1_prior=p3run.P1_NARROW)
        r = p3run.run(m, f"valB_{K_true:.2f}_{seed}", nlive=600, walks=30, nproc=2)
        i = m.ix
        K = r["samples"][:, i["K1"]]
        q = np.percentile(K, [16, 50, 84]); sig = 0.5*(q[2]-q[0])
        row = dict(K_true=K_true, seed=seed, K=[float(x) for x in q],
                   pull=float((q[1]-K_true)/sig), K95=float(np.percentile(K,95)),
                   P=float(np.median(r["samples"][:, i["P1"]])),
                   gp_l=float(np.median(r["samples"][:, i["gp_l"]])),
                   A_rv=float(np.median(r["samples"][:, i["A_rv"]])),
                   logz=r["logz"], wall=r["wall"])
        rows.append(row); json.dump(rows, open(fj,"w"), indent=1)
        print(f"K_true={K_true:5.2f} seed={seed:3d}  K={q[1]:5.2f} +{q[2]-q[1]:.2f}/-{q[1]-q[0]:.2f}"
              f"  pull={row['pull']:+.2f}  K95={row['K95']:.2f}  gp_l={row['gp_l']:.0f}"
              f"  A_rv={row['A_rv']:.2f}  {r['wall']:.0f}s", flush=True)
    d = [r for r in rows if r["K_true"] > 0]
    p = np.array([r["pull"] for r in d])
    print(f"\npull mean={p.mean():+.2f} sd={p.std(ddof=1):.2f} (want 0 +- 1)")
