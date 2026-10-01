"""K-stability table: DRS/SERVAL x white/long-GP/QP-GP x circular/eccentric
x clip/Student-t.  Posteriors only; sampler selectable."""
import os, json, sys, itertools, pickle
import numpy as np
import p3data, p3model, p3run, p3mcmc

CELLS = list(itertools.product(("drs", "serval"), ("white", "m32", "qp"),
                               (False, True), ("clip", "studentt")))

def make(pipeline, kernel, ecc, noise):
    d = p3data.build(pipeline, clip=(4.0 if noise == "clip" else 0.0))
    return p3model.Joint(d, n_planets=1, ecc=ecc,
                         gp=(None if kernel == "white" else kernel),
                         like=("gauss" if noise == "clip" else "studentt"),
                         p1_prior=p3run.P1_NARROW), d

def cell_tag(c):
    pl, k, e, n = c
    return f"stab_{pl}_{k}_{'ecc' if e else 'circ'}_{n}"

def run_cell(c, sampler="ns", nlive=500, walks=25, nproc=2):
    m, d = make(*c)
    tag = cell_tag(c)
    if sampler == "ns":
        r = p3run.run(m, tag, nlive=nlive, walks=walks, nproc=nproc)
        th, extra = r["samples"], dict(logz=r["logz"], logzerr=r["logzerr"], wall=r["wall"])
    else:
        f = f"out/mc_{tag}.pkl"
        if os.path.exists(f):
            th, extra = pickle.load(open(f, "rb"))
        else:
            import time; t0 = time.time()
            th, s, lmax = p3mcmc.run_mcmc(m, nstep=12000, burn=6000, thin=15)
            extra = dict(acc=float(np.mean(s.acceptance_fraction)),
                         tau=float(np.max(s.get_autocorr_time(quiet=True))),
                         wall=time.time() - t0)
            pickle.dump((th, extra), open(f, "wb"))
    K = th[:, m.ix["K1"]]; P = th[:, m.ix["P1"]]
    q = np.percentile(K, [16, 50, 84])
    row = dict(pipeline=c[0], kernel=c[1], ecc=c[2], noise=c[3], n=d["n"],
               K=[float(x) for x in q], sigK=float(0.5*(q[2]-q[0])),
               P=float(np.median(P)), ndim=m.ndim, **extra)
    if c[2]:
        e = th[:, m.ix["ecc"]]
        row["e"] = [float(x) for x in np.percentile(e, [16, 50, 84])]
        row["e95"] = float(np.percentile(e, 95))
    return row

if __name__ == "__main__":
    sampler = sys.argv[1] if len(sys.argv) > 1 else "ns"
    fj = f"out/stability_{sampler}.json"
    rows = json.load(open(fj)) if os.path.exists(fj) else []
    done = {(r["pipeline"], r["kernel"], r["ecc"], r["noise"]) for r in rows}
    for c in CELLS:
        if c in done: continue
        row = run_cell(c, sampler=sampler)
        rows.append(row); json.dump(rows, open(fj, "w"), indent=1)
        print(f"{c[0]:>7} {c[1]:>5} {'ecc' if c[2] else 'circ':>4} {c[3]:>8}  "
              f"n={row['n']:<4} K={row['K'][1]:5.2f} +{row['K'][2]-row['K'][1]:.2f}"
              f"/-{row['K'][1]-row['K'][0]:.2f}  P={row['P']:.5f}  {row.get('wall',0):.0f}s",
              flush=True)
