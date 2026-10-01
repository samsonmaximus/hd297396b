"""Re-run every eccentric cell at nlive=1000 / walks=50.

The control run showed the nlive=500 settings are not converged for the
eccentric models: drs/m32/ecc/clip with the Kipping prior gave lnZ = -713.23,
K = 4.50 at nlive=500 and lnZ = -707.21, K = 5.60 at nlive=1000. The circular
cells are unaffected (19-22 dimensions, unimodal) but the eccentric ones add
the K-e-omega degeneracy and need the heavier settings."""
import json, os, itertools, numpy as np
import p3data, p3model, p3run

fj = "out/stability_kipping_hi.json"
rows = json.load(open(fj)) if os.path.exists(fj) else []
done = {(r["pipeline"], r["kernel"], r["noise"]) for r in rows}
for pl, kern, noise in itertools.product(("drs", "serval"), ("white", "m32", "qp"),
                                         ("clip", "studentt")):
    if (pl, kern, noise) in done: continue
    d = p3data.build(pl, clip=(4.0 if noise == "clip" else 0.0))
    m = p3model.Joint(d, n_planets=1, ecc=True, e_prior="kipping",
                      gp=(None if kern == "white" else kern),
                      like=("gauss" if noise == "clip" else "studentt"),
                      p1_prior=p3run.P1_NARROW)
    r = p3run.run(m, f"kiphi_{pl}_{kern}_ecc_{noise}", nlive=1000, walks=50, nproc=2)
    s = p3run.summarize(r)
    rows.append(dict(pipeline=pl, kernel=kern, ecc=True, noise=noise, n=d["n"],
                     K=s["K1"], sigK=0.5*(s["K1"][2]-s["K1"][0]), P=s["P1"][1],
                     e=s["ecc"], e95=float(np.percentile(r["samples"][:, m.ix["ecc"]], 95)),
                     ndim=m.ndim, logz=r["logz"], wall=r["wall"]))
    json.dump(rows, open(fj, "w"), indent=1)
    print(f"{pl:>7} {kern:>5} ecc {noise:>9}  K={s['K1'][1]:6.2f} "
          f"+{s['K1'][2]-s['K1'][1]:.2f}/-{s['K1'][1]-s['K1'][0]:.2f}  "
          f"e={s['ecc'][1]:.3f} (95% < {rows[-1]['e95']:.3f})  lnZ={r['logz']:.2f}", flush=True)

# one circular control at the heavier settings, to confirm the circular cells
# at nlive=500 were fine
d = p3data.build("drs", clip=4.0)
m = p3model.Joint(d, n_planets=1, ecc=False, gp="m32", p1_prior=p3run.P1_NARROW)
r = p3run.run(m, "circ_control_drs_m32", nlive=1000, walks=50, seed=9, nproc=2)
s = p3run.summarize(r)
print(f"circular control drs/m32/clip @nlive1000: K={s['K1'][1]:.2f} "
      f"[{s['K1'][0]:.2f},{s['K1'][2]:.2f}]  (nlive=500 table value 5.61)", flush=True)
