"""Control: how reproducible is a single eccentric cell? The uniform-e and
Kipping-prior runs of drs/m32/ecc/clip differ by 8.4 in lnZ, which a prior
change that concentrates mass on the well-fitting region cannot produce.
Either the sampler is unreliable here or one run is stuck."""
import numpy as np, json, p3data, p3model, p3run
d = p3data.build("drs", clip=4.0)
rows = []
for ep in ("kipping", "uniform"):
    for seed, nlive, walks in ((7, 1000, 50), (8, 1000, 50)):
        m = p3model.Joint(d, n_planets=1, ecc=True, e_prior=ep, gp="m32",
                          p1_prior=p3run.P1_NARROW)
        r = p3run.run(m, f"ectl_{ep}_s{seed}", nlive=nlive, walks=walks, seed=seed, nproc=2)
        s = p3run.summarize(r)
        rows.append(dict(e_prior=ep, seed=seed, nlive=nlive, logz=r["logz"],
                         logzerr=r["logzerr"], K=s["K1"], e=s["ecc"],
                         loglmax=r["logl_max"]))
        json.dump(rows, open("out/ecc_control.json", "w"), indent=1)
        print(f"{ep:>8} seed={seed} nlive={nlive}: lnZ={r['logz']:8.2f} "
              f"lnLmax={r['logl_max']:8.2f} K={s['K1'][1]:5.2f} "
              f"[{s['K1'][0]:.2f},{s['K1'][2]:.2f}] e={s['ecc'][1]:.3f}", flush=True)
