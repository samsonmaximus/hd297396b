"""Verification: run-to-run stability of the headline evidence.

Nested-sampling lnZ from rwalk in ~22 dimensions can be biased low if the walks
under-explore, and dynesty's own logzerr does not capture that. Repeat the two
runs that make the headline number with different seeds and heavier settings,
and quote the observed scatter in dlnZ rather than the internal error.
"""
import json, numpy as np, p3data, p3model, p3run

d = p3data.build("serval", clip=4.0)
rows = []
for seed, nlive, walks in [(11, 1000, 50), (22, 1000, 50), (33, 1500, 60)]:
    z = {}
    for lab, npl in (("0p", 0), ("1p1c", 1)):
        m = p3model.Joint(d, n_planets=npl, gp="m32", p1_prior=p3run.P1_NARROW)
        r = p3run.run(m, f"ver_{lab}_s{seed}_n{nlive}", nlive=nlive, walks=walks,
                      seed=seed, nproc=2)
        z[lab] = r["logz"]; z[lab + "_err"] = r["logzerr"]
        if lab == "1p1c":
            s = p3run.summarize(r); z["K"] = s["K1"]; z["gp_l"] = s["gp_l"]
    z.update(seed=seed, nlive=nlive, walks=walks,
             dlnZ=z["1p1c"] - z["0p"], dlnZ_wide=z["1p1c"] - z["0p"] - 4.2306)
    rows.append(z); json.dump(rows, open("out/verify.json", "w"), indent=1)
    print(f"seed={seed} nlive={nlive} walks={walks}: lnZ0p={z['0p']:.2f} "
          f"lnZ1p1c={z['1p1c']:.2f}  dlnZ={z['dlnZ']:.2f} (wide {z['dlnZ_wide']:.2f})  "
          f"K={z['K'][1]:.2f} [{z['K'][0]:.2f},{z['K'][2]:.2f}]  gp_l={z['gp_l'][1]:.0f}",
          flush=True)
dz = np.array([r["dlnZ"] for r in rows])
print(f"\ndlnZ across {len(dz)} independent runs: mean {dz.mean():.2f}  sd {dz.std(ddof=1):.2f}"
      f"  range {dz.min():.2f}-{dz.max():.2f}")
