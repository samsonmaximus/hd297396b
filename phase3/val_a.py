"""Validation A: pin the GP amplitudes to zero and check we recover Phase 1."""
import numpy as np, p3data, p3model, p3run, json
res = {}
for pl in ("drs", "serval"):
    d = p3data.build(pl, clip=4.0)
    m = p3model.Joint(d, n_planets=1, ecc=False, gp="m32", gp_off=True,
                      p1_prior=p3run.P1_NARROW)
    r = p3run.run(m, f"valA_{pl}", nlive=500, walks=25)
    s = p3run.summarize(r)
    res[pl] = dict(P=s["P1"], K=s["K1"], logz=r["logz"], logzerr=r["logzerr"],
                   wall=r["wall"], ncall=r["ncall"],
                   jit={k: v for k, v in s.items() if k.startswith("s_rv")},
                   beta_rv=s["beta_rv"])
    print(pl, "P=%.5f +%.5f -%.5f" % (s["P1"][1], s["P1"][2]-s["P1"][1], s["P1"][1]-s["P1"][0]),
          " K=%.2f +%.2f -%.2f" % (s["K1"][1], s["K1"][2]-s["K1"][1], s["K1"][1]-s["K1"][0]),
          " lnZ=%.2f+-%.2f" % (r["logz"], r["logzerr"]), " %.0fs" % r["wall"], flush=True)
json.dump(res, open("out/val_a.json", "w"), indent=1)
