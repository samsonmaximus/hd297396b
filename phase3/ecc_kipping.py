"""Re-run every eccentric fit with the Kipping (2013) Beta(0.867,3.03)
eccentricity prior. The uniform e prior up to 0.9 let the Student-t + QP
combination find an unphysical e=0.885 / K=21.5 m/s periastron-spike mode;
at P = 4.27 d around a 0.76 Msun K dwarf, tidal circularisation makes that
region of parameter space not a live hypothesis."""
import json, os, numpy as np, itertools
import p3data, p3model, p3run, report

rows, lad = [], {}
fj, fl = "out/stability_kipping.json", "out/ladder_kipping.json"
if os.path.exists(fj): rows = json.load(open(fj))
if os.path.exists(fl): lad = json.load(open(fl))
done = {(r["pipeline"], r["kernel"], r["noise"]) for r in rows}

for pl, kern, noise in itertools.product(("drs", "serval"), ("white", "m32", "qp"),
                                         ("clip", "studentt")):
    if (pl, kern, noise) in done: continue
    d = p3data.build(pl, clip=(4.0 if noise == "clip" else 0.0))
    m = p3model.Joint(d, n_planets=1, ecc=True, e_prior="kipping",
                      gp=(None if kern == "white" else kern),
                      like=("gauss" if noise == "clip" else "studentt"),
                      p1_prior=p3run.P1_NARROW)
    r = p3run.run(m, f"kip_{pl}_{kern}_ecc_{noise}", nlive=500, walks=25, nproc=2)
    s = p3run.summarize(r)
    rows.append(dict(pipeline=pl, kernel=kern, ecc=True, noise=noise, n=d["n"],
                     K=s["K1"], sigK=0.5*(s["K1"][2]-s["K1"][0]), P=s["P1"][1],
                     e=s["ecc"], e95=float(np.percentile(
                         r["samples"][:, m.ix["ecc"]], 95)),
                     ndim=m.ndim, logz=r["logz"], wall=r["wall"]))
    json.dump(rows, open(fj, "w"), indent=1)
    print(f"{pl:>7} {kern:>5} ecc {noise:>9}  K={s['K1'][1]:6.2f} "
          f"+{s['K1'][2]-s['K1'][1]:.2f}/-{s['K1'][1]-s['K1'][0]:.2f}  "
          f"e={s['ecc'][1]:.3f} (95% < {rows[-1]['e95']:.3f})  lnZ={r['logz']:.2f}", flush=True)

# ladder M2 with the same prior, for a fair circular-vs-eccentric comparison
for kern in ("m32", "qp"):
    tag = f"kip_{kern}_1p_serval"
    if tag in lad: continue
    d = p3data.build("serval", clip=4.0)
    m = p3model.Joint(d, n_planets=1, ecc=True, e_prior="kipping", gp=kern,
                      p1_prior=p3run.P1_NARROW)
    r = p3run.run(m, tag, nlive=750, walks=30, nproc=2)
    s = p3run.summarize(r)
    lad[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], summary=s, ndim=m.ndim)
    json.dump(lad, open(fl, "w"), indent=1)
    L = json.load(open("out/ladder.json"))
    z0 = L[f"{kern}_0p_serval"]["logz"]; zc = L[f"{kern}_1p1c_serval"]["logz"]
    print(f"ladder {kern} 1p(Kipping): lnZ={r['logz']:.2f}  dlnZ vs 0p={r['logz']-z0:.2f}"
          f"  vs 1p1c={r['logz']-zc:+.2f}  e={s['ecc'][1]:.3f}", flush=True)
