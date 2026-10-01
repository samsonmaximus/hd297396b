"""Phase 3 model ladder with the GP on. Evidences via dynesty nested sampling."""
import os, json, sys
import numpy as np
import p3data, p3model, p3run

CLIP = 4.0
ALIAS = (1.295, 1.308)     # 1-sidereal-day alias of P0, found by the period scan

def spec(kernel, model, pipeline="serval"):
    d = p3data.build(pipeline, clip=CLIP)
    gp = None if kernel == "white" else kernel
    kw = dict(gp=gp, p1_prior=p3run.P1_NARROW, p2_prior=p3run.P2_NARROW)
    if model == "0p":      m = p3model.Joint(d, n_planets=0, **kw)
    elif model == "1p1c":  m = p3model.Joint(d, n_planets=1, ecc=False, **kw)
    elif model == "1p":    m = p3model.Joint(d, n_planets=1, ecc=True, **kw)
    elif model == "2p1c2c":m = p3model.Joint(d, n_planets=2, ecc=False, **kw)
    elif model == "1p1c_alias":
        kw2 = dict(kw); kw2["p1_prior"] = ALIAS
        m = p3model.Joint(d, n_planets=1, ecc=False, **kw2)
    elif model == "1p1c+toi":
        m = p3model.Joint(d, n_planets=1, ecc=False, toi=p3run.TOI, **kw)
    else: raise ValueError(model)
    return m

JOBS = []
for k in ("m32", "qp"):
    for mo in ("0p", "1p1c", "1p", "2p1c2c", "1p1c_alias", "1p1c+toi"):
        JOBS.append((k, mo, "serval"))
for k in ("white",):
    for mo in ("0p", "1p1c"):
        JOBS.append((k, mo, "serval"))
for k in ("m32",):
    for mo in ("0p", "1p1c"):
        JOBS.append((k, mo, "drs"))

if __name__ == "__main__":
    only = sys.argv[1:] or None
    out = {}
    fjson = "out/ladder.json"
    if os.path.exists(fjson): out = json.load(open(fjson))
    for k, mo, pl in JOBS:
        tag = f"{k}_{mo}_{pl}".replace("+", "-")
        if only and not any(o in tag for o in only): continue
        m = spec(k, mo, pl)
        r = p3run.run(m, tag, nlive=750, walks=30, nproc=2)
        s = p3run.summarize(r)
        out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                        ndim=r["ndim"], wall=r["wall"], ncall=r["ncall"],
                        loglmax=r["logl_max"], summary=s)
        json.dump(out, open(fjson, "w"), indent=1)
        print(f"{tag:<28} ndim={r['ndim']:<3} lnZ={r['logz']:9.2f} +- {r['logzerr']:.2f}"
              f"  pcorr={r['pcorr']:+.2f}  {r['wall']:.0f}s", flush=True)
