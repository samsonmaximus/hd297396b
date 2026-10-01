"""
Phase 3, step 1 of the work order: every remaining paper rung, on `data-v1`.

Six rungs existed only on the Phase-3-native clipped set (alias, +TOI, the whole
QP branch, the white-noise reference through this same code, DRS Student-t
2p1c2c / drs_out, and serval_out Student-t). They are re-run here on the frozen
Phase-1 rows under fresh `v2_*` cache tags, because p3run.run() keys its pickle
on the tag alone and a new data selection under an old tag would silently return
the old result.
"""
import os, json, sys
import numpy as np
import p3data, p3model, p3run

ALIAS = (1.295, 1.308)      # 1-sidereal-day alias of P0, from the period scan
FJ = "out/ladder_v2.json"

# (pipeline, plus51, kernel, like, model)
JOBS = [
    # 1.3  white-noise reference THROUGH p3model -- the control that separates
    #      "the GP" from "a change of code, sampler and priors".
    ("serval", True,  "white", "gauss",    "0p"),
    ("serval", True,  "white", "gauss",    "1p1c"),
    ("drs",    True,  "white", "gauss",    "0p"),
    ("drs",    True,  "white", "gauss",    "1p1c"),
    # 1.1  the daily alias, by evidence
    ("serval", True,  "m32",   "gauss",    "1p1c_alias"),
    ("serval", True,  "m32",   "studentt", "1p1c_alias"),
    # 1.2  TOI-6263.01 mass limit
    ("serval", True,  "m32",   "gauss",    "1p1c+toi"),
    ("serval", True,  "m32",   "studentt", "1p1c+toi"),
    # 1.4  the QP branch -- the only direct constraint on P_rot
    ("serval", True,  "qp",    "gauss",    "0p"),
    ("serval", True,  "qp",    "gauss",    "1p1c"),
    ("serval", True,  "qp",    "gauss",    "2p1c2c"),
    # 1.6  serval_out Student-t (the +51-out column is Gaussian-only)
    ("serval", False, "m32",   "studentt", "0p"),
    ("serval", False, "m32",   "studentt", "1p1c"),
    ("serval", False, "m32",   "studentt", "2p1c2c"),
    # 1.5  DRS Student-t 2p1c2c, and the drs_out column
    ("drs",    True,  "m32",   "studentt", "2p1c2c"),
    ("drs",    False, "m32",   "gauss",    "0p"),
    ("drs",    False, "m32",   "gauss",    "1p1c"),
    ("drs",    False, "m32",   "gauss",    "2p1c2c"),
]


def make(pipeline, plus51, kernel, like, model):
    d = p3data.build_v1(pipeline, plus51)
    gp = None if kernel == "white" else kernel
    kw = dict(gp=gp, like=like, p1_prior=p3run.P1_NARROW,
              p2_prior=p3run.P2_NARROW, e_prior="kipping")
    if model == "0p":
        m = p3model.Joint(d, n_planets=0, **kw)
    elif model == "1p1c":
        m = p3model.Joint(d, n_planets=1, ecc=False, **kw)
    elif model == "1p":
        m = p3model.Joint(d, n_planets=1, ecc=True, **kw)
    elif model == "2p1c2c":
        m = p3model.Joint(d, n_planets=2, ecc=False, **kw)
    elif model == "1p1c_alias":
        kw2 = dict(kw); kw2["p1_prior"] = ALIAS
        m = p3model.Joint(d, n_planets=1, ecc=False, **kw2)
    elif model == "1p1c+toi":
        m = p3model.Joint(d, n_planets=1, ecc=False, toi=p3run.TOI, **kw)
    else:
        raise ValueError(model)
    return m, d


def tag_of(pipeline, plus51, kernel, like, model):
    return (f"v2_{pipeline}_{'in' if plus51 else 'out'}_{kernel}_{like}_{model}"
            .replace("+", "-"))


if __name__ == "__main__":
    only = sys.argv[1:] or None
    out = json.load(open(FJ)) if os.path.exists(FJ) else {}
    for job in JOBS:
        tag = tag_of(*job)
        if only and not any(o in tag for o in only):
            continue
        if tag in out:
            continue
        m, d = make(*job)
        nlive = 1000 if job[4] == "1p" else 750
        r = p3run.run(m, tag, nlive=nlive, walks=30, nproc=2)
        s = p3run.summarize(r)
        out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                        ndim=r["ndim"], wall=r["wall"], loglmax=r["logl_max"],
                        n=d["n"], nlive=nlive, job=list(job), summary=s)
        json.dump(out, open(FJ, "w"), indent=1)
        extra = ""
        for k in ("K1", "K2", "P2", "nu", "eta3", "K_toi", "P1"):
            if k in s:
                extra += f"  {k}={s[k][1]:.4g}"
        print(f"{tag:<40} ndim={r['ndim']:<3} lnZ={r['logz']:9.2f}+-{r['logzerr']:.2f}"
              f"  pcorr={r['pcorr']:+.2f}{extra}  {r['wall']:.0f}s", flush=True)
    print("DONE ladder_v2", flush=True)
