"""Phase 3 ladder re-run on the Phase-1 frozen data set (`data-v1`).

Phase 3's own builder uses a residual-based 4-sigma MAD clip; Phase 1 froze the
data under pre-registered, non-circular criteria and kept the +51 m/s night.
This script puts the correlated-noise ladder on those exact rows so the numbers
sit beside Phases 1 and 4, and answers Phase 4 section 6.2 directly: does a GP or
Student-t likelihood stop dlnZ swinging by 6 nats on one night?
"""
import os, json, sys
import numpy as np
import p3data, p3model, p3run

JOBS = [
    ("serval", True,  "gauss",    "0p"), ("serval", True,  "gauss",    "1p1c"),
    ("serval", True,  "gauss",    "2p1c2c"),
    ("serval", True,  "studentt", "0p"), ("serval", True,  "studentt", "1p1c"),
    ("serval", True,  "studentt", "2p1c2c"),
    ("drs",    True,  "gauss",    "0p"), ("drs",    True,  "gauss",    "1p1c"),
    ("drs",    True,  "studentt", "0p"), ("drs",    True,  "studentt", "1p1c"),
    ("serval", False, "gauss",    "0p"), ("serval", False, "gauss",    "1p1c"),
    ("serval", False, "gauss",    "2p1c2c"),
    ("serval", True,  "gauss",    "1p"),
]

def make(pipeline, plus51, like, model):
    d = p3data.build_v1(pipeline, plus51)
    kw = dict(gp="m32", like=like, p1_prior=p3run.P1_NARROW,
              p2_prior=p3run.P2_NARROW, e_prior="kipping")
    npl = {"0p": 0, "1p1c": 1, "1p": 1, "2p1c2c": 2}[model]
    return p3model.Joint(d, n_planets=npl, ecc=(model == "1p"), **kw), d

if __name__ == "__main__":
    fj = "out/ladder_v1.json"
    out = json.load(open(fj)) if os.path.exists(fj) else {}
    for pipeline, plus51, like, model in JOBS:
        tag = f"v1_{pipeline}_{'in' if plus51 else 'out'}_{like}_{model}"
        if tag in out: continue
        m, d = make(pipeline, plus51, like, model)
        r = p3run.run(m, tag, nlive=750, walks=30, nproc=2)
        s = p3run.summarize(r)
        out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                        ndim=r["ndim"], wall=r["wall"], loglmax=r["logl_max"],
                        n=d["n"], summary=s)
        json.dump(out, open(fj, "w"), indent=1)
        base = f"v1_{pipeline}_{'in' if plus51 else 'out'}_{like}_0p"
        dz = (r["logz"] - out[base]["logz"]) if base in out else float("nan")
        extra = ""
        if "K1" in s: extra += f"  K={s['K1'][1]:.2f}"
        if "K2" in s: extra += f"  P2={s['P2'][1]:.2f} K2={s['K2'][1]:.2f}"
        if "nu" in s: extra += f"  nu={s['nu'][1]:.1f}"
        print(f"{tag:<34} ndim={r['ndim']:<3} lnZ={r['logz']:9.2f}+-{r['logzerr']:.2f}"
              f"  dlnZ={dz:7.2f}{extra}  {r['wall']:.0f}s", flush=True)
