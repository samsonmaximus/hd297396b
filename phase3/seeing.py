"""Step 2 of the work order: the seeing-dependent noise term.

Read-out and form selection are fixed in PREREGISTRATION_step2.md, written
before any run here was executed.
"""
import os, json, sys
import numpy as np
import p3data, p3model, p3run

FJ = "out/seeing.json"

STAGE1 = [  # (pipeline, plus51, like, model, mode, thr)
    ("serval", True, "gauss", "0p",   "power",  None),
    ("serval", True, "gauss", "0p",   "hinge",  None),
    ("serval", True, "gauss", "0p",   "thresh", 1.50),
    ("serval", True, "gauss", "1p1c", "power",  None),
    ("serval", True, "gauss", "1p1c", "hinge",  None),
    ("serval", True, "gauss", "1p1c", "thresh", 1.50),
]


def make(pipeline, plus51, like, model, mode, thr):
    d = p3data.build_v1(pipeline, plus51)
    s = p3data.seeing_v1(d["t"])
    see = dict(mode=mode, s=s)
    if thr is not None:
        see["thr"] = thr
    kw = dict(gp="m32", like=like, p1_prior=p3run.P1_NARROW,
              p2_prior=p3run.P2_NARROW, e_prior="kipping", seeing=see)
    npl = {"0p": 0, "1p1c": 1, "1p": 1, "2p1c2c": 2}[model]
    return p3model.Joint(d, n_planets=npl, ecc=(model == "1p"), **kw), d


def tag_of(pipeline, plus51, like, model, mode, thr):
    t = f"see_{pipeline}_{'in' if plus51 else 'out'}_{like}_{model}_{mode}"
    if thr is not None and mode == "thresh":
        t += f"{thr:.2f}".replace(".", "")
    return t


def run_jobs(jobs):
    out = json.load(open(FJ)) if os.path.exists(FJ) else {}
    for job in jobs:
        tag = tag_of(*job)
        if tag in out:
            continue
        m, d = make(*job)
        r = p3run.run(m, tag, nlive=750, walks=30, nproc=2)
        s = p3run.summarize(r)
        out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                        ndim=r["ndim"], wall=r["wall"], loglmax=r["logl_max"],
                        n=d["n"], job=list(job), summary=s)
        json.dump(out, open(FJ, "w"), indent=1)
        extra = ""
        for k in ("K1", "gam_see", "q_see", "s_bad", "s_nodimm", "s_rv0", "nu", "K2", "P2"):
            if k in s:
                extra += f"  {k}={s[k][1]:.3g}"
        print(f"{tag:<40} ndim={r['ndim']:<3} lnZ={r['logz']:9.2f}+-{r['logzerr']:.2f}{extra}"
              f"  {r['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    if which == "stage1":
        run_jobs(STAGE1)
    else:
        run_jobs(json.load(open(which)))
    print(f"DONE seeing {which}", flush=True)
