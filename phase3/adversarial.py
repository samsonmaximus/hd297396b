"""Step 3 of the work order: the adversarial shared-latent coupling on data-v1.

Rank-1 cross-covariance between the RV and dLW GPs: dLW pulls directly on the RV
noise *realisation*, not just on the hyperparameters. This is the configuration
most able to eat the planet, and it had never been run on the frozen rows.

val_a (GP amplitude pinned to zero) on data-v1 is not repeated here: it is
numerically the same model as the white-noise reference rungs already run in
ladder_v2.py (v2_*_white_gauss_1p1c), and is read off those.
"""
import os, json
import numpy as np
import p3data, p3model, p3run

FJ = "out/adversarial.json"
# The coupling is put on the ADOPTED noise model, so that the comparison against
# the adopted shared-hyperparameter pair changes only the coupling and tests the
# configuration the headline result actually rests on. Two further variants were
# dropped for compute: the coupling on the plain Matern model (superseded by the
# adopted one) and on the quasi-periodic kernel (the disfavoured noise model on
# the planet-free evidence, hence the weaker test). Each shared-latent run costs
# roughly an order of magnitude more than its shared-hyperparameter counterpart,
# because the covariance is 2N x 2N rather than two N x N blocks.
JOBS = [  # (pipeline, plus51, kernel, like, model, seeing)
    ("serval", True, "m32", "gauss", "0p",   True),
    ("serval", True, "m32", "gauss", "1p1c", True),
]

if __name__ == "__main__":
    out = json.load(open(FJ)) if os.path.exists(FJ) else {}
    for pipeline, plus51, kernel, like, model, see in JOBS:
        tag = (f"lat_{pipeline}_{'in' if plus51 else 'out'}_{kernel}_{like}_{model}"
               + ("_power" if see else ""))
        if tag in out:
            continue
        d = p3data.build_v1(pipeline, plus51)
        kw = {}
        if see:
            kw["seeing"] = dict(mode="power", s=p3data.seeing_v1(d["t"]))
        m = p3model.Joint(d, n_planets={"0p": 0, "1p1c": 1}[model], ecc=False,
                          gp=kernel, like=like, coupling="shared_latent",
                          p1_prior=p3run.P1_NARROW, p2_prior=p3run.P2_NARROW, **kw)
        r = p3run.run(m, tag, nlive=750, walks=30, nproc=2)
        s = p3run.summarize(r)
        out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                        ndim=r["ndim"], wall=r["wall"], loglmax=r["logl_max"],
                        n=d["n"], summary=s)
        json.dump(out, open(FJ, "w"), indent=1)
        extra = "".join(f"  {k}={s[k][1]:.3g}" for k in ("K1", "A_rv", "A_dlw", "eta3") if k in s)
        print(f"{tag:<40} ndim={r['ndim']:<3} lnZ={r['logz']:9.2f}+-{r['logzerr']:.2f}{extra}"
              f"  {r['wall']:.0f}s", flush=True)
    print("DONE adversarial", flush=True)
