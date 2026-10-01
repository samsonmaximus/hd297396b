"""Eccentric cells on `data-v1`, at nlive high enough to be quotable.

Phase 3 section 7.2 records that eccentric models are wrong by 6 nats in lnZ and
1.1 m/s in K at nlive=500 -- the K-e-omega degeneracy needs the live points. The
one eccentric run that exists on the frozen rows (v1_serval_in_gauss_1p, lnZ
-741.44) was made at nlive=750, i.e. below the project's own stated floor of
1000. It is superseded here.
"""
import os, json
import numpy as np
import p3data, p3model, p3run

FJ = "out/ecc_v1.json"
NLIVE = 1500
JOBS = [  # (pipeline, plus51, like)
    ("serval", True,  "gauss"),
    ("serval", True,  "studentt"),
    ("serval", False, "gauss"),
]

if __name__ == "__main__":
    out = json.load(open(FJ)) if os.path.exists(FJ) else {}
    for pipeline, plus51, like in JOBS:
        tag = f"ecc_{pipeline}_{'in' if plus51 else 'out'}_{like}_1p_nl{NLIVE}"
        if tag in out:
            continue
        d = p3data.build_v1(pipeline, plus51)
        m = p3model.Joint(d, n_planets=1, ecc=True, gp="m32", like=like,
                          e_prior="kipping", p1_prior=p3run.P1_NARROW,
                          p2_prior=p3run.P2_NARROW)
        r = p3run.run(m, tag, nlive=NLIVE, walks=30, nproc=2)
        s = p3run.summarize(r)
        ei = m.pnames.index("ecc")
        e = r["samples"][:, ei]
        out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                        ndim=r["ndim"], wall=r["wall"], loglmax=r["logl_max"],
                        n=d["n"], nlive=NLIVE, summary=s,
                        e95=float(np.percentile(e, 95)), e90=float(np.percentile(e, 90)),
                        e68=float(np.percentile(e, 68)))
        json.dump(out, open(FJ, "w"), indent=1)
        print(f"{tag:<40} ndim={r['ndim']:<3} lnZ={r['logz']:9.2f}+-{r['logzerr']:.2f}"
              f"  K={s['K1'][1]:.2f}  e50={s['ecc'][1]:.3f}  e95={out[tag]['e95']:.3f}"
              f"  {r['wall']:.0f}s", flush=True)
    print("DONE ecc_v1", flush=True)
