"""Eccentric cell under the ADOPTED noise model (Matern GP + seeing term).

ecc_v1.py runs the eccentric cells on the plain Matern GP, for comparison with
Phase 1. The eccentricity quoted in the paper should come from the same noise
model as everything else it quotes, which is this.
"""
import os, json
import numpy as np
import p3data, p3model, p3run

FJ = "out/ecc_v1.json"
NLIVE = 1500
if __name__ == "__main__":
    out = json.load(open(FJ)) if os.path.exists(FJ) else {}
    tag = f"ecc_serval_in_gauss_1p_nl{NLIVE}_power"
    if tag not in out:
        d = p3data.build_v1("serval", True)
        m = p3model.Joint(d, n_planets=1, ecc=True, gp="m32", like="gauss",
                          e_prior="kipping", p1_prior=p3run.P1_NARROW,
                          p2_prior=p3run.P2_NARROW,
                          seeing=dict(mode="power", s=p3data.seeing_v1(d["t"])))
        r = p3run.run(m, tag, nlive=NLIVE, walks=30, nproc=2)
        s = p3run.summarize(r)
        e = r["samples"][:, m.pnames.index("ecc")]
        out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                        ndim=r["ndim"], wall=r["wall"], loglmax=r["logl_max"],
                        n=d["n"], nlive=NLIVE, summary=s,
                        e95=float(np.percentile(e, 95)), e90=float(np.percentile(e, 90)),
                        e68=float(np.percentile(e, 68)))
        json.dump(out, open(FJ, "w"), indent=1)
        print(f"{tag:<44} lnZ={r['logz']:9.2f}+-{r['logzerr']:.2f}  K={s['K1'][1]:.2f}"
              f"  e50={s['ecc'][1]:.3f}  e95={out[tag]['e95']:.3f}  {r['wall']:.0f}s", flush=True)
    print("DONE ecc_see", flush=True)
