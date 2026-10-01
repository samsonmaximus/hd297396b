"""Step 4.2: run-to-run evidence scatter on the adopted pair.

dynesty's internal logzerr understates the reproducibility of lnZ. Every
evidence uncertainty quoted in the paper is the empirical scatter of repeated
runs of the same model with different random seeds and live-point counts, which
this script measures.
"""
import os, json
import numpy as np
import p3run
import figs_final as F

FJ = "out/verify_v1.json"
ADOPT = os.environ.get("ADOPT", "see_serval_in_gauss_1p1c_power")
ADOPT0 = os.environ.get("ADOPT0P", "see_serval_in_gauss_0p_power")
SEEDS = [(101, 750), (202, 1200), (303, 2000)]

if __name__ == "__main__":
    out = json.load(open(FJ)) if os.path.exists(FJ) else {}
    for base in (ADOPT0, ADOPT):
        for seed, nlive in SEEDS:
            tag = f"ver_{base}_s{seed}"
            if tag in out:
                continue
            d, m, th = F.rebuild(base)
            r = p3run.run(m, tag, nlive=nlive, walks=30, seed=seed, nproc=2)
            s = p3run.summarize(r)
            out[tag] = dict(logz=r["logz"], logzerr=r["logzerr"], pcorr=r["pcorr"],
                            ndim=r["ndim"], wall=r["wall"], loglmax=r["logl_max"],
                            n=d["n"], nlive=nlive, seed=seed, base=base, summary=s)
            json.dump(out, open(FJ, "w"), indent=1)
            print(f"{tag:<48} nlive={nlive} lnZ={r['logz']:9.2f}+-{r['logzerr']:.2f}"
                  f"  {r['wall']:.0f}s", flush=True)
    # report
    A = [v["logz"] for k, v in out.items() if v["base"] == ADOPT]
    B = [v["logz"] for k, v in out.items() if v["base"] == ADOPT0]
    n = min(len(A), len(B))
    if n >= 2:
        dd = np.array(A[:n]) - np.array(B[:n])
        print(f"dlnZ across {n} independent repeats: {np.round(dd,2)}  sd={dd.std(ddof=1):.2f}")
    print("DONE verify_v1", flush=True)
