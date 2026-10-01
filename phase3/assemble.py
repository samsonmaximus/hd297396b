"""Merge every Phase-3 run JSON into one canonical table.

Reads out/ladder_v1.json (the 14 original data-v1 rungs), out/ladder_v2.json
(step 1), out/seeing.json (step 2), out/adversarial.json (step 3) and
out/ecc_v1.json (step 4.1), normalises them onto one schema, and writes
out/phase3_final.json. Everything downstream -- report tables, figures and the
paper's numbers.tex -- reads only that file, so no number in the paper can drift
from the run that produced it.
"""
import json, os, glob
import numpy as np

SRC = {
    "ladder_v1": "out/ladder_v1.json",
    "ladder_v2": "out/ladder_v2.json",
    "seeing": "out/seeing.json",
    "adversarial": "out/adversarial.json",
    "ecc_v1": "out/ecc_v1.json",
    "verify": "out/verify_v1.json",
}

# Canonical description of every tag we may see.
def describe(tag, rec):
    """(pipeline, plus51, kernel, coupling, like, model, seeing_mode)"""
    p = dict(pipeline=None, plus51=None, kernel=None, coupling="shared_hypers",
             like=None, model=None, seeing=None, nlive=rec.get("nlive", 750))
    t = tag.split("_")
    if tag.startswith("v1_"):          # v1_<pipe>_<in|out>_<like>_<model>
        p.update(pipeline=t[1], plus51=(t[2] == "in"), kernel="m32",
                 like=t[3], model="_".join(t[4:]))
    elif tag.startswith("v2_"):        # v2_<pipe>_<in|out>_<kernel>_<like>_<model>
        p.update(pipeline=t[1], plus51=(t[2] == "in"), kernel=t[3],
                 like=t[4], model="_".join(t[5:]))
    elif tag.startswith("see_"):       # see_<pipe>_<in|out>_<like>_<model>_<mode>
        job = rec.get("job")
        if job:
            p.update(pipeline=job[0], plus51=job[1], kernel="m32", like=job[2],
                     model=job[3],
                     seeing=job[4] + ("" if job[5] is None else f"@{job[5]:.2f}"))
        else:                          # e.g. a ver_ repeat, which stores no job
            mode = t[5]
            if mode.startswith("thresh") and len(mode) > 6:
                mode = f"thresh@{int(mode[6:])/100:.2f}"
            p.update(pipeline=t[1], plus51=(t[2] == "in"), kernel="m32",
                     like=t[3], model=t[4], seeing=mode)
    elif tag.startswith("lat_"):       # lat_<pipe>_<in|out>_<kernel>_<like>_<model>[_power]
        rest = t[5:]
        seeing = None
        if rest and rest[-1] == "power":
            seeing = "power"; rest = rest[:-1]
        p.update(pipeline=t[1], plus51=(t[2] == "in"), kernel=t[3],
                 coupling="shared_latent", like=t[4], model="_".join(rest),
                 seeing=seeing)
    elif tag.startswith("ecc_"):       # ecc_<pipe>_<in|out>_<like>_1p_nl<N>[_power]
        p.update(pipeline=t[1], plus51=(t[2] == "in"), kernel="m32",
                 like=t[3], model="1p", nlive=int(t[5][2:]),
                 seeing=("power" if len(t) > 6 and t[6] == "power" else None))
    elif tag.startswith("ver_"):       # ver_<base>_s<seed>
        base = "_".join(t[1:-1])
        q = describe(base, rec)
        q["repeat"] = t[-1]
        return q
    else:
        raise ValueError(tag)
    return p


def load():
    runs = {}
    for src, f in SRC.items():
        if not os.path.exists(f):
            continue
        for tag, rec in json.load(open(f)).items():
            d = describe(tag, rec)
            d.update(tag=tag, source=src, logz=rec["logz"], logzerr=rec["logzerr"],
                     pcorr=rec["pcorr"], ndim=rec["ndim"], n=rec.get("n"),
                     wall=rec.get("wall"), summary=rec["summary"])
            for k in ("e95", "e90", "e68"):
                if k in rec:
                    d[k] = rec[k]
            runs[tag] = d
    return runs


def key(r):
    """Everything that must match for two runs to be comparable as a ladder pair."""
    return (r["pipeline"], r["plus51"], r["kernel"], r["coupling"], r["like"],
            r["seeing"], r.get("repeat"))


def dlnz(runs, tag_a, tag_b):
    """lnZ(a) - lnZ(b), both corrected to the wide search prior."""
    a, b = runs[tag_a], runs[tag_b]
    return (a["logz"] + a["pcorr"]) - (b["logz"] + b["pcorr"])


def raw_dlnz(runs, tag_a, tag_b):
    """lnZ(a) - lnZ(b) as sampled (narrow period window), the convention used in
    the Phase-3 report and in Phases 1/4."""
    return runs[tag_a]["logz"] - runs[tag_b]["logz"]


def pairs(runs, num="1p1c", den="0p"):
    """All (num - den) evidence differences that share every other setting."""
    by = {}
    for t, r in runs.items():
        by.setdefault((key(r), r["model"]), t)
    out = {}
    for (k, m), t in by.items():
        if m != num:
            continue
        tb = by.get((k, den))
        if tb is None:
            continue
        out[t] = dict(num=t, den=tb, dlnz=raw_dlnz(runs, t, tb),
                      dlnz_wide=dlnz(runs, t, tb),
                      err=float(np.hypot(runs[t]["logzerr"], runs[tb]["logzerr"])),
                      **{f"k_{i}": v for i, v in zip(
                          ("pipeline", "plus51", "kernel", "coupling", "like", "seeing", "repeat"), k)})
    return out


def kval(r):
    if "K1" not in r["summary"]:
        return None
    lo, med, hi = r["summary"]["K1"]
    return med, 0.5 * (hi - lo), (0.5 * (hi - lo)) / med


if __name__ == "__main__":
    runs = load()
    json.dump(runs, open("out/phase3_final.json", "w"), indent=1)
    print(f"{len(runs)} runs\n")
    hdr = f"{'tag':<42}{'pipe':<8}{'51':<4}{'kern':<7}{'coup':<15}{'like':<10}{'seeing':<14}{'model':<12}{'lnZ':>10}{'K':>16}"
    print(hdr); print("-" * len(hdr))
    for t in sorted(runs, key=lambda t: (runs[t]["pipeline"] or "", not runs[t]["plus51"],
                                         runs[t]["kernel"] or "", runs[t]["like"] or "",
                                         str(runs[t]["seeing"]), runs[t]["model"] or "")):
        r = runs[t]
        k = kval(r)
        ks = f"{k[0]:.2f}+-{k[1]:.2f} ({100*k[2]:.1f}%)" if k else ""
        print(f"{t:<42}{r['pipeline']:<8}{'in' if r['plus51'] else 'out':<4}{r['kernel']:<7}"
              f"{r['coupling']:<15}{r['like']:<10}{str(r['seeing']):<14}{r['model']:<12}"
              f"{r['logz']:10.2f}{ks:>16}")
    for num, den in (("1p1c", "0p"), ("2p1c2c", "1p1c"), ("1p", "1p1c")):
        pr = pairs(runs, num, den)
        if not pr:
            continue
        print(f"\n=== dlnZ({num} - {den}) ===")
        for t, v in sorted(pr.items()):
            print(f"  {t:<42} {v['dlnz']:+8.2f} +- {v['err']:.2f}   (wide-prior {v['dlnz_wide']:+8.2f})")
