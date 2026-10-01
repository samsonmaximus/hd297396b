"""P_rot from the HARPS activity indicators, season by season.

The paper reports a global null: no indicator has significant power at P0, and
none is localised on any rotation period.  That global statement is the right
test for a coherent planetary signal and the WRONG test for rotation.  Stellar
activity is quasi-periodic: spot groups live a few rotations, so the phase of a
rotational signal is scrambled between observing seasons.  Coherently summing
18 yr of data destroys it.  The standard tool is per-season periodograms plus a
stacked (incoherently averaged) power spectrum.

Search window 10-60 d, which brackets the 22.6-35.9 d allowed by log R'HK.

Outputs season_prot.json and prints a table.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIB = 2457174.5

d = pd.read_csv(os.path.join(ROOT, "jitter_clean", "rvbank_with_moon.csv"))
# the frozen data-v1 cuts, as in phase4/indic.py
diff = d.RVdrsnzp_c - d.DRVmlcnzp_c
d["era"] = np.where(d.BJD < FIB, "pre", "post")
dd = diff - diff.groupby(d.era).transform("median")
d = d[~((dd.abs() > 5 * np.hypot(d.e_RVdrsnzp, d.e_DRVmlcnzp)) | (d.SNRDRS < 20))]
t = d.BJD.values

# --- seasons: split on gaps longer than 100 d --------------------------------
o = np.argsort(t)
d = d.iloc[o].reset_index(drop=True); t = d.BJD.values
gap = np.diff(t)
edges = np.concatenate([[0], np.where(gap > 100)[0] + 1, [len(t)]])
seasons = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
print("%d epochs, %d seasons (split at gaps > 100 d)" % (len(t), len(seasons)))

PMIN, PMAX = 10.0, 60.0
fr = np.linspace(1 / PMAX, 1 / PMIN, 4000)
P = 1 / fr


def gls(tt, yy, ee):
    """Weighted Delta chi^2 with a floating mean; returns power on the grid."""
    w = 1.0 / ee ** 2
    yp = yy - np.average(yy, weights=w)
    chi0 = np.sum(w * yp ** 2)
    out = np.empty(len(fr))
    for a in range(0, len(fr), 1000):
        f = fr[a:a + 1000][:, None]
        ph = 2 * np.pi * f * tt[None, :]
        C, S = np.cos(ph), np.sin(ph)
        C = C - (w * C).sum(1)[:, None] / w.sum()
        S = S - (w * S).sum(1)[:, None] / w.sum()
        cc = (w * C * C).sum(1); ss = (w * S * S).sum(1); cs = (w * C * S).sum(1)
        cy = (w * C * yp).sum(1); sy = (w * S * yp).sum(1)
        det = cc * ss - cs * cs
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        A = (ss * cy - cs * sy) / det
        B = (-cs * cy + cc * sy) / det
        out[a:a + 1000] = (A * cy + B * sy) / chi0        # normalised GLS power
    return np.nan_to_num(out)


IND = [("RHKp", "e_RHKp", "log R'HK"), ("Halpha", "e_Halpha", "Halpha"),
       ("NaD1", "e_NaD1", "Na I D1"), ("NaD2", "e_NaD2", "Na I D2"),
       ("dLW", "e_dLW", "dLW"), ("FWHMDRS", None, "CCF FWHM"),
       ("BIS", None, "BIS"), ("Contrast", None, "CCF contrast"),
       ("DRVmlcnzp_c", "e_DRVmlc", "RV (SERVAL)")]
MINEP = 8                      # a season needs at least this many epochs
rng = np.random.default_rng(11)
out = {}

print("\n%-13s %6s %8s %9s %9s  %s" % ("indicator", "nseas", "stackP", "stackpk",
                                       "FAP1%", "per-season peaks [d]"))
for col, ecol, lab in IND:
    y = d[col].values.astype(float)
    ok = np.isfinite(y)
    stack, npow, peaks, nsea = np.zeros(len(fr)), 0, [], 0
    for (a, b) in seasons:
        m = ok[a:b]
        if m.sum() < MINEP:
            continue
        tt, yy = t[a:b][m], y[a:b][m]
        if np.ptp(tt) < 25:                      # need a season long enough to see 10-60 d
            continue
        ee = (d[ecol].values[a:b][m].astype(float) if ecol
              else np.full(m.sum(), np.std(yy)))
        ee = np.where(np.isfinite(ee) & (ee > 0), ee, np.nanmedian(ee[np.isfinite(ee)]))
        ee = np.hypot(ee, 0.3 * np.std(yy))
        p = gls(tt, yy, ee)
        stack += p; npow += 1; nsea += 1
        peaks.append(float(P[int(np.argmax(p))]))
    if npow == 0:
        continue
    stack /= npow
    # bootstrap the stacked power: shuffle y within each season, restack
    mx = []
    for _ in range(300):
        s2, n2 = np.zeros(len(fr)), 0
        for (a, b) in seasons:
            m = ok[a:b]
            if m.sum() < MINEP:
                continue
            tt, yy = t[a:b][m], y[a:b][m]
            if np.ptp(tt) < 25:
                continue
            ee = (d[ecol].values[a:b][m].astype(float) if ecol
                  else np.full(m.sum(), np.std(yy)))
            ee = np.where(np.isfinite(ee) & (ee > 0), ee, np.nanmedian(ee[np.isfinite(ee)]))
            ee = np.hypot(ee, 0.3 * np.std(yy))
            s2 += gls(tt, rng.permutation(yy), ee); n2 += 1
        mx.append((s2 / n2).max())
    thr = float(np.percentile(mx, 99))
    ipk = int(np.argmax(stack))
    out[lab] = dict(nseasons=nsea, stack_peak_P=float(P[ipk]),
                    stack_peak_power=float(stack[ipk]), fap1=thr,
                    detected=bool(stack[ipk] > thr), season_peaks=peaks)
    print("%-13s %6d %8.2f %9.4f %9.4f  %s  %s"
          % (lab, nsea, P[ipk], stack[ipk], thr,
             " ".join("%.0f" % x for x in peaks),
             "<<< DETECT" if stack[ipk] > thr else ""))
    np.save(os.path.join(HERE, "stack_%s.npy" % col), np.vstack([P, stack]))

out["_meta"] = dict(n_epochs=int(len(t)), n_seasons=len(seasons),
                    season_spans=[[float(t[a]), float(t[b - 1]), int(b - a)]
                                  for a, b in seasons],
                    Pmin=PMIN, Pmax=PMAX, min_epochs_per_season=MINEP)
json.dump(out, open(os.path.join(HERE, "season_prot.json"), "w"), indent=1)
print("\nseason spans (BJD-2450000, n):")
for a, b in seasons:
    print("   %8.1f - %8.1f  (%3.0f d, %d epochs)"
          % (t[a] - 2450000, t[b - 1] - 2450000, t[b - 1] - t[a], b - a))
