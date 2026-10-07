"""Regenerate the search counts of Sect. 2.2 and the archive-wide trials arithmetic of Sect. 4.2
from the stored outputs of the 6 September 2026 survey.

Inputs (this folder): rvbank_gls_survey_all_stars.csv (survey.py output, one row per searched star,
three prewhitened peaks each) and rvbank_vetting_results.csv (vet.py output for the 109 planet-like
peaks). Writes search_counts.log. Run: python search_counts.py
"""
import numpy as np, pandas as pd

out = []
def say(*a):
    line = " ".join(str(x) for x in a); print(line); out.append(line)

s = pd.read_csv("rvbank_gls_survey_all_stars.csv")
peaks = pd.concat([s[["target", f"P{k}", f"FAP{k}"]].rename(columns={f"P{k}": "P", f"FAP{k}": "FAP"})
                   for k in (1, 2, 3)], ignore_index=True)
say("searched stars:", len(s), "(Sect. 2.2: 1328)")
say("peaks (three per star):", len(peaks))
say("peaks with analytic FAP < 1e-4 in the stored table:", int((peaks.FAP < 1e-4).sum()),
    "(the 6 Sept report gives 659; the paper uses neither)")

v = pd.read_csv("rvbank_vetting_results.csv")
say("planet-like peaks:", len(v), "on", v.target.nunique(), "stars (Sect. 2.2: 109 on 98)")
say("  the four cuts hold for all of them: P < baseline/2", bool((v.P < v.span / 2).all()),
    "| M sin i < 13 M_Jup", bool((v.msini < 13).all()), f"(largest {v.msini.max():.1f})",
    "| K > 2.5 x median error", bool((v.K > 2.5 * v.med_err).all()), "| >= 25 nights", bool((v.nn >= 25).all()))
say("  and all have FAP < 1e-4:", bool((v.FAP < 1e-4).all()))

# Activity-indicator screen. The 6 Sept report gives the count (48) but its code was not stored.
# This rule reproduces the count exactly; it is a reconstruction, not the recorded code.
screen = (v.ind_minfap > 0.01) & (v.ind_maxr.abs() < 0.4)
say("activity screen (no indicator FAP < 1% at the period, all |r| < 0.4):", int(screen.sum()), "peaks on",
    v[screen].target.nunique(), "stars (Sect. 2.2: 48)")
say("  HD 297396 among them:", bool(v[screen].target.str.contains("297396").any()))

# Archive-wide trials factor (Sect. 4.2): the survey band started at 1.0 d, the paper's FAP band at 1.05 d.
band = (1 / 1.0 - 1 / 1000) / (1 / 1.05 - 1 / 1000)
say(f"band factor (1.0-1000 d over 1.05-1000 d, in frequency): {band:.4f}")
say(f"expected archive false alarms, 104 epochs: 1.4e-3 x {len(s)} x {band:.3f} = {1.4e-3 * len(s) * band:.2f}")
say(f"expected archive false alarms, 103 epochs: 6.7e-6 x {len(s)} x {band:.3f} = {6.7e-6 * len(s) * band:.2e}")
open("search_counts.log", "w").write("\n".join(out) + "\n")
