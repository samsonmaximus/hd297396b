"""Adversarial checks on the 200.3 d second signal and on GP prior sensitivity."""
import numpy as np, json
from scipy.linalg import cho_factor, cho_solve
from astropy.timeseries import LombScargle
import p3data, p3model
from pscan import gls_scan, fit_0p

out = {}
d = p3data.build("serval", clip=4.0)

# --- 1. is 200.3 d in the ACTIVITY indicator, or only in RV?
print("[1] indicator power at the RV second-signal period")
import pandas as pd
df = pd.read_csv(p3data.CSV); df = df[df.Flag == 0]
night = np.floor(df.BJD.values - 0.5).astype(int)
for ind in ("dLW", "Halpha", "FWHMDRS", "Contrast", "BIS", "CRX", "RHKp", "NaD1", "NaD2"):
    y = df[ind].values.astype(float)
    m = np.isfinite(y)
    t = df.BJD.values[m]; yy = y[m] - np.median(y[m])
    ls = LombScargle(t, yy)
    fap = ls.false_alarm_level([0.01], method="baluev")[0]
    p200 = float(ls.power(np.array([1/200.26]))[0])
    p427 = float(ls.power(np.array([1/4.26836]))[0])
    fr = np.linspace(1/2000, 1/2.0, 100000); pw = ls.power(fr); top = 1/fr[np.argmax(pw)]
    print(f"   {ind:<9} n={m.sum():<4} pow@200.3={p200:.4f} ({p200/fap:5.2f}x FAP1%)"
          f"  pow@4.268={p427:.4f} ({p427/fap:5.2f}x)  strongest P={top:9.2f} d")
    out[ind] = dict(p200=p200, p427=p427, fap1=float(fap), top=float(top))

# --- 2. window function at 200 d and its 1-yr aliases
print("\n[2] window function")
ls = LombScargle(d["t"], np.ones_like(d["t"]), fit_mean=False, center_data=False)
for P in (200.26, 129.3, 182.6, 365.25, 4.26836, 1.301301):
    print(f"   window power at {P:9.3f} d = {float(ls.power(np.array([1/P]))[0]):.4f}")
a = 1/200.26
print(f"   1/200.26 -/+ 1/yr  ->  {1/abs(a-1/365.25):8.2f} d and {1/(a+1/365.25):8.2f} d")

# --- 3. GP prior sensitivity: does the 50 d floor on the Matern timescale matter?
print("\n[3] GP timescale prior floor")
import p3run
for lo in (10.0, 50.0):
    m = p3model.Joint(d, n_planets=1, gp="m32", p1_prior=p3run.P1_NARROW)
    # rebuild the timescale prior with a different floor
    m.priors[m.ix["gp_l"]] = p3model.LU(lo, 5000.0)
    r = p3run.run(m, f"floor{int(lo)}_m32_1p1c_serval", nlive=600, walks=25, nproc=1)
    s = p3run.summarize(r)
    print(f"   floor={lo:5.0f} d  lnZ={r['logz']:8.2f}  K={s['K1'][1]:.2f} "
          f"[{s['K1'][0]:.2f},{s['K1'][2]:.2f}]  gp_l={s['gp_l'][1]:.0f}  A_rv={s['A_rv'][1]:.2f}")
    out[f"floor{int(lo)}"] = dict(logz=r["logz"], K=s["K1"], gp_l=s["gp_l"])
json.dump(out, open("out/checks.json", "w"), indent=1, default=float)
