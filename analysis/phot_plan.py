# v15: photometric campaign needed to measure a 30-50 d rotation period (Sect. 8.4).
# Sinusoid of semi-amplitude A at P ~ U(30,50) d with random phase, N nightly points spread at random
# over a span T, Gaussian noise sigma per night. Detection: highest GLS peak over 5-100 d above the 1%
# false-alarm level (from noise-only draws with the same sampling) and within 10% of the true period.
import numpy as np, json
from astropy.timeseries import LombScargle
rng = np.random.default_rng(3050)
f = np.linspace(1/100, 1/5, 4000)
def run(N, T, sig, A, n=600):
    thr_draws = []; hits = 0
    for i in range(n):
        t = np.sort(rng.uniform(0, T, N))
        y0 = rng.normal(0, sig, N)
        thr_draws.append(LombScargle(t, y0).power(f).max())
    thr = np.percentile(thr_draws, 99)
    for i in range(n):
        t = np.sort(rng.uniform(0, T, N)); P = rng.uniform(30, 50)
        y = A*np.sin(2*np.pi*t/P + rng.uniform(0, 2*np.pi)) + rng.normal(0, sig, N)
        pw = LombScargle(t, y).power(f); k = np.argmax(pw)
        hits += (pw[k] > thr) and abs(1/f[k] - P) < 0.1*P
    return hits/n
out = {}
for sig in [2.0, 3.0]:
    for N, T in [(25, 60), (25, 90), (30, 90), (40, 120)]:
        for A in [3.0, 5.0]:
            r = run(N, T, sig, A); out[f'{sig}_{N}_{T}_{A}'] = r
            print(f'sigma {sig} mmag, {N} nights over {T} d, A = {A} mmag: recovered {r:.2f}', flush=True)
json.dump(out, open('phot_plan.json', 'w'), indent=1)
