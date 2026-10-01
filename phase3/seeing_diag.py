"""Diagnostics for the seeing predictor, saved for the paper's macros.

The question a referee will ask about the seeing term is whether its empirical
support in these velocities comes from more than the one epoch it excuses. This
script answers it: per-bin scatter, the individual high-seeing epochs, and the
|RV|-seeing correlation with and without the discrepant epoch.
"""
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
import p3data

d = p3data.build_v1("serval", True)
s = p3data.seeing_v1(d["t"])
ok = np.isfinite(s)
rv = d["rv"]
i51 = int(np.argmax(np.abs(rv)))
hi = ok & (s > 1.5)

out = dict(
    n=int(len(rv)), n_seeing=int(ok.sum()), n_nodimm=int((~ok).sum()),
    n_hi=int(hi.sum()),
    hi_epochs=[dict(bjd=float(d["t"][i]), seeing=float(s[i]), rv=float(rv[i]),
                    err=float(d["erv"][i]), label=str(d["prog"][i]))
               for i in np.where(hi)[0]],
    bins=[dict(lo=a, hi=b, n=int((ok & (s >= a) & (s < b)).sum()),
               sd=float(np.std(rv[ok & (s >= a) & (s < b)])))
          for a, b in ((0, 0.8), (0.8, 1.1), (1.1, 1.5), (1.5, 9))],
    sd_nodimm=float(np.std(rv[~ok])),
    sd_hi_without_51=float(np.std(rv[hi & (np.arange(len(rv)) != i51)])),
)
r = np.abs(rv[ok]); ss = s[ok]
out["pearson_all"] = [float(x) for x in pearsonr(ss, r)]
out["spearman_all"] = [float(x) for x in spearmanr(ss, r)]
m = ok.copy(); m[i51] = False
out["pearson_no51"] = [float(x) for x in pearsonr(s[m], np.abs(rv[m]))]
out["spearman_no51"] = [float(x) for x in spearmanr(s[m], np.abs(rv[m]))]
json.dump(out, open("out/seeing_diag.json", "w"), indent=1)
for k, v in out.items():
    if k not in ("hi_epochs", "bins"):
        print(f"{k}: {v}")
