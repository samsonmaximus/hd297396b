import numpy as np, json
from scipy.linalg import cho_factor, cho_solve
import p3data, p3model
from pscan import gls_scan, fit_0p
res={}
for pl in ("drs","serval"):
    d = p3data.build(pl, clip=4.0)
    S, hp = fit_0p(d)
    # remove the 4.26836 d signal by GLS before scanning
    c = cho_factor(S, lower=True)
    cols=[(d["epoch"]==k).astype(float) for k in range(d["epoch"].max()+1)]
    ph=2*np.pi*d["t"]/4.26836
    X=np.vstack([np.sin(ph),np.cos(ph)]+cols).T
    Si=cho_solve(c,X); b=np.linalg.solve(X.T@Si, Si.T@d["rv"]); r=d["rv"]-X@b
    T=d["t"].max()-d["t"].min()
    f=np.arange(1/2000.,1/10.,1./(20*T)); P=1/f
    dl=gls_scan(d["t"], r, d["erv"], d["epoch"], S, P)
    o=np.argsort(dl)[::-1]; seen=[]
    print(f"{pl}: residual scan (10-2000 d), GP l={hp['l']:.0f} d")
    for i in o:
        if all(abs(np.log(P[i])-np.log(s))>0.02 for s in seen):
            seen.append(P[i]); print(f"    P={P[i]:9.2f} d  dlnL={dl[i]:6.2f}")
        if len(seen)>=6: break
    res[pl]=[[float(P[i]),float(dl[i])] for i in o[:1]]+[[float(s),0] for s in seen]
json.dump(res, open("out/pscan2.json","w"), indent=1)
