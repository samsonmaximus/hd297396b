# Periodogram weighted by a posterior-typical covariance of the adopted model (medians of the phase-3
# posterior: A_rv = 5.2 m/s, ell = 1207 d, beta_rv = 1.74, jitters 8.25/6.03/3.39/4.34 m/s for
# 072/183/other/post; and an upper-range case A_rv = 9.3 m/s, ell = 750 d), and its FAP from draws.
import numpy as np, json
exec(open('gpfap.py').read().split("res={}")[0])
rng=np.random.default_rng(15); out={}
for tag,h in [('median',dict(Ar=5.19,ell=1207.,br=1.74,jr=[8.25,6.03,3.39,4.34])),('upper',dict(Ar=9.30,ell=750.,br=1.74,jr=[8.25,6.03,3.39,4.34]))]:
    for drop in [False,True]:
        g,t,li,Lm,K=cov(drop,h); Lc=np.linalg.cholesky(K)
        z=gls_pg(t,Lm,Lc,g.rv.values,freqs); k0=np.argmin(abs(freqs-1/4.26838)); zP=float(z[k0-3:k0+4].max())
        mx=np.concatenate([gls_pg(t,Lm,Lc,Lc@rng.normal(size=(len(t),1000)),freqs).max(0) for _ in range(3)])
        key=f"{tag}_{'103' if drop else '104'}"; out[key]=dict(z_P0=zP,P_max=float(1/freqs[np.argmax(z)]),n_exceed=int((mx>=zP).sum()),ndraw=len(mx))
        print(key,out[key],flush=True)
json.dump(out,open('gp_typical.json','w'),indent=1)
