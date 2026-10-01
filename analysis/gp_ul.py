# Posterior upper limits on the velocity amplitude of the correlated-noise terms (uniform amplitude prior
# 0-20 m/s), for the adopted Matern-3/2 model and for a quasi-periodic model with P_rot free over 10-100 d.
# Joint velocity + dLW likelihood as in gpns.py; offsets marginalised analytically (flat priors).
import numpy as np, pandas as pd, emcee, json, sys
from scipy.linalg import cho_factor, cho_solve
G=pd.read_csv('frozen104.csv')
res={}
for drop in [False,True]:
    g=G[np.abs(G.t-2454922.53)>0.2].reset_index(drop=True) if drop else G
    t=g.t.values-2455000; r=t[:,None]-t[None,:]; ar=np.abs(r); li=g.li.values; Lm=(li[:,None]==np.arange(4)[None,:]).astype(float)
    rv,erv,lw,elw=g.rv.values,g.erv.values,g.lw.values,g.elw.values
    def gll(y,e,jit,Kc):
        K=Kc.copy(); K[np.diag_indices_from(K)]+=e**2+jit[li]**2
        c=cho_factor(K,lower=True,check_finite=False)
        CiD=cho_solve(c,Lm,check_finite=False); Ciy=cho_solve(c,y,check_finite=False); F=Lm.T@CiD
        p=np.linalg.solve(F,Lm.T@Ciy); rr=y-Lm@p
        return -0.5*rr@cho_solve(c,rr,check_finite=False)-np.sum(np.log(np.diag(c[0])))-0.5*np.linalg.slogdet(F)[1]
    for kern in ['matern','qp']:
        nd=13 if kern=='qp' else 11
        def lnp(x):
            jr=np.exp(x[0:4]); Ar=x[4]; jl=np.exp(x[5:9]); Al=np.exp(x[9])
            if not (0<=Ar<=20 and np.all((x[0:4]>np.log(0.01))&(x[0:4]<np.log(50))) and np.all((x[5:9]>np.log(0.01))&(x[5:9]<np.log(50))) and np.log(0.1)<x[9]<np.log(500)): return -np.inf
            if kern=='matern':
                ell=np.exp(x[10])
                if not (np.log(50)<x[10]<np.log(5000)): return -np.inf
                s3=np.sqrt(3)*ar/ell; k=(1+s3)*np.exp(-s3)
            else:
                lam,P,w=np.exp(x[10]),np.exp(x[11]),np.exp(x[12])
                if not (np.log(10)<x[10]<np.log(1000) and np.log(10)<x[11]<np.log(100) and np.log(0.1)<x[12]<np.log(2)): return -np.inf
                k=np.exp(-r*r/(2*lam*lam)-np.sin(np.pi*r/P)**2/(2*w*w))
            try: return gll(rv,erv,jr,Ar*Ar*k)+gll(lw,elw,jl,Al*Al*k)
            except Exception: return -np.inf
        rng=np.random.default_rng(5); nw=40
        x0=np.r_[np.log([5,5,4,4]),1.0,np.log([3,1,5,1]),np.log(15),np.log(700)] if kern=='matern' else np.r_[np.log([5,5,4,4]),1.0,np.log([3,1,5,1]),np.log(15),np.log(300),np.log(40),np.log(1.0)]
        p0=x0+rng.normal(size=(nw,nd))*0.05; p0[:,4]=np.abs(rng.uniform(0.2,3,nw))
        sam=emcee.EnsembleSampler(nw,nd,lnp); sam.run_mcmc(p0,6000,progress=False)
        ch=sam.get_chain(discard=2000,flat=True)
        A=ch[:,4]; out=dict(A_rv_95=float(np.percentile(A,95)),A_rv_med=float(np.median(A)))
        if kern=='qp': out['Prot_16_50_84']=np.percentile(np.exp(ch[:,11]),[16,50,84]).tolist()
        res[f"{'103' if drop else '104'}_{kern}"]=out; print(('103' if drop else '104'),kern,out,flush=True)
json.dump(res,open('gp_ul.json','w'),indent=1)
