# Maximum-likelihood hyperparameters of the adopted noise model (joint RV + dLW Matern-3/2 GP,
# shared length scale, per-label offsets [profiled analytically by GLS], jitters, error scalings).
import numpy as np, pandas as pd, sys, json
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
G=pd.read_csv('frozen104.csv')
def prep(drop):
    g=G[np.abs(G.t-2454922.53)>0.2].reset_index(drop=True) if drop else G
    t=g.t.values-2455000; r=np.abs(t[:,None]-t[None,:]); li=g.li.values
    Lm=(li[:,None]==np.arange(4)[None,:]).astype(float)
    return g,t,r,li,Lm
def gll_prof(y,e,jit,beta,A,ell,r,li,Lm,X=None):
    s3=np.sqrt(3)*r/ell; K=A*A*(1+s3)*np.exp(-s3)
    K[np.diag_indices_from(K)]+=(beta*e)**2+jit[li]**2
    c=cho_factor(K,lower=True,check_finite=False)
    D=Lm if X is None else np.hstack([Lm,X])
    CiD=cho_solve(c,D,check_finite=False); Ciy=cho_solve(c,y,check_finite=False)
    p=np.linalg.solve(D.T@CiD,D.T@Ciy); res=y-D@p
    ll=-0.5*res@cho_solve(c,res,check_finite=False)-np.sum(np.log(np.diag(c[0])))-0.5*len(y)*np.log(2*np.pi)
    return ll,p
def fit(drop,P=None,nstart=5,seed=0):
    g,t,r,li,Lm=prep(drop); rv,erv,lw,elw=g.rv.values,g.erv.values,g.lw.values,g.elw.values
    X=None if P is None else np.c_[np.cos(2*np.pi*t/P),np.sin(2*np.pi*t/P)]
    def unpack(x): return np.exp(x[0:4]),np.exp(x[4]),np.exp(x[5]),np.exp(x[6:10]),np.exp(x[10]),np.exp(x[11]),np.exp(x[12])
    def nll(x):
        jr,br,Ar,jl,bl,Al,ell=unpack(x)
        try:
            a,_=gll_prof(rv,erv,jr,br,Ar,ell,r,li,Lm,X); b,_=gll_prof(lw,elw,jl,bl,Al,ell,r,li,Lm)
            return -(a+b)
        except Exception: return 1e10
    bnd=[(np.log(0.01),np.log(50))]*4+[(np.log(0.1),np.log(10)),(np.log(0.1),np.log(50))]+[(np.log(0.01),np.log(50))]*4+[(np.log(0.1),np.log(10)),(np.log(0.1),np.log(500)),(np.log(50),np.log(5000))]
    rng=np.random.default_rng(seed); best=None
    for k in range(nstart):
        x0=np.r_[np.log(rng.uniform(1,5,4)),rng.normal(0,0.2),np.log(rng.uniform(1,6)),np.log(rng.uniform(1,10,4)),rng.normal(0,0.2),np.log(rng.uniform(3,20)),np.log(rng.uniform(100,3000))]
        res=minimize(nll,x0,method='L-BFGS-B',bounds=bnd)
        if best is None or res.fun<best.fun: best=res
    jr,br,Ar,jl,bl,Al,ell=unpack(best.x)
    _,p=gll_prof(rv,erv,jr,br,Ar,ell,r,li,Lm,X)
    out=dict(jr=jr.tolist(),br=br,Ar=Ar,jl=jl.tolist(),bl=bl,Al=Al,ell=ell,offr=p[:4].tolist(),lnL=-best.fun,x=best.x.tolist())
    if P: out['K']=float(np.hypot(*p[4:6]))
    return out
if __name__=='__main__':
    res={}
    for drop in [False,True]:
        for P in [None,4.26837]:
            tag=('103' if drop else '104')+('_1p' if P else '_0p')
            res[tag]=fit(drop,P); print(tag,{k:(np.round(v,3).tolist() if isinstance(v,list) else round(v,3)) for k,v in res[tag].items() if k!='x'},flush=True)
    json.dump(res,open('gpmap.json','w'),indent=1)
