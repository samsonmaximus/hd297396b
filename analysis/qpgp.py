# Quasi-periodic GP with the rotation period free over 10-100 d, fitted jointly to the velocities and dLW
# (shared P_rot, evolution time and harmonic complexity; separate amplitudes; per-label offsets profiled;
# per-label jitters). Maximum likelihood from many starts; then the periodogram weighted by that
# covariance and its FAP at P0 from draws of the same covariance.
import numpy as np, pandas as pd, json
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
G=pd.read_csv('frozen104.csv')
def prep(drop):
    g=G[np.abs(G.t-2454922.53)>0.2].reset_index(drop=True) if drop else G
    t=g.t.values-2455000; r=t[:,None]-t[None,:]; li=g.li.values; Lm=(li[:,None]==np.arange(4)[None,:]).astype(float)
    return g,t,r,li,Lm
def Kqp(r,A,lam,P,w): return A*A*np.exp(-r*r/(2*lam*lam)-np.sin(np.pi*r/P)**2/(2*w*w))
def gll(y,e,jit,A,lam,P,w,r,li,Lm):
    K=Kqp(r,A,lam,P,w); K[np.diag_indices_from(K)]+=e**2+jit[li]**2
    c=cho_factor(K,lower=True,check_finite=False)
    CiD=cho_solve(c,Lm,check_finite=False); Ciy=cho_solve(c,y,check_finite=False)
    p=np.linalg.solve(Lm.T@CiD,Lm.T@Ciy); res=y-Lm@p
    return -0.5*res@cho_solve(c,res,check_finite=False)-np.sum(np.log(np.diag(c[0]))),K
def fit(drop,seed=0):
    g,t,r,li,Lm=prep(drop); rv,erv,lw,elw=g.rv.values,g.erv.values,g.lw.values,g.elw.values
    # x: ln jr(4), ln Ar, ln jl(4), ln Al, ln lam, ln P, ln w
    bnd=[(np.log(0.01),np.log(50))]*4+[(np.log(0.01),np.log(50))]+[(np.log(0.01),np.log(50))]*4+[(np.log(0.01),np.log(500)),(np.log(10),np.log(1000)),(np.log(10),np.log(100)),(np.log(0.1),np.log(2))]
    def nll(x):
        jr,Ar,jl,Al,lam,P,w=np.exp(x[0:4]),np.exp(x[4]),np.exp(x[5:9]),np.exp(x[9]),np.exp(x[10]),np.exp(x[11]),np.exp(x[12])
        try: return -(gll(rv,erv,jr,Ar,lam,P,w,r,li,Lm)[0]+gll(lw,elw,jl,Al,lam,P,w,r,li,Lm)[0])
        except Exception: return 1e10
    rng=np.random.default_rng(seed); res=[]
    for P0s in [12,15,18,22,25,28,31,35,40,50,65,85]:
        x0=np.r_[np.log([4,5,4,4]),np.log(2),np.log([3,1,5,1]),np.log(10),np.log(rng.uniform(20,200)),np.log(P0s),np.log(0.5)]
        rr=minimize(nll,x0,method='L-BFGS-B',bounds=bnd); res.append(rr)
    res=sorted(res,key=lambda z:z.fun)
    b=res[0]; x=b.x
    out=dict(nll=b.fun,jr=np.exp(x[0:4]).tolist(),Ar=float(np.exp(x[4])),Al=float(np.exp(x[9])),lam=float(np.exp(x[10])),Prot=float(np.exp(x[11])),w=float(np.exp(x[12])),
             others=[(round(float(np.exp(z.x[11])),1),round(float(z.fun-b.fun),2),round(float(np.exp(z.x[4])),2)) for z in res[1:6]])
    return out,g,t,r,li,Lm
T=6536.0; freqs=np.arange(1/1000,1/1.05,1/(5*T))
def gls_pg(t,Lm,Lc,Y,fr):
    Lw=solve_triangular(Lc,Lm,lower=True); Q,_=np.linalg.qr(Lw); P=np.eye(len(t))-Q@Q.T
    yp=P@solve_triangular(Lc,Y,lower=True); out=[]
    for ch in np.array_split(fr,max(1,len(fr)//2000)):
        ph=2*np.pi*np.outer(t,ch); C=P@solve_triangular(Lc,np.cos(ph),lower=True); S_=P@solve_triangular(Lc,np.sin(ph),lower=True)
        cc=(C*C).sum(0); ss=(S_*S_).sum(0); cs=(C*S_).sum(0); det=cc*ss-cs**2; a=C.T@yp; b=S_.T@yp
        out.append(((ss*a*a-2*cs*a*b+cc*b*b)/det) if yp.ndim==1 else ((ss[:,None]*a*a-2*cs[:,None]*a*b+cc[:,None]*b*b)/det[:,None]))
    return np.concatenate(out)
R={}; rng=np.random.default_rng(9)
for drop in [False,True]:
    tag='103' if drop else '104'
    o,g,t,r,li,Lm=fit(drop)
    K=Kqp(r,o['Ar'],o['lam'],o['Prot'],o['w']); K[np.diag_indices_from(K)]+=g.erv.values**2+np.array(o['jr'])[li]**2
    Lc=np.linalg.cholesky(K)
    z=gls_pg(t,Lm,Lc,g.rv.values,freqs); k0=np.argmin(abs(freqs-1/4.26838)); zP=float(z[k0-3:k0+4].max())
    mx=np.concatenate([gls_pg(t,Lm,Lc,Lc@rng.normal(size=(len(t),1000)),freqs).max(0) for _ in range(5)])
    o.update(z_P0=zP,P_max=float(1/freqs[np.argmax(z)]),z_max=float(z.max()),n_exceed=int((mx>=zP).sum()),ndraw=len(mx))
    R[tag]=o; print(tag,json.dumps({k:(np.round(v,3).tolist() if isinstance(v,(list,float)) else v) for k,v in o.items()}),flush=True)
json.dump(R,open('qpgp.json','w'),indent=1)
