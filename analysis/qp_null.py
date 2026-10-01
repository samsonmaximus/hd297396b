# Calibration of the dLW quasi-periodicity: simulate aperiodic dLW (the maximum-likelihood squared-
# exponential covariance plus per-label jitters, same 103 epochs and errors), and for each realisation
# record the largest profile-likelihood gain of a quasi-periodic kernel over the aperiodic one, scanning
# 40 periods over 20-100 d (coarser than the 110-181-point grids used on the data, so the null is slightly
# conservative in favour of the real feature being significant only if it beats it clearly).
import numpy as np, pandas as pd, json, sys
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
G=pd.read_csv('frozen104.csv'); g=G[np.abs(G.t-2454922.53)>0.2].reset_index(drop=True)
t=g.t.values-2455000; r=t[:,None]-t[None,:]; li=g.li.values; Lm=(li[:,None]==np.arange(4)[None,:]).astype(float); e=g.elw.values
def gll(y,jit,K0):
    K=K0.copy(); K[np.diag_indices_from(K)]+=e**2+jit[li]**2
    c=cho_factor(K,lower=True,check_finite=False); CiD=cho_solve(c,Lm,check_finite=False); Ciy=cho_solve(c,y,check_finite=False)
    p=np.linalg.solve(Lm.T@CiD,Lm.T@Ciy); rr=y-Lm@p
    return -0.5*rr@cho_solve(c,rr,check_finite=False)-np.sum(np.log(np.diag(c[0])))
bnd=[(np.log(0.01),np.log(50))]*4+[(np.log(0.1),np.log(500)),(np.log(10),np.log(1000))]
def fit_se(y):
    f=lambda x:-gll(y,np.exp(x[:4]),np.exp(2*x[4])*np.exp(-r*r/(2*np.exp(2*x[5]))))
    return minimize(f,np.r_[np.log([3,1,5,1]),np.log(15),np.log(300)],method='L-BFGS-B',bounds=bnd)
def fit_qp(y,P,x0):
    f=lambda x:-gll(y,np.exp(x[:4]),np.exp(2*x[4])*np.exp(-r*r/(2*np.exp(2*x[5]))-np.sin(np.pi*r/P)**2/(2*np.exp(2*x[6]))))
    return -minimize(f,x0,method='L-BFGS-B',bounds=bnd+[(np.log(0.1),np.log(2))]).fun
se=fit_se(g.lw.values); x=se.x
jit=np.exp(x[:4]); A=np.exp(x[4]); lam=np.exp(x[5])
K=A*A*np.exp(-r*r/(2*lam*lam)); K[np.diag_indices_from(K)]+=e**2+jit[li]**2; Lc=np.linalg.cholesky(K)
Ps=np.exp(np.linspace(np.log(20),np.log(100),40)); rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 3)
NS=int(sys.argv[2]) if len(sys.argv)>2 else 30; out=[]
for k in range(NS):
    y=Lc@rng.normal(size=len(t))
    s0=fit_se(y); l0=-s0.fun
    best=max(fit_qp(y,P,np.r_[s0.x,np.log(1.0)]) for P in Ps)
    out.append(best-l0); print(k,round(out[-1],2),flush=True)
out=np.array(out)
res=dict(NS=NS,gains=out.tolist(),median=float(np.median(out)),p95=float(np.percentile(out,95)),max=float(out.max()),frac_ge_14=float((out>=14.0).mean()),
         se_fit=dict(jit=jit.tolist(),A=A,lam=lam))
print(res,flush=True); json.dump(res,open(f'qp_null_{sys.argv[1] if len(sys.argv)>1 else 3}.json','w'),indent=1)
